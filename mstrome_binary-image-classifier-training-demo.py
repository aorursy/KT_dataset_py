import os, sys, random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
%matplotlib inline
import matplotlib.pyplot as plt
from torchvision.transforms import Normalize
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
import torchvision.models as models
#https://www.kaggle.com/humananalog/deepfakes-inference-demo
sys.path.insert(0, "/kaggle/input/deepfakes-inference-demo")
from helpers.read_video_1 import VideoReader
from helpers.face_extract_1 import FaceExtractor
#https://www.kaggle.com/humananalog/blazeface-pytorch
sys.path.insert(0, "/kaggle/input/blazeface-pytorch")
from blazeface import BlazeFace
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#The size of the images to bed fed in to the network
image_size = 224
batch_size = 64
input_size = 224

#mean and std of the RGB channels in training set, precomputed
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

#All the large image networks such as Resnet expect normalized data
normalize_transform = Normalize(mean, std)


#Here is where you set up the path to import the faces_224 data and metadata
crops_dir = "../input/deepfake-faces/faces_224"
metadata_df = pd.read_csv("../input/deepfake-faces/metadata.csv")
test_dir = "/kaggle/input/deepfake-detection-challenge/train_sample_videos/"
test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])
#Vertical flip with 25% chance
def random_vflip(img, p=0.25):
    """Random horizontal flip."""
    if random.random() < p:
        return cv2.flip(img, 0)
    else:
        return img

#90 degree rotate with 25% chance
def random_rotate(img, p=0.25):
    if random.random() < p:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:
        return img
    
#Horizontal flip with 25% chance
def random_hflip(img, p=0.25):
    """Random horizontal flip."""
    if random.random() < p:
        return cv2.flip(img, 1)
    else:
        return img
def load_image_and_label(filename, cls, crops_dir, image_size, augment):
    """Loads an image into a tensor. Also returns its label."""
    img = cv2.imread(os.path.join(crops_dir, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if augment:
        #our code
        img = random_hflip(img)
        img = random_vflip(img)
        img = random_rotate(img)

    img = cv2.resize(img, (image_size, image_size))

    img = torch.tensor(img).permute((2, 0, 1)).float().div(255)
    img = normalize_transform(img)

    target = 1 if cls == "FAKE" else 0
    return img, target

class VideoDataset(Dataset):
    """Face crops dataset.

    Arguments:
        crops_dir: base folder for face crops
        df: Pandas DataFrame with metadata
        split: if "train", applies data augmentation
        image_size: resizes the image to a square of this size
        sample_size: evenly samples this many videos from the REAL
            and FAKE subfolders (None = use all videos)
        seed: optional random seed for sampling
    """
    def __init__(self, crops_dir, df, split, image_size, sample_size=None, seed=None):
        self.crops_dir = crops_dir
        self.split = split
        self.image_size = image_size
        
        if sample_size is not None:
            real_df = df[df["label"] == "REAL"]
            fake_df = df[df["label"] == "FAKE"]
            sample_size = np.min(np.array([sample_size, len(real_df), len(fake_df)]))
            print("%s: sampling %d from %d real videos" % (split, sample_size, len(real_df)))
            print("%s: sampling %d from %d fake videos" % (split, sample_size, len(fake_df)))
            real_df = real_df.sample(sample_size, random_state=seed)
            fake_df = fake_df.sample(sample_size, random_state=seed)
            self.df = pd.concat([real_df, fake_df])
        else:
            self.df = df

        num_real = len(self.df[self.df["label"] == "REAL"])
        num_fake = len(self.df[self.df["label"] == "FAKE"])
        print("%s dataset has %d real videos, %d fake videos" % (split, num_real, num_fake))

    def __getitem__(self, index):
        row = self.df.iloc[index]
        filename = row["videoname"][:-4] + ".jpg"
        cls = row["label"]
        return load_image_and_label(filename, cls, self.crops_dir, 
                                    self.image_size, self.split == "train")
    def __len__(self):
        return len(self.df)

    
def make_splits(crops_dir, metadata_df, frac):
    # Make a validation split. Sample a percentage of the real videos, 
    # and also grab the corresponding fake videos.
    real_rows = metadata_df[metadata_df["label"] == "REAL"]
    real_df = real_rows.sample(frac=frac, random_state=666)
    fake_df = metadata_df[metadata_df["original"].isin(real_df["videoname"])]
    val_df = pd.concat([real_df, fake_df])

    # The training split is the remaining videos.
    train_df = metadata_df.loc[~metadata_df.index.isin(val_df.index)]

    return train_df, val_df

def create_data_loaders(crops_dir, metadata_df, image_size, batch_size, num_workers):
    train_df, val_df = make_splits(crops_dir, metadata_df, frac=0.05)

    train_dataset = VideoDataset(crops_dir, train_df, "train", image_size, sample_size=10000)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)

    val_dataset = VideoDataset(crops_dir, val_df, "val", image_size, sample_size=500, seed=1234)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

train_loader, val_loader = create_data_loaders(crops_dir, metadata_df, image_size, 
                                               batch_size, num_workers=2)

def evaluate(net, data_loader, device, silent=False):
    net.train(False)

    bce_loss = 0
    total_examples = 0

    with tqdm(total=len(data_loader), desc="Evaluation", leave=False, disable=silent) as pbar:
        for batch_idx, data in enumerate(data_loader):
            with torch.no_grad():
                batch_size = data[0].shape[0]
                x = data[0].to(device)
                y_true = data[1].to(device).float()

                y_pred = net(x)
                y_pred = y_pred.squeeze()

                bce_loss += F.binary_cross_entropy_with_logits(y_pred, y_true).item() * batch_size

            total_examples += batch_size
            pbar.update()

    bce_loss /= total_examples

    if silent:
        return bce_loss
    else:
        print("BCE: %.4f" % (bce_loss))
        

def fit(epochs):
    global history, iteration, epochs_done, lr

    with tqdm(total=len(train_loader), leave=False) as pbar:
        for epoch in range(epochs):
            pbar.reset()
            pbar.set_description("Epoch %d" % (epochs_done + 1))
            
            bce_loss = 0
            total_examples = 0

            net.train(True)

            for batch_idx, data in enumerate(train_loader):
                batch_size = data[0].shape[0]
                x = data[0].to(gpu)
                y_true = data[1].to(gpu).float()
                
                optimizer.zero_grad()

                y_pred = net(x)
                y_pred = y_pred.squeeze()
                
                loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
                loss.backward()
                optimizer.step()
                
                batch_bce = loss.item()
                bce_loss += batch_bce * batch_size
                history["train_bce"].append(batch_bce)

                total_examples += batch_size
                iteration += 1
                pbar.update()

            bce_loss /= total_examples
            epochs_done += 1

            print("Epoch: %3d, train BCE: %.4f" % (epochs_done, bce_loss))

            val_bce_loss = evaluate(net, val_loader, device=gpu, silent=True)
            history["val_bce"].append(val_bce_loss)
            
            print("              val BCE: %.4f" % (val_bce_loss))

            print("")
model = models.resnext50_32x4d(pretrained=True)
#model = models.wide_resnet50_2(pretrained=True)
model.fc = nn.Linear(2048, 1)
net = model.to(gpu)
def freeze_until(net, param_name):
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name
#to do, play with freezing
freeze_until(net, "layer4.0.conv1.weight")
#freeze_until(net, "layer3.0.conv1.weight")
#freeze_until(net, "fc.weight")

#print all unfreezed layers
print([k for k,v in net.named_parameters() if v.requires_grad])

evaluate(net, val_loader, device=gpu)

lr = 0.01
wd = 0.0

history = { "train_bce": [], "val_bce": [] }
iteration = 0
epochs_done = 0

optimizer = torch.optim.Adam(net.parameters(), lr=.05, weight_decay=wd)

fit(3)

optimizer = torch.optim.Adam(net.parameters(), lr=.01, weight_decay=wd)

fit(3)

optimizer = torch.optim.Adam(net.parameters(), lr=.001, weight_decay=wd)

fit(3)

optimizer = torch.optim.Adam(net.parameters(), lr=.0001, weight_decay=wd)

fit(3)

plt.plot(history["train_bce"])
plt.plot(history["val_bce"])
plt.plot(history["val_bce"])
facedet = BlazeFace().to(gpu)
facedet.load_weights("/kaggle/input/blazeface-pytorch/blazeface.pth")
facedet.load_anchors("/kaggle/input/blazeface-pytorch/anchors.npy")
_ = facedet.train(False)

frames_per_video = 17

video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn, facedet)

def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size

    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized


def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)


def predict_on_video(video_path, batch_size):
    try:
        # Find the faces for N frames in the video.
        faces = face_extractor.process_video(video_path)

        # Only look at one face per frame.
        face_extractor.keep_only_best_face(faces)
        
        if len(faces) > 0:
            # NOTE: When running on the CPU, the batch size must be fixed
            # or else memory usage will blow up. (Bug in PyTorch?)
            x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)

            # If we found any faces, prepare them for the model.
            n = 0
            for frame_data in faces:
                for face in frame_data["faces"]:
                    # Resize to the model's required input size.
                    # We keep the aspect ratio intact and add zero
                    # padding if necessary.                    
                    resized_face = isotropically_resize_image(face, input_size)
                    resized_face = make_square_image(resized_face)

                    if n < batch_size:
                        x[n] = resized_face
                        n += 1
                    else:
                        print("WARNING: have %d faces but batch size is %d" % (n, batch_size))
                    

            if n > 0:
                x = torch.tensor(x, device=gpu).float()

                # Preprocess the images.
                x = x.permute((0, 3, 1, 2))

                for i in range(len(x)):
                    x[i] = normalize_transform(x[i] / 255.)

                # Make a prediction, then take the average.
                with torch.no_grad():
                    y_pred = net(x)
                    y_pred = torch.sigmoid(y_pred.squeeze())
                    return y_pred[:n].mean().item()

    except Exception as e:
        print("Prediction error on video %s: %s" % (video_path, str(e)))

    return 0.5

def predict_on_video_set(videos, num_workers):
    def process_file(i):
        filename = videos[i]
        y_pred = predict_on_video(os.path.join(test_dir, filename), batch_size=frames_per_video)
        return y_pred

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        predictions = ex.map(process_file, range(len(videos)))

    return list(predictions)
def h(name):
    if name == "FAKE":
        return 1.0
    else:
        return 0.0

c = 30
i = 0
j = 0
right = 0
lab = []
pred = []
while j < c:
    a = test_videos[i:i+1][0]
    if len(np.argwhere(metadata_df["videoname"]==a)) == 0:
        i += 1
        continue
    a = np.argwhere(metadata_df["videoname"]==a).item()
    a = metadata_df.iloc[a, :]
    if a["label"] == "FAKE":
        i += 1
        continue
    predictions = predict_on_video_set(test_videos[i:i+1], num_workers=4)
    #print(a["videoname"], a["label"], predictions[0])
    lab.append(h(a["label"]))
    pred.append(predictions[0])
    if np.rint(predictions[0]) == h(a["label"]):
        right +=1
    i += 1
    j += 1
    print(j)
i = 0
j = 0
while j < c:
    print(j)
    a = test_videos[i:i+1][0]
    if len(np.argwhere(metadata_df["videoname"]==a)) == 0:
        i+=1
        continue
    a = np.argwhere(metadata_df["videoname"]==a).item()
    a = metadata_df.iloc[a, :]
    if a["label"] == "REAL":
        i+=1
        continue
    predictions = predict_on_video_set(test_videos[i:i+1], num_workers=4)
    #print(a["videoname"], a["label"], predictions[0])
    lab.append(h(a["label"]))
    pred.append(predictions[0])
    if np.rint(predictions[0]) == h(a["label"]):
        right +=1
    i += 1
    j += 1
    print(j)
print("accuracy", right/(c*2))
for i in range(len(pred)):
    a = 0.0
    if pred[i] >= .1:
        a = 1.0
    if a == lab[i]:
        #print(pred[i], t, lab[i])
        right +=1.0
print(best_right/len(pred), best_t)