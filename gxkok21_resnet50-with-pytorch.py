%matplotlib inline

import os
import math
import copy

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import torch
import torchvision

# import skimage
from skimage.io import imread

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
EPOCHS = 5
USE_GPU = True
labels_df = pd.read_csv("../input/train_labels.csv")
labels_df.head()
labels_df["label"].value_counts().plot(kind="pie")
train_indices, test_indices = train_test_split(labels_df.index, test_size=0.25)
train_indices.shape, test_indices.shape

class HistopathologicCancerDataset(torch.utils.data.Dataset):
    """
    This is our custom dataset class which will load the images, perform transforms on them,
    and load their corresponding labels.
    """
    
    def __init__(self, img_dir, labels_csv_file=None, transform=None):
        self.img_dir = img_dir
        
        if labels_csv_file:
            self.labels_df = pd.read_csv(labels_csv_file)
        else:
            self.images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".tif")]
            
        self.transform = transform
        
    def __getitem__(self, idx):
        try:
            img_path = os.path.join(
                self.img_dir,
                "{}.tif".format(self.labels_df.iloc[idx, 0])
            )
        except AttributeError:
            img_path = self.images[idx]

#         print("img_path:", img_path)
        img = imread(img_path)
        
        if self.transform:
            img = self.transform(img)
        
        sample = {
            "image": img,
        }
        try:
            sample["label"] = self.labels_df.loc[idx, "label"]
            sample["id"] = self.labels_df.loc[idx, "id"]
        except AttributeError:
            sample["id"] = os.path.basename(self.images[idx]).replace(".tif", "")
        
        return sample
    
    def __len__(self):
        try:
            return self.labels_df.shape[0]
        except AttributeError:
            return len(self.images)
transform_pipe = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(), # Convert np array to PILImage
    
    # Resize image to 224 x 224 as required by most vision models
    torchvision.transforms.Resize(
        size=(224, 224)
    ),
    
    # Convert PIL image to tensor with image values in [0, 1]
    torchvision.transforms.ToTensor(),
    
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
train_data = HistopathologicCancerDataset(
    img_dir="../input/train/",
    labels_csv_file="../input/train_labels.csv",
    transform=transform_pipe
)
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=64,
    sampler=torch.utils.data.SubsetRandomSampler(
        train_indices
    )
#     shuffle=True,
#     num_workers=8
)
test_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=64,
    sampler=torch.utils.data.SubsetRandomSampler(
        test_indices
    )
#     shuffle=True,
#     num_workers=8
)
dataloaders = {
    "train": train_loader,
    "test": test_loader
}

class Flatten(torch.nn.Module):
    """
    Custom flatten module like what is available in Keras.
    """
    
    def forward(self, input):
        return input.view(input.size(0), -1)
# model = torch.nn.Sequential(
#     torch.nn.Conv2d(
#         in_channels=3,
#         out_channels=8,
#         kernel_size=3,
#     ),
#     torch.nn.MaxPool2d(
#         kernel_size=2
#     ),
#     torch.nn.ReLU(),
    
#     torch.nn.Conv2d(
#         in_channels=8,
#         out_channels=16,
#         kernel_size=3
#     ),
#     torch.nn.MaxPool2d(
#         kernel_size=2
#     ),
#     torch.nn.ReLU(),
    
#     torch.nn.Conv2d(
#         in_channels=16,
#         out_channels=32,
#         kernel_size=3
#     ),
#     torch.nn.MaxPool2d(
#         kernel_size=2
#     ),
#     torch.nn.ReLU(),
    
#     torch.nn.Conv2d(
#         in_channels=32,
#         out_channels=64,
#         kernel_size=3
#     ),
#     torch.nn.MaxPool2d(
#         kernel_size=2
#     ),
#     torch.nn.ReLU(),
    
#     Flatten(),
    
#     torch.nn.Linear(
#         in_features=1024,
#         out_features=1
#     ),
#     torch.nn.Sigmoid()
# )
# model = torchvision.models.resnet50(pretrained=True)
model = torchvision.models.resnet50()
model
model.fc = torch.nn.Sequential(
    torch.nn.Linear(
        in_features=2048,
        out_features=1
    ),
    torch.nn.Sigmoid()
)
model
# out = model(train_data[0]["image"].view(1, 3, 224, 224))
# out.shape
# Some utils functions.
# Seems like PyTorch does not auto-infer tensor shapes in a sequential model, so we need to figure the shapes ourself.

def compute_conv2d_output_dimensions(Hin, Win, kernel_size, padding=(0, 0), dilation=(1, 1), stride=(1, 1)):
    Hout = math.floor(((Hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1)
    Wout = math.floor(((Win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1)
    return Hout, Wout


def compute_maxpooling2d_output_dimensions(Hin, Win, kernel_size, stride=None, padding=(0, 0), dilation=(1, 1)):
    if stride is None:
        stride = kernel_size
    
    Hout = math.floor(((Hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1)
    Wout = math.floor(((Win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1)
    return Hout, Wout
# compute_conv2d_output_dimensions(96, 96, (3, 3))
# compute_maxpooling2d_output_dimensions(94, 94, kernel_size=(2, 2))
if USE_GPU:
    model = model.cuda() # Should be called before instantiating optimizer according to docs: https://pytorch.org/docs/stable/nn.html

optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.BCELoss()

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for i in range(EPOCHS):
    for phase in ["train", "test"]:
        if phase == "train":
            model.train()
        else:
            model.eval()
        
        samples = 0
        loss_sum = 0
        correct_sum = 0
        for j, batch in enumerate(dataloaders[phase]):
            X = batch["image"]
            labels = batch["label"]
            if USE_GPU:
                X = X.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                y = model(X)
                loss = criterion(
                    y, 
                    labels.view(-1, 1).float()
                )

                if phase == "train":
                    loss.backward()
                    optimizer.step()
                    
                loss_sum += loss.item() * X.shape[0] # We need to multiple by batch size as loss is the mean loss of the samples in the batch
                samples += X.shape[0]
                num_corrects = torch.sum((y >= 0.5).float() == labels.view(-1, 1).float())
                correct_sum += num_corrects
                
                # Print batch statistics every 50 batches
                if j % 50 == 49 and phase == "train":
                    print("{}:{} - loss: {}, acc: {}".format(
                        i + 1, 
                        j + 1, 
                        float(loss_sum) / float(samples), 
                        float(correct_sum) / float(samples)
                    ))
                
        # Print epoch statistics
        epoch_acc = float(correct_sum) / float(samples)
        epoch_loss = float(loss_sum) / float(samples)
        print("epoch: {} - {} loss: {}, {} acc: {}".format(i + 1, phase, epoch_loss, phase, epoch_acc))
        
        # Deep copy the model
        if phase == "test" and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "resnet50.pth")
# torch.save(best_model_wts, "resnet50.pth")
model1 = torchvision.models.resnet50()
model1.fc = torch.nn.Sequential(
    torch.nn.Linear(
        in_features=2048,
        out_features=1
    ),
    torch.nn.Sigmoid()
)
model1.load_state_dict(torch.load("resnet50.pth"))
test_data = HistopathologicCancerDataset(
    img_dir="../input/test/",
    transform=transform_pipe
)
test_loader1 = torch.utils.data.DataLoader(
    test_data,
    batch_size=64,
#     shuffle=True,
#     num_workers=8
)
model1.eval()
if USE_GPU:
    model1 = model1.cuda()

ids_all = []
predictions = []

for j, batch in enumerate(test_loader1):
    X = batch["image"]
    ids = batch["id"]
    if USE_GPU:
        X = X.cuda()
    
    for _id in ids:
        ids_all.append(_id)

    with torch.set_grad_enabled(False):
        y_pred = model1(X)
        predictions.append((y_pred >= 0.5).float().cpu().numpy())
        
print("Done making predictions!")
submissions = pd.DataFrame({
    "id": ids_all,
    "label": np.concatenate(predictions).reshape(-1,).astype("int")
}).set_index("id")
submissions.head()
submissions.to_csv("submissions.csv")
