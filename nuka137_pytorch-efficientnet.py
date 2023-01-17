!mkdir -p /tmp/pip/cache

!cp ../input/piplocal/efficientnet_pytorch-0.7.0.xyz /tmp/pip/cache/efficientnet_pytorch-0.7.0.tar

!cp ../input/piplocal/torch_optimizer-0.0.1a15-py3-none-any.xyz /tmp/pip/cache/torch_optimizer-0.0.1a15-py3-none-any.whl

!cp ../input/piplocal/pytorch_ranger-0.1.1-py3-none-any.xyz /tmp/pip/cache/pytorch_ranger-0.1.1-py3-none-any.whl

!pip install --no-index --find-links /tmp/pip/cache/ efficientnet_pytorch torch_optimizer pytorch_ranger



!mkdir -p /root/.cache/torch/checkpoints

!cp ../input/pytorchcheckpoints/efficientnet-b0-355c32eb.xyz /root/.cache/torch/checkpoints/efficientnet-b0-355c32eb.pth
import pandas as pd

import matplotlib.pyplot as plt

from efficientnet_pytorch import EfficientNet

from PIL import Image

import torch

from torchvision import transforms

import torch_optimizer

import time

import random

import os

import numpy as np

import gc

from sklearn.preprocessing import LabelEncoder
BATCH_SIZE = 64

LOG_STEPS = 10

MIN_SAMPLES_PER_CLASS = 150
gc.enable()
def get_device():

    if torch.cuda.is_available():

        device_type = "cuda"

        print("Train on GPU.")

    else:

        device_type = "cpu"

        print("Train on CPU.")

    device = torch.device(device_type)

    

    return device
def fix_randomness(seed):

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True

fix_randomness(1)
device = get_device()
train_df = pd.read_csv("../input/landmark-recognition-2020/train.csv")



counts = train_df["landmark_id"].value_counts()

selected = counts[counts >= MIN_SAMPLES_PER_CLASS].index

print('classes with at least N samples:', selected.shape[0])

train_df = train_df[train_df["landmark_id"].isin(selected)]



label_encoder = LabelEncoder()

label_encoder.fit(train_df["landmark_id"].values)

assert len(label_encoder.classes_) == selected.shape[0]



train_df["landmark_id"] = label_encoder.transform(train_df["landmark_id"])
test_df = pd.read_csv("../input/landmark-recognition-2020/sample_submission.csv")
image_dir = "../input/landmark-recognition-2020"
id_ = train_df["id"].iloc[2]

print(f"ID: {id_}")

filepath = "{}/train/{}/{}/{}/{}.jpg".format(image_dir, id_[0], id_[1], id_[2], id_)

img = Image.open(filepath)



plt.figure(figsize=(5, 5))

plt.imshow(img)

plt.title("Landmark: {}".format(train_df[train_df["id"] == id_]["landmark_id"].iloc[0]))
class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, df, image_dir, mode="train"):

        self.df = df

        self.mode = mode

        self.image_dir = image_dir

        

        transform_list = [

            transforms.Resize((64, 64)),

            transforms.ToTensor(),

            transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                std=[0.229, 0.224, 0.225]),

        ]

        self.transforms = transforms.Compose(transform_list)

    

    def __getitem__(self, index):

        id_ = self.df["id"].iloc[index]

        filepath = "{}/{}/{}/{}/{}/{}.jpg".format(image_dir, self.mode, id_[0], id_[1], id_[2], id_)

        img = Image.open(filepath)

        img = self.transforms(img)

        

        if self.mode == "train":

            return {"image": img, "target": self.df["landmark_id"].iloc[index]}

        elif self.mode == "test":

            return {"image": img}



    def __len__(self):

        return self.df.shape[0]
class CustomizedEfficientNet(torch.nn.Module):

    def __init__(self, num_classes):

        super(CustomizedEfficientNet, self).__init__()

        self.base = EfficientNet.from_pretrained("efficientnet-b0")

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        out = self.base._fc.in_features

        self.fc = torch.nn.Linear(out, num_classes)

    

    def forward(self, x):

        x = self.base.extract_features(x)

        x = self.avg_pool(x).squeeze(-1).squeeze(-1)

        x = self.fc(x)

        return x
def GAP(predicts, confs, targets):

    _, indices = torch.sort(confs, descending=True)



    confs = confs.cpu().numpy()

    predicts = predicts[indices].cpu().numpy()

    targets = targets[indices].cpu().numpy()

    

    correct = 0

    for i, (p, c, t) in enumerate(zip(predicts, confs, targets)):

        rel = int(p == t)

        correct += rel

        precision = correct * rel / (i+1)

    

    return correct / targets.shape[0]   # NEED TO FIX
num_classes = train_df["landmark_id"].unique().shape[0]

print(num_classes)



train_dataset = ImageDataset(train_df, image_dir)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=True)

test_dataset = ImageDataset(test_df, image_dir, mode="test")

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)



model = CustomizedEfficientNet(num_classes)

model.to(device)

optimizer = torch_optimizer.RAdam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*1, eta_min=1e-6)





loss_fn = torch.nn.CrossEntropyLoss()



num_steps = len(train_loader)



model.train()

for i, data in enumerate(train_loader):

    start = time.time()

    

    optimizer.zero_grad()



    X = data["image"].to(device)

    y = data["target"].to(device)



    output = model(X)

    loss = loss_fn(output, y)

    confs, preds = torch.max(output.detach(), dim=1)

    gap = GAP(preds, confs, y)

    

    loss.backward()

    optimizer.step()

    scheduler.step()

    lr = optimizer.param_groups[0]['lr']

    

    elapsed = time.time() - start

    

    if i % LOG_STEPS == 0:

        print(f"[{i}/{num_steps}]: time {elapsed}, GAP {gap}, lr {lr}")



all_confs = []

all_preds = []

with torch.no_grad():

    for i, data in enumerate(test_loader):

        X = data["image"].to(device)

        

        output = model(X)

        confs, preds = torch.topk(output, 20)

        all_confs.append(confs)

        all_preds.append(preds)
submission_confs = torch.cat(all_confs).cpu()

submission_confs = np.array(submission_confs)



submission_preds = torch.cat(all_preds).cpu()

submission_preds = [label_encoder.inverse_transform(p) for p in submission_preds]

submission_preds = np.array(submission_preds)



landmark = []

for c, p in zip(submission_confs, submission_preds):

    c0 = c[0]

    p0 = p[0]

    landmark.append(f"{c0} {p0}")

    #landmark.append(" ".join([f"{cc} {pp}" for cc, pp in zip(c, p)]))
submission_df = pd.read_csv("../input/landmark-recognition-2020/sample_submission.csv")



submission_df["landmarks"] = landmark

submission_df.set_index("id", inplace=True)

submission_df.to_csv("submission.csv")