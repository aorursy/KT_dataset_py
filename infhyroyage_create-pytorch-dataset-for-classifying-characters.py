%matplotlib inline

from datetime import datetime

import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path

import pandas as pd

from PIL import Image

import random

import seaborn as sns

from tqdm import tqdm

import torch

import torchvision
sns.set(style="darkgrid", context="notebook", palette="muted")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 7

random.seed(seed)

np.random.seed(seed)

torch.manual_seed(seed)

if torch.cuda.is_available():

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False
input_path = Path("../input/kuzushiji-recognition")

train_imgs_path = input_path / "train_images"

print("Train Images:%d" % len(list(train_imgs_path.glob("*jpg"))))
train = pd.read_csv(input_path / "train.csv")

uc_trans = pd.read_csv(input_path / "unicode_translation.csv")
train.head()
train.info()
train_nan_labels = train[train["labels"].isnull()]

train_nan_labels.head(6)
train_nan_labels.info()
fig = plt.figure(figsize=(20, 80))

for i in range(6):

    image_id = train_nan_labels["image_id"].iloc[i]

    file_name = image_id + ".jpg"

    train_img_path = train_imgs_path / file_name

    train_img = np.asarray(Image.open(train_img_path))

    fig.add_subplot(1, 6, i+1, title=file_name)

    plt.axis("off")

    plt.imshow(train_img)

plt.show()
train = train.dropna()

train = train.reset_index(drop=True)

train.info()
train.head()
train_chars = {}

train_chars_num = 0

for i in tqdm(range(train.shape[0])):

    image_id = train.iloc[i]["image_id"]

    labels = train.iloc[i]["labels"].split(" ")

    values = {"Unicode" : [],

              "X" : [],

              "Y" : [],

              "Width" : [],

              "Height" : []}

    for j in range(0, len(labels), 5):

        uc = labels[j]

        x = int(labels[j+1])

        y = int(labels[j+2])

        w = int(labels[j+3])

        h = int(labels[j+4])

        values["Unicode"].append(uc)

        values["X"].append(x)

        values["Y"].append(y)

        values["Width"].append(w)

        values["Height"].append(h)

        train_chars_num += 1

    train_chars[image_id] = values

train_chars_num
fig = plt.figure(figsize=(20, 80))

image_id_1st = train.iloc[0]["image_id"]

img_1st = Image.open(train_imgs_path/(image_id_1st+".jpg"))

for i in range(6):

    uc = train_chars[image_id_1st]["Unicode"][i]

    x = train_chars[image_id_1st]["X"][i]

    y = train_chars[image_id_1st]["Y"][i]

    w = train_chars[image_id_1st]["Width"][i]

    h = train_chars[image_id_1st]["Height"][i]

    img = img_1st.crop((x, y, x+w, y+h))

    args = (uc, x, y, w, h)

    print("Unicode:%s,X:%d,Y:%d,Width:%d,Height:%d" % args)

    fig.add_subplot(1, 6, i+1, title="Unicode:%s" % uc)

    plt.axis("off")

    plt.imshow(np.asarray(img))

plt.show()
plot_data = []

for train_chars_value in train_chars.values():

    plot_data.extend(train_chars_value["Width"])

sns.distplot(plot_data, kde=False, rug=True)
plot_data = []

for train_chars_value in train_chars.values():

    plot_data.extend(train_chars_value["Height"])

sns.distplot(plot_data, kde=False, rug=True)
w_resize = 48

h_resize = 48
uc_trans.head()
uc_trans.info()
train_chars_ucs = set()

for train_chars_value in train_chars.values():

    train_chars_ucs |= set(train_chars_value["Unicode"])

uc_trans[~uc_trans["Unicode"].isin(train_chars_ucs)].info()
uc_trans = uc_trans[uc_trans["Unicode"].isin(train_chars_ucs)]

uc_trans.info()
uc_list = uc_trans["Unicode"].values.tolist()

uc_list.index("U+306F")
class KuzushijiCharDataset(torch.utils.data.Dataset):

    def __init__(self,

                 chars: dict,

                 uc_list: list,

                 train_imgs_path: Path,

                 scale_resize: tuple):

        self._x_in_list = []

        self._y_list = []

        for image_id, values in tqdm(chars.items()):

            # Open PIL Image each image_id

            img = Image.open(train_imgs_path/(image_id+".jpg"))

            values_zip = zip(values["Unicode"],

                             values["X"],

                             values["Y"],

                             values["Width"],

                             values["Height"])

            for uc, x, y, w, h in values_zip:

                # Crop as Character's PIL Image

                img_char = img.crop((x, y, x+w, y+h))

                # Resize Character's PIL Image

                img_char = img_char.resize(scale_resize)

                # Gray-Scale Character's PIL Image where the channel is 1

                img_char = img_char.convert('L')

                # Convert from Character's PIL Image to Tensor

                img_char = torchvision.transforms.functional.to_tensor(img_char)

                # Add Training Data

                self._x_in_list.append(img_char)

                # Add Training Label

                uc_idx = uc_list.index(uc)

                self._y_list.append(uc_idx)



    def __len__(self):

        return len(self._y_list)

    

    def __getitem__(self, idx: int):

        x_in = self._x_in_list[idx]

        y = self._y_list[idx]

        return x_in, y
%%time

dataset = KuzushijiCharDataset(train_chars,

                               uc_list,

                               train_imgs_path,

                               (w_resize, h_resize))

len(dataset)
train_size = int(len(dataset) * 0.9)

valid_size = len(dataset) - train_size

train_dataset, valid_dataset = torch.utils.data.random_split(dataset,

                                                             [train_size, valid_size])

args = (len(dataset), len(train_dataset), len(valid_dataset))

print("Total:%d,Training:%d,Validation:%d" % args)
class DemoModel(torch.nn.Module):

    def __init__(self):

        super(DemoModel, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=1,

                                     out_channels=16,

                                     kernel_size=7)

        self.relu1 = torch.nn.ReLU(inplace=True)

        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv2 = torch.nn.Conv2d(in_channels=16,

                                     out_channels=128,

                                     kernel_size=6)

        self.relu2 = torch.nn.ReLU(inplace=True)

        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)

        self.fc = torch.nn.Linear(in_features=128*8*8,

                                  out_features=4212,

                                  bias=True)

        self.log_softmax = torch.nn.LogSoftmax(dim=-1)



    def forward(self, x):

        out = self.conv1(x) # (batch, 1, 48, 48) -> (batch, 16, 42, 42)

        out = self.relu1(out)

        out = self.maxpool1(out) # (batch, 16, 42, 42) -> (batch, 16, 21, 21)

        out = self.conv2(out) # (batch, 16, 21, 21) -> (batch, 128, 16, 16)

        out = self.relu2(out)

        out = self.maxpool2(out) # (batch, 128, 16, 16) -> (batch, 128, 8, 8)

        out = out.view(out.size(0), -1) # (batch, 128, 8, 8) -> (batch, 8192)

        out = self.fc(out) # (batch, 8192) -> (batch, 4212)

        out = self.log_softmax(out)

        return out
network = DemoModel().to(device)

network
max_epochs = 10

batch_size = 1024

lr = 0.005

optimizer = torch.optim.Adam(network.parameters())

criterion = torch.nn.NLLLoss()

train_dataLoader = torch.utils.data.DataLoader(train_dataset,

                                               batch_size=batch_size,

                                               shuffle=True)

valid_dataLoader = torch.utils.data.DataLoader(valid_dataset)

args = (len(train_dataLoader), len(valid_dataLoader))

print("Training:%d,Validation:%d" % args)
%%time

result = {"Epoch" : [],

          "Type" : [],

          "Average Loss" : [],

          "Accuracy" : []}

for epoch in range(1, max_epochs+1):

    # Training

    sum_loss = 0.0

    correct = 0

    for x_in, y in tqdm(train_dataLoader):

        network.zero_grad()

        x_out = network(x_in.to(device))

        loss = criterion(x_out, y.to(device))

        loss.backward()

        optimizer.step()

        sum_loss += loss.item() * x_in.shape[0]

        correct += int(torch.sum(torch.argmax(x_out, 1) == y.to(device)))

    ave_loss = sum_loss / len(train_dataset)

    accuracy = 100.0 * correct / len(train_dataset)

    result["Epoch"].append(epoch)

    result["Type"].append("Training")

    result["Average Loss"].append(ave_loss)

    result["Accuracy"].append(accuracy)

    args = (datetime.now().isoformat(), epoch, max_epochs, ave_loss, accuracy)

    print("Type:Training,Time:%s,Epoch:%d/%d,Average Loss:%.3f,Accuracy:%.3f%%" % args)



    # Validation

    sum_loss = 0.0

    correct = 0

    for x_in, y in tqdm(valid_dataLoader):

        x_out = network(x_in.to(device))

        loss = criterion(x_out, y.to(device))

        sum_loss += loss.item() * x_in.shape[0]

        correct += int(torch.sum(torch.argmax(x_out, 1) == y.to(device)))

    ave_loss = sum_loss / len(valid_dataset)

    accuracy = 100.0 * correct / len(valid_dataset)

    result["Epoch"].append(epoch)

    result["Type"].append("Validation")

    result["Average Loss"].append(ave_loss)

    result["Accuracy"].append(accuracy)

    args = (datetime.now().isoformat(), epoch, max_epochs, ave_loss, accuracy)

    print("Type:Validation,Time:%s,Epoch:%d/%d,Average Loss:%.3f,Accuracy:%.3f%%" % args)
sns.relplot(x="Epoch",

            y="Average Loss",

            hue="Type",

            kind="line",

            data=pd.DataFrame(result))
sns.relplot(x="Epoch",

            y="Accuracy",

            hue="Type",

            kind="line",

            data=pd.DataFrame(result))