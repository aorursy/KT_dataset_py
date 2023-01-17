%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from pathlib import Path

import random

import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

import torch

from tqdm import tqdm
cfg = {

    # Batch Size for Training and Varidation

        "batch_size": 1024,

    # CUDA:0 or CPU

        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),

    # Epoch Size for Training and Validation

        "epoch_size": 50,

    # Path to Dig-MNIST.csv

        "path_Dig-MNIST_csv": Path("../input/Kannada-MNIST/Dig-MNIST.csv"),

    # Path to train.csv

        "path_train_csv": Path("../input/Kannada-MNIST/train.csv"),

    # Random Seed

        "seed": 17122019,

    # Ratio of Training Dataset against Overall One

        "train_dataset_ratio": 0.9,

}
random.seed(cfg["seed"])

np.random.seed(cfg["seed"])

torch.manual_seed(cfg["seed"])

if torch.cuda.is_available():

    torch.cuda.manual_seed(cfg["seed"])

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False
sns.set(style="darkgrid", context="notebook", palette="muted")
class KannadaMNISTDataset(torch.utils.data.Dataset):

    def __init__(self,

                 path_csv: Path,

                 cfg: dict):

        df_csv = pd.read_csv(path_csv)



        self.imgs = df_csv.drop(["label"], axis=1).values.astype(np.int32)

        # Reshape Image from (data_size, 784) to (data_size, 1, 28, 28)

        self.imgs = self.imgs.reshape(-1, 1, 28, 28)

        # Scale Image from [0, 255] to [0.0, 1.0]

        self.imgs = torch.tensor(self.imgs/255.0,

                                 dtype=torch.float32,

                                 device=cfg["device"])



        self.labels = torch.tensor(df_csv["label"],

                                   dtype=torch.int64,

                                   device=cfg["device"])



    def __len__(self):

        return self.labels.shape[0]



    def __getitem__(self, idx):

        img = self.imgs[idx]

        label = self.labels[idx]

        return img, label
def create_training_datasets(cfg: dict):

    # Create Overall Dataset Setting KannadaMNISTTransform

    overall_dataset = KannadaMNISTDataset(cfg["path_Dig-MNIST_csv"], cfg)

    # Split Overall Dataset into Training and Validation Ones

    train_size = int(len(overall_dataset) * cfg["train_dataset_ratio"])

    valid_size = len(overall_dataset) - train_size

    train_dataset, valid_dataset = torch.utils.data.random_split(overall_dataset,

                                                                 [train_size, valid_size])

    return train_dataset, valid_dataset
%%time

# Training Datasets

train_dataset, valid_dataset = create_training_datasets(cfg)

# Test Dataset

test_dataset = KannadaMNISTDataset(cfg["path_train_csv"], cfg)
args = (len(train_dataset), len(valid_dataset), len(test_dataset))

print("Train:%d,Valid:%d,Test:%d" % args)
class ThisNetwork(torch.nn.Module):

    def __init__(self):

        super(ThisNetwork, self).__init__()



        # STN Localization(CNN)

        self.loc_cnn = torch.nn.Sequential(

            # (batch,1,28,28) -> (batch,8,24,24)

            torch.nn.Conv2d(in_channels=1,

                            out_channels=8,

                            kernel_size=5),

            torch.nn.BatchNorm2d(num_features=8),

            torch.nn.ReLU(inplace=True),

            # (batch,8,24,24) -> (batch,8,12,12)

            torch.nn.MaxPool2d(kernel_size=2,

                               stride=2),

            # (batch,8,12,12) -> (batch,16,8,8)

            torch.nn.Conv2d(in_channels=8,

                            out_channels=16,

                            kernel_size=5),

            torch.nn.BatchNorm2d(num_features=16),

            torch.nn.ReLU(inplace=True),

            # (batch,16,8,8) -> (batch,16,4,4)

            torch.nn.MaxPool2d(kernel_size=2,

                               stride=2),

        )



        # STN Localization(FC)

        self.loc_fc = torch.nn.Sequential(

            # (batch,256) -> (batch,64)

            torch.nn.Linear(in_features=256,

                            out_features=64),

            torch.nn.ReLU(inplace=True),

            torch.nn.Dropout(p=0.5),

            # (batch,64) -> (batch,6)

            torch.nn.Linear(in_features=64,

                            out_features=6),

        )



        self.cnn = torch.nn.Sequential(

            # (batch,1,28,28) -> (batch,64,28,28)

            torch.nn.Conv2d(in_channels=1,

                            out_channels=64,

                            kernel_size=3,

                            padding=1),

            torch.nn.BatchNorm2d(num_features=64),

            torch.nn.ReLU(inplace=True),

            # (batch,64,28,28) -> (batch,64,14,14)

            torch.nn.MaxPool2d(kernel_size=2,

                               stride=2),

            # (batch,64,14,14) -> (batch,128,14,14)

            torch.nn.Conv2d(in_channels=64,

                            out_channels=128,

                            kernel_size=3,

                            padding=1),

            torch.nn.BatchNorm2d(num_features=128),

            torch.nn.ReLU(inplace=True),

            # (batch,128,14,14) -> (batch,128,7,7)

            torch.nn.MaxPool2d(kernel_size=2,

                               stride=2),

        )



        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=3)



        self.fc = torch.nn.Sequential(

            # (batch,1152) -> (batch,256)

            torch.nn.Linear(in_features=1152,

                            out_features=256),

            torch.nn.ReLU(inplace=True),

            torch.nn.Dropout(p=0.5),

            # (batch,256) -> (batch,10)

            torch.nn.Linear(in_features=256,

                            out_features=10),

        )

        self.log_softmax = torch.nn.LogSoftmax(dim=-1)



    def forward(self, x):

        # STN Localization Network

        # (batch,1,28,28) -> (batch,256)

        theta = self.loc_cnn(x)

        theta = theta.view(theta.size(0), -1)

        # (batch,256) -> (batch,2,3)

        theta = self.loc_fc(theta).view(-1, 2, 3)



        # STN Grid Generator

        # (batch,1,28,28), (batch,2,3) -> (batch,28,28,2)

        grid = torch.nn.functional.affine_grid(theta, x.size(),

                                               align_corners=True)



        # STN Sampler

        # (batch,1,28,28), (batch,28,28,2) -> (batch,1,28,28)

        x = torch.nn.functional.grid_sample(x, grid,

                                            align_corners=True)



        # Non-STN

        # (batch,1,28,28) -> (batch,128,7,7)

        x = self.cnn(x)

        # (batch,128,7,7) -> (batch,1152)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        # (batch,1152) -> (batch,10)

        x = self.fc(x)

        return self.log_softmax(x)
network = ThisNetwork().to(cfg["device"])
def learn(network: torch.nn.Module,

          train_dataset: KannadaMNISTDataset,

          valid_dataset: KannadaMNISTDataset,

          cfg: dict):

    result = {"Epoch" : [],

              "Type" : [],

              "Average Loss" : [],

              "Accuracy" : []}

    criterion = torch.nn.NLLLoss()

    optimizer = torch.optim.Adam(network.parameters())

    train_loader = torch.utils.data.DataLoader(train_dataset,

                                               batch_size=cfg["batch_size"],

                                               shuffle=True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,

                                               batch_size=cfg["batch_size"],

                                               shuffle=True)



    # Start

    for epoch in range(1, cfg["epoch_size"]+1):

        # Training

        sum_loss = 0.0

        sum_correct = 0

        for imgs, true_labels in tqdm(train_loader):

            network.zero_grad()

            pred_probs = network(imgs)

            pred_labels = torch.argmax(pred_probs, dim=1)

            loss = criterion(pred_probs, true_labels)

            loss.backward()

            optimizer.step()

            sum_loss += loss.item() * imgs.shape[0]

            sum_correct += int(torch.sum(pred_labels == true_labels))

        ave_loss = sum_loss / len(train_dataset)

        accuracy = 100.0 * sum_correct / len(train_dataset)

        result["Epoch"].append(epoch)

        result["Type"].append("Training")

        result["Average Loss"].append(ave_loss)

        result["Accuracy"].append(accuracy)

        args = (epoch, cfg["epoch_size"], ave_loss, accuracy)

        print_str = "[Training]Epoch:%d/%d,Average Loss:%.3f,Accuracy:%.2f%%"

        print(print_str % args)



        # Validation

        sum_loss = 0.0

        sum_correct = 0

        for imgs, true_labels in tqdm(valid_loader):

            pred_probs = network(imgs)

            pred_labels = torch.argmax(pred_probs, dim=1)

            loss = criterion(pred_probs, true_labels)

            sum_loss += loss.item() * imgs.shape[0]

            sum_correct += int(torch.sum(pred_labels == true_labels))

        ave_loss = sum_loss / len(valid_dataset)

        accuracy = 100.0 * sum_correct / len(valid_dataset)

        result["Epoch"].append(epoch)

        result["Type"].append("Validation")

        result["Average Loss"].append(ave_loss)

        result["Accuracy"].append(accuracy)

        args = (epoch, cfg["epoch_size"], ave_loss, accuracy)

        print_str = "[Validation]Epoch:%d/%d,Average Loss:%.3f,Accuracy:%.2f%%"

        print(print_str % args)



    return result
%%time

result = learn(network,

               train_dataset,

               valid_dataset,

               cfg)
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
def test(test_dataset: KannadaMNISTDataset,

         network: torch.nn.Module,

         cfg: dict):

    test_true_labels = np.array([])

    test_pred_labels = np.array([])

    test_loader = torch.utils.data.DataLoader(test_dataset,

                                              batch_size=cfg["batch_size"])



    # Prediction

    for imgs, true_labels in tqdm(test_loader):

        pred_probs = network(imgs)

        pred_labels = torch.argmax(pred_probs, dim=1)

        test_true_labels = np.concatenate([test_true_labels,

                                           true_labels.cpu().numpy()])

        test_pred_labels = np.concatenate([test_pred_labels,

                                           pred_labels.cpu().numpy()])

    return test_true_labels, test_pred_labels
%%time

test_true_labels, test_pred_labels = test(test_dataset, network, cfg)
target_str = ["Image No.%d" % num for num in range(10)]

report_str = classification_report(test_true_labels,

                                   test_pred_labels,

                                   target_names=target_str,

                                   digits=3)

print(report_str)
cm = pd.DataFrame(confusion_matrix(test_true_labels, test_pred_labels),

                  columns=np.unique(test_true_labels),

                  index=np.unique(test_pred_labels))

cm.index.name = "True Image No."

cm.columns.name = "Predicted Image No."

sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")