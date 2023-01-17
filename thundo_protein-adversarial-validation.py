import numpy as np
import pandas as pd
import torch
import cv2
import random
import math

from PIL import Image
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet34
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from pathlib import Path


INPUT_PATH = Path("../input")
TRAIN_PATH = INPUT_PATH / "train"
TEST_PATH = INPUT_PATH / "test"

FILTERS = ["red", "green", "blue", "yellow"]
MEAN = [0.08069, 0.05258, 0.05487, 0.08282]
STD = [0.13704, 0.10145, 0.15313, 0.13814]
SEED = 666
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
def null_collate(batch):
    batch_size = len(batch)
    images = np.array([x[0] for x in batch])
    images = torch.from_numpy(images)
    
    labels = np.array([x[1] for x in batch])
    labels = torch.from_numpy(labels)

    assert(images.shape[0] == labels.shape[0] == batch_size)

    return images, labels


class ProteinDataset(Dataset):
    def __init__(self, size, augment=None):
        super(ProteinDataset, self).__init__()
        self.augment = augment

        train_csv_filepath = INPUT_PATH / "train.csv"
        self.train_df = self._load_dataframe(train_csv_filepath)
        test_csv_filepath = INPUT_PATH / "sample_submission.csv"
        self.test_df = self._load_dataframe(test_csv_filepath)
        
        self.train_df_len = len(self.train_df)
        self.test_df_len = len(self.test_df)
        
        self.total_length = len(self.train_df) + len(self.test_df)
        print("Train dataset: {}, test dataset {}, total {}".format(self.train_df_len, self.test_df_len, len(self.train_df) + len(self.test_df)))
        
        self.labels = np.concatenate((
            np.ones(self.train_df_len),
            np.zeros(self.test_df_len)
        ), axis=0)
    
    @staticmethod
    def _load_dataframe(path):
        print("Loading csv from {}".format(path))
        return pd.read_csv(path)

    @staticmethod
    def _load_image(path, size):
        img = Image.open(path)
        img = cv2.resize(np.array(img), (size, size), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(img, axis=2)
        return img
    
    @staticmethod
    def _get_row_id(df, index):
        return df.loc[index, "Id"]

    def __getitem__(self, index):       
        if index > len(self.train_df) - 1:
            # Index belongs to test dataset
            offset_index = index - len(self.train_df)
            image_id = self._get_row_id(self.test_df, offset_index)
            img = np.concatenate([
                self._load_image(TEST_PATH / (image_id + "_" + i + ".png"), size) for i in FILTERS
            ], axis=2)
            label = 0
        else:
            # Index belongs to train dataset
            image_id = self._get_row_id(self.train_df, index)
            img = np.concatenate([
                self._load_image(TRAIN_PATH / (image_id + "_" + i + ".png"), size) for i in FILTERS
            ], axis=2)
            label = 1

        img = np.transpose(img, axes=[2, 0, 1])

        if self.augment is not None:
            img = self.augment(img)

        return img, label

    def __len__(self):
        return self.total_length
class ResNet34(nn.Module):
    def __init__(self, num_classes=1, dropout=0.5, middle_features=128):
        super(ResNet34, self).__init__()
        resnet = resnet34(pretrained=True)

        # Support for 4-channels
        w = resnet.conv1.weight
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = torch.nn.Parameter(
            torch.cat((w, torch.zeros(64, 1, 7, 7)), dim=1)
        )
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))#resnet.avgpool
        
        bottleneck_features = resnet.fc.in_features
        self.fc = nn.Sequential(
            nn.BatchNorm1d(bottleneck_features),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_features, middle_features),
            nn.ReLU(),
            nn.BatchNorm1d(middle_features),
            nn.Dropout(dropout),
            nn.Linear(middle_features, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        mean = MEAN
        std = STD
        x = x / 255.
        x = torch.cat([
            (x[:, [0]] - mean[0]) / std[0],
            (x[:, [1]] - mean[1]) / std[1],
            (x[:, [2]] - mean[2]) / std[2],
            (x[:, [3]] - mean[3]) / std[3],
        ], 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x) 
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
class Accuracy(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, y_true, y_pred):
        y_pred = (y_pred > self.threshold).int()
        y_true = y_true.int()
        return (y_pred == y_true).float().mean()
class Trainer:
    def __init__(self, batch_size, size):
        self.batch_size = batch_size
        self.size = size

        self.optimizer = None
        self.scheduler = None

        self.train_dataset = ProteinDataset(size)
        self.validation_dataset = ProteinDataset(size)
        
        self.train_idx, self.validation_idx = train_test_split(
            list(range(len(self.train_dataset))),
            test_size=0.1,
            stratify=self.train_dataset.labels
        )
        
        print("Train len: {}, validation len: {}".format(len(self.train_idx), len(self.validation_idx)))

        loader_params = dict(
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            collate_fn=null_collate
        )
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            sampler=SubsetRandomSampler(self.train_idx),
            **loader_params
        )
        self.validation_loader = DataLoader(
            dataset=self.validation_dataset,
            sampler=SubsetRandomSampler(self.validation_idx),
            **loader_params
        )
        print("Train set: {}".format(len(self.train_idx)))
        print("Validation set: {}".format(len(self.validation_idx)))

        self.it_per_epoch = math.ceil(len(self.train_idx) / self.batch_size)
        
        
    def run(self):
        model = ResNet34()
        model = model.cuda()

        lr = 0.2
        it = 0
        epoch = 0
        max_epochs = 20
        it_save = self.it_per_epoch * 5
        it_log = self.it_per_epoch / 5
        it_smooth = 50
        
        self.optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0001)
        self.scheduler = StepLR(self.optimizer, 5 * self.it_per_epoch, gamma=0.5)

        criterion = nn.BCELoss()
        criterion = criterion.cuda()
        metrics = [Accuracy(), roc_auc_score]

        print("{}'".format(self.optimizer))
        print("{}'".format(self.scheduler))
        print("{}'".format(criterion))
        print("{}'".format(metrics))

        train_loss = 0
        train_acc = 0

        print('                    |         VALID         |        TRAIN          |         ')
        print(' lr     iter  epoch | loss    roc    acc    | loss    roc    acc    |  time   ')
        print('------------------------------------------------------------------------------')

        start = timer()
        while epoch < max_epochs:
            smoothed_train_loss = 0
            smoothed_sum = 0

            for inputs, labels in self.train_loader:
                epoch = (it + 1) / self.it_per_epoch

                # checkpoint
                if it % it_save == 0 and it != 0:
                    self.save(model, self.optimizer, it, epoch)

                # training
                self.scheduler.step()

                lrs = self.scheduler.get_lr()
                lr = lrs[-1]

                model.train()
                inputs = inputs.cuda().float()
                labels = labels.cuda().float()

                preds = model(inputs)
                loss = criterion(preds, labels)
                with torch.no_grad():
                    train_acc, train_roc = [i(labels, preds).item() for i in metrics]

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()

                smoothed_train_loss += loss.item()
                smoothed_sum += 1
                if it % it_smooth == 0:
                    train_loss = smoothed_train_loss / smoothed_sum
                    smoothed_train_loss = 0
                    smoothed_sum = 0

                if it % it_log == 0:
                    batch_loss = loss.item()
                    print(
                        "{:5f} {:4.1f} {:5.1f} |                    | {:0.3f}  {:0.3f}  {:0.3f}  | {:6.2f}".format(
                            lr, it / 1000, epoch, batch_loss, train_roc, train_acc, timer() - start
                        ))

                it += 1

            # validation
            valid_loss, valid_m = self.do_valid(model, criterion, metrics)
            valid_acc, valid_roc = valid_m

            print(
                "{:5f} {:4.1f} {:5.1f} | {:0.3f}* {:0.3f}  {:0.3f}  | {:0.3f}  {:0.3f}  {:0.3f}  | {:6.2f}".format(
                    lr, it / 1000, epoch, valid_loss, valid_roc, valid_acc, train_loss, train_roc, train_acc, timer() - start
                ))

            # Data loader end
        # Training end

        self.save(model, self.optimizer, it, epoch)

    def do_valid(self, model, criterion, metrics):
        model.eval()
        valid_num = 0
        losses = []

        for inputs, labels in self.validation_loader:
            inputs = inputs.cuda().float()
            labels = labels.cuda().float()

            with torch.no_grad():
                preds = model(inputs)
                loss = criterion(preds, labels)
                m = [i(labels, preds).item() for i in metrics]

            valid_num += len(inputs)
            losses.append(loss.data.cpu().numpy())

        assert (valid_num == len(self.validation_loader.sampler))
        loss = np.array(losses).mean()
        return loss, m
    
    def save(self, model, optimizer, iter, epoch):
        torch.save(model.state_dict(), "{}_model.pth".format(iter))
        torch.save({
            "optimizer": optimizer.state_dict(),
            "iter": iter,
            "epoch": epoch
        }, "{}_optimizer.pth".format(iter))
batch_size = 64
size = 256
trainer = Trainer(batch_size, size)
trainer.run()