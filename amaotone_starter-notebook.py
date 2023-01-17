import gc

import glob

from multiprocessing import cpu_count

import subprocess

import time

import warnings



import cv2

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import torch

import torch.backends.cudnn as cudnn

import torch.nn as nn

import torch.optim as optim

import torchvision

from albumentations import (Compose, HorizontalFlip, HueSaturationValue,

                            Normalize, RandomBrightnessContrast, Resize,

                            ShiftScaleRotate)

import albumentations as alb

from torch.utils.data import DataLoader, Dataset



warnings.filterwarnings('ignore')
LEARNING_RATE = 1e-5

BATCH_SIZE = 16

EPOCH = 5

IMAGE_SIZE = 32



CLASS_WEIGHT = [1.0, 1.0, 1.0, 1.25, 1.25, 2.5, 2.5, 10.0, 10.0, 10.0]

N_CLASSES = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if torch.cuda.is_available():

    cudnn.benchmark = True
class Network(nn.Module):

    def __init__(self):

        super(Network, self).__init__()

        # if you are interested in resnet archtecture, check below link.

        # [https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py]

        self.model = torchvision.models.resnext101_32x8d(pretrained=True)

        self.model.fc = nn.Linear(self.model.fc.in_features, 10)



    def forward(self, x):

        out = self.model(x)

        return out
class Imbalanced_CIFAR10_Dataset(Dataset):

    def __init__(self, mode, visualize=False):

        self.mode = mode

        self.labels = pd.read_csv("../input/train.csv")["label"].values

        self.visualize = visualize



        if mode == "test":

            self.image_files = sorted(glob.glob("../input/images/test/*.png"))

            self.labels = [0] * len(self.image_files)

        else:  # train or valid

            self.image_files = sorted(glob.glob("../input/images/train/*.png"))



            if mode == "train":  # 80% training

                self.image_files = self.image_files[

                    0 : int(len(self.image_files) * 0.8)

                ]

                

                self.labels = self.labels[0 : int(len(self.labels) * 0.8)]



            else:  # 20% validation

                self.image_files = self.image_files[int(len(self.image_files) * 0.8) :]

                self.labels = self.labels[int(len(self.labels) * 0.8) :]



        assert len(self.image_files) == len(self.labels)



        self.class_weight = [(CLASS_WEIGHT[x]) for x in self.labels]



        print("Loading {} images on memory...".format(mode))

        self.images = np.zeros((len(self.image_files), 32, 32, 3)).astype("uint8")



        for i in range(len(self.image_files)):

            self.images[i] = cv2.imread(self.image_files[i])

            if self.visualize and i>10:

                break



    def _augmentation(self, img):

        #-------

        def _albumentations(mode, visualize):

            aug_list = []



            aug_list.append(alb.Resize(36, 36, interpolation=cv2.INTER_CUBIC, p=1.0))

            if mode == "train": # use data augmentation only with train mode

                aug_list.append(alb.HorizontalFlip(p=0.5))

                aug_list.append(alb.ShiftScaleRotate(p=1.0, shift_limit=0.05, scale_limit=0.2, rotate_limit=20))

                

            aug_list.append(alb.Resize(224, 224, interpolation=cv2.INTER_CUBIC, p=1.0))

            if not visualize:

                aug_list.append(

                    Normalize(

                        p=1.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

                    )  # rgb

                )  # based on imagenet



            return Compose(aug_list, p=1.0)



        def _cutout(img):

            # [https://arxiv.org/pdf/1708.04552.pdf]

            mask_value = [

                int(np.mean(img[:, :, 0])),

                int(np.mean(img[:, :, 1])),

                int(np.mean(img[:, :, 2])),

            ]



            mask_size_v = int(IMAGE_SIZE * np.random.randint(10, 60) * 0.01)

            mask_size_h = int(IMAGE_SIZE * np.random.randint(10, 60) * 0.01)



            cutout_top = np.random.randint(

                0 - mask_size_v // 2, IMAGE_SIZE - mask_size_v

            )

            cutout_left = np.random.randint(

                0 - mask_size_h // 2, IMAGE_SIZE - mask_size_h

            )

            cutout_bottom = cutout_top + mask_size_v

            cutout_right = cutout_left + mask_size_h



            if cutout_top < 0:

                cutout_top = 0



            if cutout_left < 0:

                cutout_left = 0



            img[cutout_top:cutout_bottom, cutout_left:cutout_right, :] = mask_value



            return img

        #-------

        

        img = _albumentations(self.mode, self.visualize)(image=img)["image"]

        if (

            self.mode == "train"

            and np.random.uniform() >= 0.5  # 50%

        ):

            img = _cutout(img)

        

        return img

    

    def __len__(self):

        return len(self.image_files)



    def __getitem__(self, idx):

        img = self.images[idx]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB



        img = self._augmentation(img)

        img = img.transpose(2, 0, 1)  # (h, w, c) -> (c, h, w)





        return (torch.tensor(img), torch.tensor(self.labels[idx]))
class Runner(object):

    def __init__(self):

        self.model, self.criterion, self.optimizer = self._build_model()

        self.train_loss_history = []

        self.valid_loss_history = []

        self.valid_weighted_accuracy_history = []



    def _build_model(self):

        model = Network()

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(

            model.parameters(),

            lr=LEARNING_RATE,

        )



        return model.to(DEVICE), criterion, optimizer



    def _build_loader(self, mode):

        dataset = Imbalanced_CIFAR10_Dataset(mode=mode)



        if mode == "train":

            drop_last_flag = True

#             sampler = torch.utils.data.sampler.RandomSampler(data_source=dataset.image_files)

            

            # if you want to sample data based on metric weight, use below sampler.

            

            sampler = torch.utils.data.sampler.WeightedRandomSampler(

                dataset.class_weight, dataset.__len__()

            )

            

        else:  # valid, test

            drop_last_flag = False

            sampler = torch.utils.data.sampler.SequentialSampler(data_source=dataset.image_files)



        loader = DataLoader(

            dataset,

            batch_size=BATCH_SIZE,

            sampler=sampler,

            num_workers=cpu_count(),

            worker_init_fn=lambda x: np.random.seed(),

            drop_last=drop_last_flag,

            pin_memory=True,

        )



        return loader



    def _calc_weighted_accuracy(self, preds, labels):

        score = 0

        total = 0



        for (pred, label) in zip(preds, labels):

            if pred == label:

                score += CLASS_WEIGHT[label]

            total += CLASS_WEIGHT[label]



        return score / total



    def _train_loop(self, loader):

        self.model.train()

        running_loss = 0



        for (images, labels) in loader:

            images, labels = images.to(DEVICE), labels.to(DEVICE)



            outputs = self.model.forward(images)



            train_loss = self.criterion(outputs, labels)



            self.optimizer.zero_grad()

            train_loss.backward()



            self.optimizer.step()



            running_loss += train_loss.item()



        train_loss = running_loss / len(loader)

        

        return train_loss



    def _valid_loop(self, loader):

        self.model.eval()

        running_loss = 0



        valid_preds, valid_labels = [], []



        for (images, labels) in loader:

            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = self.model.forward(images)

            valid_loss = self.criterion(outputs, labels)

            running_loss += valid_loss.item()



            _, predicted = torch.max(outputs.data, 1)



            valid_preds.append(predicted.cpu())

            valid_labels.append(labels.cpu())



        valid_loss = running_loss / len(loader)



        valid_preds = torch.cat(valid_preds)

        valid_labels = torch.cat(valid_labels)

        valid_weighted_accuracy = self._calc_weighted_accuracy(

            valid_preds, valid_labels

        )



        return valid_loss, valid_weighted_accuracy



    def _test_loop(self, loader):

        self.model.eval()



        test_preds = []



        for (images, labels) in loader:

            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = self.model.forward(images)

            _, predicted = torch.max(outputs.data, 1)



            test_preds.append(predicted.cpu())



        test_preds = torch.cat(test_preds)



        return test_preds



    #-------



    def train_model(self):

        train_loader = self._build_loader(mode="train")

        valid_loader = self._build_loader(mode="valid")



#         scheduler = optim.lr_scheduler.MultiStepLR(

#             self.optimizer, milestones=[int(EPOCH * 0.8), int(EPOCH * 0.9)], gamma=0.1

#         )

        

        # scheduler examples: [http://katsura-jp.hatenablog.com/entry/2019/01/30/183501]

        # if you want to use cosine annealing, use below scheduler.

        scheduler = optim.lr_scheduler.CosineAnnealingLR(

            self.optimizer, T_max=EPOCH, eta_min=0.0001

        )

        

        for current_epoch in range(1, EPOCH + 1, 1):

            start_time = time.time()

            train_loss = self._train_loop(train_loader)

            valid_loss, valid_weighted_accuracy = self._valid_loop(valid_loader)



            print(

                "epoch: {} / ".format(current_epoch)

                + "train loss: {:.5f} / ".format(train_loss)

                + "valid loss: {:.5f} / ".format(valid_loss)

                + "valid w-acc: {:.5f} / ".format(valid_weighted_accuracy)

                + "lr: {:.5f} / ".format(self.optimizer.param_groups[0]["lr"])

                + "time: {}sec".format(int(time.time()-start_time))

            )



            self.train_loss_history.append(train_loss)

            self.valid_loss_history.append(valid_loss)

            self.valid_weighted_accuracy_history.append(valid_weighted_accuracy)



            scheduler.step()

            

    def make_submission_file(self):

        test_loader = self._build_loader(mode="test")

        test_preds = self._test_loop(test_loader)



        submission_df = pd.read_csv("../input/sample_submission.csv")

        submission_df["label"] = test_preds

        submission_df.to_csv("./submission.csv", index=False)



        print("---submission.csv---")

        print(submission_df.head())



    def plot_history(self):

        plt.figure(figsize=(20, 5))

        plt.subplot(1, 2, 1)

        plt.plot(

            np.arange(EPOCH) + 1,

            self.train_loss_history,

            label="train loss",

            color="red",

            linestyle="dashed",

            linewidth=3.0,

        )

        plt.plot(

            np.arange(EPOCH) + 1,

            self.valid_loss_history,

            label="valid loss",

            color="red",

            linestyle="solid",

            linewidth=3.0,

        )

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.ylim(0, 2)

        plt.legend()

        plt.grid()



        plt.subplot(1, 2, 2)

        plt.plot(

            np.arange(EPOCH) + 1,

            self.valid_weighted_accuracy_history,

            label="valid w-acc",

            color="red",

            linestyle="solid",

            linewidth=3.0,

            marker="o",

        )

        plt.xlabel("Epoch")

        plt.ylabel("Weighted Accuracy")

        plt.ylim(0, 1)

        plt.legend()

        plt.grid()



        plt.show()

        plt.savefig("./loss.png", dpi=100)
def check_augmentation():

    dataset = Imbalanced_CIFAR10_Dataset(mode="train", visualize=True)

    for i in range(10):

        plt.figure(figsize=(15,5))

        for j in range(8):

            plt.subplot(1,8,j+1)

            plt.imshow(dataset.__getitem__(i)[0].cpu().numpy().transpose(1,2,0))

        plt.tight_layout()

        plt.show()

    del dataset

    gc.collect()



def main():

    print("visualizing dataset...")

    check_augmentation()

    

    print("Initializing...")

    runner = Runner()

    

    print("Start training...")

    runner.train_model()

    runner.plot_history()

    

    print("Making submission file...")    

    runner.make_submission_file()



    

if __name__ == "__main__":

    main()