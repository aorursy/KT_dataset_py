import os

import math

import time

import random



import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

import seaborn as sn

import albumentations as A

import torch

from torch.utils import data as torch_data

from torch import nn as torch_nn

from torch.nn import functional as torch_functional

import torchvision

from sklearn import metrics as sk_metrics

from sklearn import model_selection as sk_model_selection
# The directory to the dataset

BASE_DIR = '../input/lego-minifigures-classification/'

PATH_INDEX = os.path.join(BASE_DIR, "index.csv")

PATH_TEST = os.path.join(BASE_DIR, "test.csv")

PATH_METADATA = os.path.join(BASE_DIR, "metadata.csv")
config = {

    "seed": 42,

    

    "valid_size": 0.3,

    "image_size": (512, 512),

    

    "train_batch_size": 4,

    "valid_batch_size": 1,

    "test_batch_size": 1,

    

    "model": "mobilenet_v2",

    

    "epochs": 50,

    "model_save_path": "model-best.torch",

    "patience_stop": 3,

    

    "optimizer": "adam",

    "adam_lr": 0.0001,

    

    "criterion": "cross_entropy",

}
# Try to set random seet that our experiment repeated between (We have some problem to set seed with GPU)

def set_seed(seed):

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True





set_seed(config["seed"])
# Read information about dataset

df = pd.read_csv(PATH_INDEX)



tmp_train, tmp_valid = sk_model_selection.train_test_split(

    df, 

    test_size=config["valid_size"], 

    random_state=config["seed"], 

    stratify=df['class_id'],

)



# Get train file paths

train_paths = tmp_train['path'].values

# Get train labels

train_targets = tmp_train['class_id'].values

# Create full train paths (base dir + concrete file)

train_paths = list(map(lambda x: os.path.join(BASE_DIR, x), train_paths))



# Get valid file paths

valid_paths = tmp_valid['path'].values

# Get valid labels

valid_targets = tmp_valid['class_id'].values

# Create full valid paths (base dir + concrete file)

valid_paths = list(map(lambda x: os.path.join(BASE_DIR, x), valid_paths))



df_test = pd.read_csv(PATH_TEST)

test_paths = df_test['path'].values

test_paths = list(map(lambda x: os.path.join(BASE_DIR, x), test_paths))

test_targets = df_test['class_id'].values
# Calculate the total number of classes in the dataset (len of unique labels in data)

df_metadata = pd.read_csv(PATH_METADATA)

n_classes = df_metadata.shape[0]

print("Number of classes: ", n_classes)
class DataRetriever(torch_data.Dataset):

    def __init__(

        self, 

        paths, 

        targets, 

        image_size=(224, 224),

        transforms=None

    ):

        self.paths = paths

        self.targets = targets

        self.image_size = image_size

        self.transforms = transforms

        self.preprocess = torchvision.transforms.Compose([

            torchvision.transforms.ToTensor(),

            torchvision.transforms.Normalize(

                mean=[0.485, 0.456, 0.406], 

                std=[0.229, 0.224, 0.225]

            ),

        ])

          

    def __len__(self):

        return len(self.targets)

    

    def __getitem__(self, index):

        img = cv2.imread(self.paths[index])

        img = cv2.resize(img, self.image_size)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:

            img = self.transforms(image=img)['image']

            

        img = self.preprocess(img)

        

        y = torch.tensor(self.targets[index] - 1, dtype=torch.long)

            

        return {'X': img, 'y': y}
def get_train_transforms():

    return A.Compose(

        [

            A.Rotate(limit=30, border_mode=cv2.BORDER_REPLICATE, p=0.5),

            A.Cutout(num_holes=8, max_h_size=20, max_w_size=20, fill_value=0, p=0.25),

            A.Cutout(num_holes=8, max_h_size=20, max_w_size=20, fill_value=1, p=0.25),

            A.HorizontalFlip(p=0.5),

            A.RandomContrast(limit=(-0.3, 0.3), p=0.5),

            A.RandomBrightness(limit=(-0.4, 0.4), p=0.5),

            A.Blur(p=0.25),

        ], 

        p=1.0

    )
train_data_retriever = DataRetriever(

    train_paths, 

    train_targets, 

    image_size=config["image_size"],

    transforms=get_train_transforms()

)



valid_data_retriever = DataRetriever(

    valid_paths, 

    valid_targets, 

    image_size=config["image_size"],

)



test_data_retriever = DataRetriever(

    test_paths, 

    test_targets, 

    image_size=config["image_size"],

)
train_loader = torch_data.DataLoader(

    train_data_retriever,

    batch_size=config["train_batch_size"],

    shuffle=True,

)



valid_loader = torch_data.DataLoader(

    valid_data_retriever, 

    batch_size=config["valid_batch_size"],

    shuffle=False,

)



test_loader = torch_data.DataLoader(

    test_data_retriever, 

    batch_size=config["test_batch_size"],

    shuffle=False,

)
def denormalize_image(image):

    return image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]



# Let's visualize some batches of the train data

plt.figure(figsize=(16, 16))

for i_batch, batch in enumerate(train_loader):

    images, labels = batch['X'], batch['y']

    for i in range(len(images)):

        plt.subplot(4, 4, 4 * i_batch + i + 1)

        plt.imshow(denormalize_image(images[i].permute(1, 2, 0).numpy()))

        plt.title(labels[i].numpy())

        plt.axis('off')

    if i_batch >= 3:

        break
# Let's visualize some batches of the train data

plt.figure(figsize=(16, 16))

for i_batch, batch in enumerate(valid_loader):

    images, labels = batch['X'], batch['y']

    plt.subplot(4, 4, i_batch + 1)

    plt.imshow(denormalize_image(images[0].permute(1, 2, 0).numpy()))

    plt.title(labels[0].numpy())

    plt.axis('off')

    if i_batch >= 15:

        break
def init_model_mobilenet_v2(n_classes):

    net = torch.hub.load("pytorch/vision:v0.6.0", "mobilenet_v2", pretrained=True)

    net.classifier = torch_nn.Linear(

        in_features=1280, 

        out_features=n_classes, 

        bias=True,

    )

    return net
class LossMeter:

    def __init__(self):

        self.avg = 0

        self.n = 0



    def update(self, val):

        self.n += 1

        # incremental update

        self.avg = val / self.n + (self.n - 1) / self.n * self.avg



        

class AccMeter:

    def __init__(self):

        self.avg = 0

        self.n = 0

        

    def update(self, y_true, y_pred):

        y_true = y_true.cpu().numpy().astype(int)

        y_pred = y_pred.cpu().numpy().argmax(axis=1).astype(int)

        last_n = self.n

        self.n += len(y_true)

        true_count = np.sum(y_true == y_pred)

        # incremental update

        self.avg = true_count / self.n + last_n / self.n * self.avg
class Trainer:

    def __init__(

        self, 

        model, 

        device, 

        optimizer, 

        criterion, 

        loss_meter, 

        score_meter

    ):

        self.model = model

        self.device = device

        self.optimizer = optimizer

        self.criterion = criterion

        self.loss_meter = loss_meter

        self.score_meter = score_meter

        

        self.best_valid_score = 0

        self.n_patience = 0

    

    def fit(self, epochs, train_loader, valid_loader, save_path, patience):

        for n_epoch in range(1, epochs + 1):

            self.info_message("EPOCH: {}", n_epoch)

            

            train_loss, train_score, train_time = self.train_epoch(train_loader)

            valid_loss, valid_score, valid_time = self.valid_epoch(valid_loader)

            

            m = '[Epoch {}: Train] loss: {:.5f}, score: {:.5f}, time: {} s'

            self.info_message(

                m, n_epoch, train_loss, train_score, train_time

            )

            m = '[Epoch {}: Valid] loss: {:.5f}, score: {:.5f}, time: {} s'

            self.info_message(

                m, n_epoch, valid_loss, valid_score, valid_time

            )

            

            if self.best_valid_score < valid_score:

                m = 'The score improved from {:.5f} to {:.5f}. Save model to "{}"'

                self.info_message(

                    m, self.best_valid_score, valid_score, save_path

                )

                self.save_model(n_epoch, save_path)

                self.best_valid_score = valid_score

                self.n_patience = 0

            else:

                self.n_patience += 1

            

            if self.n_patience >= patience:

                m = "\nValid score didn't improve last {} epochs."

                self.info_message(m, patience)

                break

            

    def train_epoch(self, train_loader):

        self.model.train()

        t = time.time()

        train_loss = self.loss_meter()

        train_score = self.score_meter()

        

        for step, batch in enumerate(train_loader, 1):

            images = batch['X'].to(self.device)

            targets = batch['y'].to(self.device)



            self.optimizer.zero_grad()

            outputs = self.model(images)



            loss = self.criterion(outputs, targets)

            loss.backward()



            train_loss.update(loss.detach().item())

            train_score.update(targets, outputs.detach())



            self.optimizer.step()

            

            _loss, _score = train_loss.avg, train_score.avg

            _time = int(time.time() - t)

            m = '[Train {}/{}] loss: {:.5f}, score: {:.5f}, time: {} s'

            self.info_message(

                m, step, len(train_loader), _loss, _score, _time, end='\r'

            )



        self.info_message('')

        

        return train_loss.avg, train_score.avg, int(time.time() - t)

    

    def valid_epoch(self, valid_loader):

        self.model.eval()

        t = time.time()

        valid_loss = self.loss_meter()

        valid_score = self.score_meter()



        for step, batch in enumerate(valid_loader, 1):

            with torch.no_grad():

                images = batch['X'].to(self.device)

                targets = batch['y'].to(self.device)



                outputs = self.model(images)

                loss = self.criterion(outputs, targets)



                valid_loss.update(loss.detach().item())

                valid_score.update(targets, outputs)



            _loss, _score = valid_loss.avg, valid_score.avg

            _time = int(time.time() - t)

            m = '[Valid {}/{}] loss: {:.5f}, score: {:.5f}, time: {} s'

            self.info_message(

                m, step, len(valid_loader), _loss, _score, _time, end='\r'

            )



        self.info_message('')

        

        return valid_loss.avg, valid_score.avg, int(time.time() - t)

    

    def save_model(self, n_epoch, save_path):

        torch.save(

            {

                'model_state_dict': self.model.state_dict(),

                'optimizer_state_dict': self.optimizer.state_dict(),

                'best_valid_score': self.best_valid_score,

                'n_epoch': n_epoch,

            },

            save_path,

        )

    

    @staticmethod

    def info_message(message, *args, end='\n'):

        print(message.format(*args), end=end)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if config["model"] == "mobilenet_v2":

    model = init_model_mobilenet_v2(n_classes)



model.to(device)



if config["optimizer"] == "adam":

    optimizer = torch.optim.Adam(model.parameters(), lr=config["adam_lr"])



if config["criterion"] == "cross_entropy":

    criterion = torch_functional.cross_entropy



trainer = Trainer(

    model, 

    device, 

    optimizer, 

    criterion, 

    LossMeter, 

    AccMeter

)

trainer.fit(

    config["epochs"], 

    train_loader, 

    valid_loader, 

    config["model_save_path"], 

    config["patience_stop"],

)
# Load the best model

checkpoint = torch.load(config["model_save_path"])



model.load_state_dict(checkpoint["model_state_dict"])

optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

best_valid_score = checkpoint["best_valid_score"]

n_epoch = checkpoint["n_epoch"]



model.eval()



print(f"Best model valid score: {best_valid_score} ({n_epoch} epoch)")
# Save the model predictions and true labels

y_pred = []

y_test = []

for batch in test_loader:

    y_pred.extend(model(batch['X'].to(device)).argmax(axis=-1).cpu().numpy())

    y_test.extend(batch['y'])



# Calculate needed metrics

print(f'Accuracy score on test data:\t{sk_metrics.accuracy_score(y_test, y_pred)}')

print(f'Macro F1 score on test data:\t{sk_metrics.f1_score(y_test, y_pred, average="macro")}')
# Load metadata to get classes people-friendly names

labels = df_metadata['minifigure_name'].tolist()



# Calculate confusion matrix

confusion_matrix = sk_metrics.confusion_matrix(y_test, y_pred)

# confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)

df_confusion_matrix = pd.DataFrame(confusion_matrix, index=labels, columns=labels)



# Show confusion matrix

plt.figure(figsize=(12, 12))

sn.heatmap(df_confusion_matrix, annot=True, cbar=False, cmap='Oranges', linewidths=1, linecolor='black')

plt.xlabel('Predicted labels', fontsize=15)

plt.xticks(fontsize=12)

plt.ylabel('True labels', fontsize=15)

plt.yticks(fontsize=12);
error_images = []

error_label = []

error_pred = []

error_prob = []

for batch in test_loader:

    _X_test, _y_test = batch['X'], batch['y']

    pred = torch.softmax(model(_X_test.to(device)), axis=-1).detach().cpu().numpy()

    pred_class = pred.argmax(axis=-1)

    if pred_class != _y_test.cpu().numpy():

        error_images.extend(_X_test)

        error_label.extend(_y_test)

        error_pred.extend(pred_class)

        error_prob.extend(pred.max(axis=-1))
plt.figure(figsize=(16, 16))

for ind, image in enumerate(error_images):

    plt.subplot(math.ceil(len(error_images) / int(len(error_images) ** 0.5)), int(len(error_images) ** 0.5), ind + 1)

    plt.imshow(denormalize_image(image.permute(1, 2, 0).numpy()))

    plt.title(f'Predict: {labels[error_pred[ind]]} ({error_prob[ind]:.2f}) Real: {labels[error_label[ind]]}')

    plt.axis('off')