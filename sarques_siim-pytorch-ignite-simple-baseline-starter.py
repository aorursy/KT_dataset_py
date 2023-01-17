# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

flag = 0

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if flag == 5:

            break

        print(os.path.join(dirname, filename))

        flag += 1

    if flag ==5:

        break



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from PIL import Image



import torch

import torch.nn as nn

import torch.utils.data as D

from torch.optim.lr_scheduler import ExponentialLR



from torchvision import models as M, transforms as T



from tqdm import tqdm_notebook



from glob import glob

from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings("ignore")



from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

from ignite.metrics import Loss, Accuracy

from ignite.contrib.handlers.tqdm_logger import ProgressBar

from ignite.handlers import ModelCheckpoint, EarlyStopping 
!pip install efficientnet_pytorch
from efficientnet_pytorch import EfficientNet
os.listdir("/kaggle/input/siim-isic-melanoma-classification/jpeg/")
BASE_PATH = "/kaggle/input/siim-isic-melanoma-classification"

df_train = pd.read_csv(BASE_PATH + "/train.csv")

df_test = pd.read_csv(BASE_PATH + "/test.csv")

df_sub = pd.read_csv(BASE_PATH + "/sample_submission.csv")
temp = plt.imread(BASE_PATH + "/jpeg/train/ISIC_4232172.jpg")

plt.xticks([])

plt.yticks([])

plt.imshow(temp)
batch_size = 32

device = "cuda"

torch.manual_seed(0)
temp.shape
BINGO_PATH = "/kaggle/input/siic-isic-224x224-images"
class ImagesDS(D.Dataset):

    def __init__(self, df, dir, mode = "train"):

        self.records = df.to_records(index = False)

        self.mode = mode

        self.dir = dir

        self.len = df.shape[0]

        

    @staticmethod

    def _load_img_as_tensor(filename):

        with Image.open(filename) as img:

            return T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])(img)

        

    def _get_image_path(self, index):

        image_id = self.records[index].image_name

        return "/".join([self.dir, self.mode, f"{image_id}.png"])

    

    def __getitem__(self, index):

        path = self._get_image_path(index)

        img = self._load_img_as_tensor(path)

        if self.mode == "train":

            return img, self.records[index].target

        else:

            return img, self.records[index].image_name

        

    def __len__(self):

        return self.len
train_data, val_data = train_test_split(df_train, test_size = 0.2, random_state = 42)
ds = ImagesDS(train_data, BINGO_PATH, mode = "train")

ds_val = ImagesDS(val_data, BINGO_PATH, mode = "train")

ds_test = ImagesDS(df_test, BINGO_PATH, mode = "test")
classes = 1

model = EfficientNet.from_pretrained("efficientnet-b0")

model.fc = nn.Linear(1280, classes, bias = True)
loader = D.DataLoader(ds, batch_size = 64, shuffle = True, num_workers = 4)

val_loader = D.DataLoader(ds_val, batch_size = batch_size, shuffle = True, num_workers = 4)

test_loader = D.DataLoader(ds_test, batch_size = batch_size, shuffle = False, num_workers = 4)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4, weight_decay = 0.00001)
metrics = {"loss" : Loss(criterion), "accuracy" : Accuracy()}



trainer = create_supervised_trainer(model, optimizer, criterion, device = device)

val_eval = create_supervised_evaluator(model, metrics = metrics, device = device)
@trainer.on(Events.EPOCH_COMPLETED)

def compute_and_display(engine):

    epoch = engine.state.epoch

    metrics = val_eval.run(val_loader).metrics

    print("Validation Results - Epoch : {} Average loss : {:.4f} Average Accuracy : {:.4f}".format(engine.state.epoch, metrics["loss"], metrics["accuracy"]))
handler = EarlyStopping(patience = 4, score_function = lambda engine : engine.state.metrics["accuracy"], trainer = trainer)



val_eval.add_event_handler(Events.COMPLETED, handler)
checkpoints = ModelCheckpoint("models", "Model", n_saved = 3, create_dir = True)



trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoints, {"Konoha_senpu" : model})
pbar = ProgressBar(bar_format = "")



pbar.attach(trainer, output_transform = lambda x : {"loss" : x})
trainer.run(loader, max_epochs = 15)
model.eval()

test_preds = np.zeros((df_test.shape[0], ))

with torch.no_grad():

    for i, data in enumerate(tqdm_notebook(test_loader, position=0, leave=True)):

        images, _ = data

        images = images.to(device)

        output = model(images)

        output = torch.softmax(output,1).cpu().detach().numpy()[:,1]

        test_preds[i*batch_size : (i+1)*batch_size] = output
test_preds
df_sub.target = test_preds
df_sub.to_csv("submission.csv", index = False)