!pip install ../input/facenet-pytorch-vggface2/facenet_pytorch-2.0.1-py3-none-any.whl

!mkdir -p /root/.cache/torch/checkpoints/

!cp ../input/facenet-pytorch-vggface2/20180402-114759-vggface2-logits.pth /root/.cache/torch/checkpoints/vggface2_DG3kwML46X.pt

!cp ../input/facenet-pytorch-vggface2/20180402-114759-vggface2-features.pth /root/.cache/torch/checkpoints/vggface2_G5aNV2VSMn.pt
import numpy as np

import pandas as pd

import os

import gc

from glob import glob

import json

import seaborn as sns

import matplotlib.pyplot as plt

import random

import cv2

from PIL import Image, ImageDraw

from tqdm.notebook import tqdm

from collections import defaultdict, deque

import sys



import torch

from torch.nn import Module

from torch import nn

from torch import optim

from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import DataLoader, Dataset, Subset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



from facenet_pytorch import MTCNN, InceptionResnetV1
submission_path = '../input/deepfake-detection-challenge/sample_submission.csv'

train_video_path = '../input/deepfake-detection-challenge/train_sample_videos'

test_video_path = '../input/deepfake-detection-challenge/test_videos'
list_train = glob(os.path.join(train_video_path, '*.mp4'))

print(f'Sum video in train: {len(list_train)}')
list_test = glob(os.path.join(test_video_path, '*.mp4'))

print(f'Sum video in test: {len(list_test)}')
train_json = glob(os.path.join(train_video_path, '*.json'))

with open(train_json[0], 'rt') as file:

    train = json.load(file)

    

train_df = pd.DataFrame()

train_df['file'] = train.keys()



label = [i['label'] for i in train.values() if isinstance(i, dict)]

train_df['label'] = label



split = [i['split'] for i in train.values() if isinstance(i, dict)]

train_df['split'] = split



original = [i['original'] for i in train.values() if isinstance(i, dict)]

train_df['original'] = original



train_df['original'] = train_df['original'].fillna(train_df['file'])

train_df.head()
# train_df = train_df.iloc[:50, :]
real = train_df[train_df['label']=='REAL']

real.reset_index(inplace=True, drop=True)

fake = train_df[train_df['label']=='FAKE']

fake.reset_index(inplace=True, drop=True)



plt.figure(figsize=(15,8))

ax = sns.countplot(y=label, data=train_df)



for p in ax.patches:

    ax.annotate('{:.2f}%'.format(100*p.get_width()/train_df.shape[0]), (p.get_x() + p.get_width() + 0.02, p.get_y() + p.get_height()/2))

    

plt.title('Distribution of label', size=25, color='b')    

plt.show()
original_same = train_df.pivot_table(values=['file'], columns=['label'], index=['original'], fill_value=0, aggfunc='count')

original_same = original_same[(original_same[('file', 'FAKE')] != 0) & (original_same[('file', 'REAL')] != 0)]



print(f'Number of file having both FAKE and REAL: {len(original_same)}')

original_same
train_df['label'] = train_df['label'].apply(lambda x: 1 if x=='FAKE' else 0)
def box_mtcnn(frame, landmarks=True):

    mtcnn = MTCNN(margin=14, keep_all=True, device=device, factor=0.5).to(device)

    if landmarks:

        boxes, scores, landmarks = mtcnn.detect(frame, landmarks=landmarks)

        return boxes, scores, landmarks

    else:

        boxes, scores = mtcnn.detect(frame, landmarks=landmarks)

        return boxes, scores
def display_video(df, number_frame=5, number_video=3):

    

    color = ['b', 'g', 'r']

    for index in range(number_video):

        

        index_random = random.randint(0, len(df))

        video = df.loc[index_random, 'file']

        

        if video in os.listdir(train_video_path):

            video_path = os.path.join(train_video_path, video)

            cap = cv2.VideoCapture(video_path)

            

            fig, axes = plt.subplots(number_frame, 2, figsize=(20, 20))

            

            frame_index = 0

            ax_ix = 0

            while True:

                    

                ret, frame = cap.read()

                

                if cv2.waitKey(1) & 0xFF == 27:

                    break

                

                if ret:                    

                    

                    if frame_index%24==0:

                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        image = Image.fromarray(image)

                        boxes, scores = box_mtcnn(image, False)

                        if scores[0]:

                            if boxes is not None:

                                box = boxes[scores.argmax()]

                                frame_crop = image.crop(box)

                                frame_crop = np.array(frame_crop)



                                for i in range(3):

                                    hist = cv2.calcHist([frame_crop], [i], None, [256], [0, 256])

                                    axes[ax_ix, 1].plot(hist, color=color[i])





                                axes[ax_ix, 0].imshow(frame_crop)

                                axes[ax_ix, 0].xaxis.set_visible(False)

                                axes[ax_ix, 0].yaxis.set_visible(False)

                                axes[ax_ix, 0].set_title(f'Frame: {frame_index}')

                                ax_ix += 1



                                fig.tight_layout()



                                fig.suptitle(video, color='b', size=20, y=1)



                                if ax_ix == number_frame:

                                    break

                                                                

                else:

                    break

                    

                

                frame_index += 1          

        

display_video(fake)
display_video(real)
def transfer(label):

    

    if label==0:

        return "real"

    else:

        return "fake"

    

def display_mtcnn(number_frame=3, number_video=2):

    

    fake_real = original_same[(original_same[('file', 'FAKE')] == 1) & (original_same[('file', 'REAL')] == 1)].index.tolist()                

    original_images = random.sample(fake_real, number_video)

    

    for original_image in original_images:

        real_video = train_df[(train_df['label']==0) & (train_df['original']==original_image)]['file'].values[0]

        fake_video = train_df[(train_df['label']==1) & (train_df['original']==original_image)]['file'].values[0]



        if (real_video in os.listdir(train_video_path)) and (fake_video in os.listdir(train_video_path)):

            real_path = os.path.join(train_video_path, real_video)

            fake_path = os.path.join(train_video_path, fake_video)







            fig, axes = plt.subplots(number_frame, 2, figsize=(20, 20))



            for ind, path in enumerate([real_path, fake_path]):



                cap = cv2.VideoCapture(path)

                frame_index = 0

                ax_ix = 0

                

                while True:

                    ret, frame = cap.read()



                    if cv2.waitKey(1) & 0xFF == 27:

                        break



                    if ret:                    



                        if frame_index%24==0:

                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                            frame = Image.fromarray(frame)

                            boxes, scores = box_mtcnn(frame, False)

                            box = boxes[scores.argmax()]

                            frame_crop = frame.crop(box)

                            

                            boxes, scores, landmarks = box_mtcnn(frame_crop)

                            if landmarks is not None:

                                landmark = landmarks[scores.argmax()]

                                axes[ax_ix, ind].scatter(landmark[:, 0], landmark[:, 1], c='red', s=8)



                                axes[ax_ix, ind].imshow(frame_crop)

                                axes[ax_ix, ind].xaxis.set_visible(False)

                                axes[ax_ix, ind].yaxis.set_visible(False)

                                axes[ax_ix, ind].set_title(f'Frame: {frame_index}_{transfer(ind)}')



                                fig.tight_layout()

                                ax_ix += 1



                                if ax_ix == number_frame:

                                    break



                    else:

                        break

                    

                    frame_index+=1

                    

            fig.suptitle(original_image, color='b', size=20, y=1)





display_mtcnn(number_frame=3, number_video=3)
class VideoDataset(Dataset):

    

    def __init__(self, df, path_video, num_frame=5, is_train=True):

        super(VideoDataset, self).__init__()

        

        self.df = df

        self.num_frame = num_frame

        self.is_train = is_train

        self.path_video = path_video

        

        index_list = deque()

        for index in tqdm(range(len(self.df))):

            

            video_name = self.df.loc[index, 'file']

            video_path = os.path.join(self.path_video, video_name)

            

            if self.landmark_mtcnn(video_path) is not None:

                index_list.append(index)

                

        index_list = list(index_list)

        self.df = self.df[self.df.index.isin(index_list)]

        self.df.reset_index(inplace=True, drop=True)

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        

        video_name = self.df.loc[idx, 'file']

        video_path = os.path.join(self.path_video, video_name)

        list_landmark = self.landmark_mtcnn(video_path)

        

        if self.is_train:

            label = self.df.loc[idx, 'label']

            return torch.from_numpy(list_landmark), torch.tensor(label, dtype=torch.float)

        else:

            return video_name, torch.from_numpy(list_landmark)

        

        

    def landmark_mtcnn(self, video_path):



        cap = cv2.VideoCapture(video_path)

        frame_index = 0



        list_landmark = deque()

        

        while len(list_landmark) < 10*self.num_frame:

            

                

            ret, frame = cap.read()



            if cv2.waitKey(1) & 0xFF == 27:

                break



            if ret:                    

                if frame_index % 24 == 0:

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    frame = Image.fromarray(frame)

                    boxes, scores, landmarks = box_mtcnn(frame)



                    if scores[0]:

                        index_max = np.argmax(scores)

                        landmark = landmarks[index_max]



                        list_landmark.extend(landmark.flatten())



            else:

                break



            frame_index+=1

        

        list_landmark = list(list_landmark)

        if len(list_landmark) == 10*self.num_frame:

            list_landmark = np.array(list_landmark).reshape(self.num_frame, 10)



            return list_landmark

        return None

    

    

dataset = VideoDataset(train_df, train_video_path)
test_size = 0.2

index_split = int(len(dataset)*test_size)

list_index = (list(range(len(dataset))))

random.shuffle(list_index)



train_idx = list_index[index_split:]

val_idx = list_index[:index_split]



train_dataset = Subset(dataset, train_idx)

val_dataset = Subset(dataset, val_idx)



train_ld = DataLoader(train_dataset, batch_size=8, shuffle=True)

val_ld = DataLoader(val_dataset, batch_size=8, shuffle=True)
class swish(Module):

  

  def __init__(self):

    super(swish, self).__init__()

    

    self.sigmoid = nn.Sigmoid()

    

  def forward(self, x):

    return x*self.sigmoid(x)





class small_model(Module):

    def __init__(self, num_class=1):

        super(small_model, self).__init__()

        

        self.conv = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),

                                  nn.BatchNorm2d(32),

                                  nn.ReLU(inplace=True),

                                  nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),

                                  nn.BatchNorm2d(64),

                                  nn.ReLU(inplace=True),

                                  nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),

                                  nn.BatchNorm2d(64),

                                  nn.Dropout(0.2))

        

        self.fc = nn.Sequential(nn.Linear(64*10*5, 128),

                                nn.ReLU(inplace=True),

                                nn.BatchNorm1d(128),

                                nn.Dropout(0.2),

                                nn.Linear(128, num_class),

                                nn.Sigmoid())

        

        

    def forward(self, x):

        

        x = self.conv(x)

        x = torch.flatten(x, 1)        

        x = self.fc(x)

        

        return x

    

model = small_model().to(device)

model.eval()    
class Trainer(object):

    

    def __init__(self, model):

        

        self.model = model

        self.creation = nn.MSELoss()

        

        self.optimizer = optim.AdamW([      

            {'params': model.conv.parameters(), 'lr': 1e-4},

            {'params': model.fc.parameters(), 'lr': 1e-3}], lr=0.001)

        

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.15)

        

    def train_process(self, train_ld, val_ld, epochs):

        score_max = 0

        check_step = 0

        loss_min = 1

        check_number = 13

        

        self.model.train()

        for epoch in range(epochs):

            train_loss, val_loss = 0, 0

            for crop, label in tqdm(train_ld):                

                crop = crop.unsqueeze(1)

                crop, label = crop.float().to(device), label.to(device)

                                

                self.optimizer.zero_grad()

                output = self.model(crop).squeeze(1)

                loss = self.creation(output, label)

                loss.backward()

                                

                self.optimizer.step()

                self.scheduler.step(train_loss)

                train_loss += loss.item()

                

                del crop, label

            

            train_loss = train_loss/len(train_ld)

            torch.cuda.empty_cache()

            

            gc.collect()

            

            self.model.eval()

            

            val_score = 0

            with torch.no_grad():

                for crop, label in tqdm(val_ld):

                    crop = crop.unsqueeze(1)

                    crop, label = crop.float().to(device), label.to(device)                    

                    

                    output = self.model(crop).squeeze(1)

                    loss = self.creation(output, label)

                    val_loss += loss.item()

                    val_score += torch.sum((output>0.5).float() == label).item()/len(label)

                

                val_loss = val_loss/len(val_ld)

                val_score = val_score/len(val_ld)

                

            self.scheduler.step(val_loss)

            

            if val_score > score_max:

                print(f'Epoch: {epoch}, train loss: {train_loss:.5f}, val_loss: {val_loss:.5f}.\nValidation score increased from {score_max:.5f} to {val_score:.5f}')

                score_max = val_score

                loss_min = val_loss

                torch.save(self.model.state_dict(), 'model.pth')

                print('Saving model!')

                check_step = 0

                

            elif val_score == score_max:

                if val_loss < loss_min:

                    print(f'Epoch: {epoch}, train loss: {train_loss:.5f}, val_loss: {val_loss:.5f}, val_score: {val_score:.5f}.\nValidation loss decreased from {loss_min:.5f} to {val_loss:.5f}')

                    loss_min = val_loss

                    torch.save(self.model.state_dict(), 'model.pth')

                    print('Saving model!')

                    check_step = 0

                else:

                    check_step += 1

                    print(f'Epoch: {epoch}, train loss: {train_loss:.5f}, val_loss: {val_loss:.5f}, val_score: {val_score:.5f}.\nModel not improve in {str(check_step)} step')

                    if check_step > check_number:

                        print('Stop trainning!')

                        break

            else:

                check_step += 1

                print(f'Epoch: {epoch}, train loss: {train_loss:.5f}, val_loss: {val_loss:.5f}.\nValidation score not increased from {val_score:.5f} in {str(check_step)} step')

                

                if check_step > check_number:

                    print('Stop trainning!')

                    break

                    

trainer = Trainer(model)        

trainer.train_process(train_ld=train_ld, val_ld=val_ld, epochs=20)