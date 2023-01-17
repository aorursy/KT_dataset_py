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

from albumentations import Compose, Normalize

from PIL import Image, ImageDraw

from tqdm.notebook import tqdm

from collections import defaultdict, deque

import sys

sys.path.append('../input/pretrainedmodels/pretrainedmodels-0.7.4')



import torch

from torch.nn import Module



from torch import nn

from torchvision.models import resnext50_32x4d as resnext50

from torch import optim

from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import DataLoader, Dataset, Subset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



from facenet_pytorch import MTCNN, InceptionResnetV1

from pretrainedmodels import xception
submission_path = '../input/deepfake-detection-challenge/sample_submission.csv'

train_video_path = '../input/deepfake-detection-challenge/train_sample_videos'

test_video_path = '../input/deepfake-detection-challenge/test_videos'



num_frame = 5

dim_trans = 5
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
submission = pd.read_csv(submission_path)

submission.head()
# submission = submission.iloc[:10, :]
def box_mtcnn(frame, landmarks=True):

    mtcnn = MTCNN(keep_all=True, device=device)   

    if landmarks:

        boxes, scores, landmarks = mtcnn.detect(frame, landmarks=landmarks)

        return boxes, scores, landmarks

    else:

        boxes, scores = mtcnn.detect(frame, landmarks=landmarks)

        return boxes, scores
def op_display(df, number_frame=5, number_video=3):

    

    for index in range(number_video):

        

        index_random = random.randint(0, len(df))

        video = df.loc[index_random, 'file']

        

        if video in os.listdir(train_video_path):

            video_path = os.path.join(train_video_path, video)

            cap = cv2.VideoCapture(video_path)

            

            fig, axes = plt.subplots(number_frame, 2, figsize=(20, 20))

            

            frame_index = 0

            ax_ix = 0

            previous_crop = ''

            

            while True:                    

                ret, frame = cap.read()

                

                if cv2.waitKey(1) & 0xFF == 27:

                    break

                

                if ret:                    

                    

                    if frame_index%24==0:

                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        image = Image.fromarray(image)

                        boxes, scores = box_mtcnn(image, False)

                        try:

                            if scores[0]:

                                if boxes is not None:

                                    box = boxes[scores.argmax()]

                                    frame_crop = image.crop(box)

                                    frame_crop = frame_crop.resize((64,64), Image.ANTIALIAS)

                                    frame_crop = np.array(frame_crop)

                                    frame_crop_gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)



                                    if len(previous_crop) != 0: 

                                        flow = cv2.calcOpticalFlowFarneback(previous_crop, frame_crop_gray,

                                                                            None, 0.5, 5, 11, 5, 5, 1.1, 0)

                                        axes[ax_ix, 1].imshow(previous_crop)

                                        axes[ax_ix, 0].imshow(frame_crop)

                                        axes[ax_ix, 0].xaxis.set_visible(False)

                                        axes[ax_ix, 0].yaxis.set_visible(False)

                                        axes[ax_ix, 0].set_title(f'Frame: {frame_index}')

                                        ax_ix += 1



                                        fig.tight_layout()



                                        fig.suptitle(video, color='b', size=20, y=1)



                                    previous_crop = frame_crop_gray



                                    if ax_ix == number_frame:

                                        break

                        except:

                            continue

                else:

                    break

                    

                

                frame_index += 1          

        

op_display(fake)
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
def display_mtcnn(number_frame=3, number_video=2):

    

    fake_real = original_same[(original_same[('file', 'FAKE')] == 1) & (original_same[('file', 'REAL')] == 1)].index.tolist()                

    original_images = random.sample(fake_real, number_video)

    

    for original_image in original_images:

        real_video = train_df[(train_df['label']==0) & (train_df['original']==original_image)]['file'].values[0]

        fake_video = train_df[(train_df['label']==1) & (train_df['original']==original_image)]['file'].values[0]



        if (real_video in os.listdir(train_video_path)) and (fake_video in os.listdir(train_video_path)):

            real_path = os.path.join(train_video_path, real_video)

            fake_path = os.path.join(train_video_path, fake_video)







            fig, axes = plt.subplots(number_frame, 2, figsize=(40, 30))



            for ind, path in enumerate([real_path, fake_path]):



                cap = cv2.VideoCapture(path)

                frame_index = 0

                ax_ix = 0

                

                while frame_index < (10*number_frame - 9):

                    ret, frame = cap.read()



                    if cv2.waitKey(1) & 0xFF == 27:

                        break



                    if ret:                    



                        if frame_index in 10*np.arange(0, number_frame):

                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                            frame = Image.fromarray(frame)

                            boxes, scores, landmarks = box_mtcnn(frame)

                            draw = ImageDraw.Draw(frame)

                            for box, score, landmark in zip(boxes, scores, landmarks):

                                if score > 0.91:

                                    draw.rectangle(box.tolist(), outline=(0, 255, 0), width=6)

                                    axes[ax_ix, ind].scatter(landmark[:, 0], landmark[:, 1], c='red', s=8)

                                                            

                            axes[ax_ix, ind].imshow(frame)

                            axes[ax_ix, ind].xaxis.set_visible(False)

                            axes[ax_ix, ind].yaxis.set_visible(False)

                            axes[ax_ix, ind].set_title(f'Frame {frame_index}')

                            

                            fig.tight_layout()

                            ax_ix += 1



                    else:

                        break

                    

                    frame_index+=1

                    

            fig.suptitle(original_image, color='b', size=20, y=1)





display_mtcnn(number_frame=3, number_video=3)
class VideoDataset(Dataset):

    

    def __init__(self, df, path_video, num_frame=20, is_train=True, transforms=None):

        super(VideoDataset, self).__init__()

        

        self.df = df

        self.num_frame = num_frame

        self.is_train = is_train

        self.path_video = path_video

        self.transforms = transforms

        

        index_list = deque()

        for index in tqdm(range(len(self.df))):

            

            video_name = self.df.loc[index, 'filename']

            video_path = os.path.join(self.path_video, video_name)

            

            try:

                if self.frame_crop(video_path) is not None:

                    index_list.append(index)

            except:

                continue

                

        index_list = list(index_list)

        self.df = self.df[self.df.index.isin(index_list)]

        self.df.reset_index(inplace=True, drop=True)

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        

        video_name = self.df.loc[idx, 'filename']

        video_path = os.path.join(self.path_video, video_name)

        list_frame = self.frame_crop(video_path)         

        

        if self.is_train:

            label = self.df.loc[idx, 'label']

            return torch.from_numpy(list_frame), torch.tensor(label, dtype=torch.float)

        else:

            return video_name, torch.from_numpy(list_frame)

        

        

    def frame_crop(self, video_path):



        cap = cv2.VideoCapture(video_path)

        frame_index = 0

        list_frame = []

        

        while True:

            ret, frame = cap.read()

            if cv2.waitKey(1) & 0xFF == 27:

                break

            if ret:                    

                if frame_index % 12 == 0:

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    

                    frame = Image.fromarray(frame)

                    boxes, scores = box_mtcnn(frame, False)

                    

                    try:

                        if scores[0]:

                            if boxes is not None:                                

                                index_max = np.argmax(scores)

                                box = boxes[index_max]

                                frame_crop = frame.crop(box)

                                frame_crop = frame_crop.resize((150,150), Image.ANTIALIAS)

                                frame_crop = np.array(frame_crop)     

                                if self.transforms is not None:

                                    frame_crop = self.transforms(image=frame_crop)['image']

                                                                

                                list_frame.append(frame_crop)                                

                                if len(list_frame) == self.num_frame: 

                                    return np.array(list_frame)

                    except:

                        continue

            else:

                return None



            frame_index+=1    

        return None

    

normalize = Compose([

    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1)

])

    

test_dataset = VideoDataset(submission, test_video_path, num_frame=num_frame, is_train=False, transforms=normalize)
test_ld = DataLoader(test_dataset, batch_size=1, shuffle=False)
class EncoderCNN(Module):



  def __init__(self, encoder_output_dim):

    super(EncoderCNN, self).__init__()



    # self.conv1 = EfficientNet.from_pretrained('efficientnet-b1')    

    self.conv1 = xception(pretrained=None, num_classes=1000)

    

    for param in self.conv1.parameters():

        param.requires_grad = False



    self.conv2 = nn.Sequential(nn.Linear(1000, encoder_output_dim),

                               nn.ReLU(inplace=True),

                               nn.Sigmoid())





  def forward(self, x_3d):

    cnn_embed_seq = []



    for i in range(x_3d.size(1)):

      

        x = self.conv1(x_3d[:, i, :, :, :].squeeze(1))

        x = self.conv2(x)

        cnn_embed_seq.append(x)

      

    cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

    return cnn_embed_seq





class DecoderRNN(Module):



  def __init__(self, encoder_output_dim, hidden_size=512, num_class=1):

    super(DecoderRNN, self).__init__()



    self.hidden_size = hidden_size

    self.lstm = nn.LSTM(input_size=encoder_output_dim, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)

    self.fc2 = nn.Linear(4*hidden_size, num_class)



  def forward(self, x_RNN):



    h0 = torch.zeros(2, x_RNN.size(0), self.hidden_size).requires_grad_()

    c0 = torch.zeros(2, x_RNN.size(0), self.hidden_size).requires_grad_()



    rnn_out, _ = self.lstm(x_RNN, (h0.to(device), c0.to(device)))

    rnn_max, _ = torch.max(rnn_out, 1)

    rnn_mean = torch.mean(rnn_out, 1)

    out = torch.cat([rnn_max, rnn_mean], axis=1)    

    out = self.fc2(out)



    return out.squeeze()



  

class lrcn(Module):



  def __init__(self, encoder_output_dim, num_class=1):

    super(lrcn, self).__init__()



    self.cnn = EncoderCNN(encoder_output_dim=encoder_output_dim)

    self.rnn = DecoderRNN(encoder_output_dim=encoder_output_dim, num_class=num_class)



  def forward(self, x_3d):

    x = self.rnn(self.cnn(x_3d))



    return x





def load_model(path, encoder_output_dim=dim_trans, load_weight=True):



    model = lrcn(encoder_output_dim=encoder_output_dim, num_class = 1)

    

    if load_weight:

        for weight in sorted(os.listdir(path)):

            if 'pth' in weight:                

                weight_path = os.path.join(path, weight)                

                state = torch.load(weight_path, map_location=lambda storage, loc: storage)

                return state

    else:

        return model.state_dict()



base_model = load_model('../input/facenet-pretrained', load_weight=True)
def focal_loss(pred, expected, alpha=0.25, gamma=2):

    

    ce = f.binary_cross_entropy(pred, expected)

    pt = torch.exp(-ce)

    fc = alpha*((1-pt)**gamma)*ce

    

    return torch.mean(fc)



def loss_fn(pred, expected):

    return 0.1*f.mse_loss(torch.sigmoid(pred), expected) + 0.9*focal_loss(pred, expected)
class Trainer(object):

    

    def __init__(self, base_model):

        

        self.base_model = base_model

        self.model = lrcn(encoder_output_dim=dim_trans, num_class = 1).to(device)

        self.creation = loss_fn        

        

    def train_process(self, folds, dataset, epochs):                

        for fold, (train_idx, val_idx) in enumerate(KFold(n_splits=5, shuffle=True, random_state=41).split(dataset)):

            print(f'fold {fold}:')

            train_dataset = Subset(dataset, train_idx)

            val_dataset = Subset(dataset, val_idx)

            train_ld = DataLoader(train_dataset, batch_size=8, shuffle=True)

            val_ld = DataLoader(val_dataset, batch_size=8, shuffle=True)

            

            del train_dataset, val_dataset            

            self.model.load_state_dict(self.base_model[fold])

            

            optimizer = optim.AdamW([      

            {'params': self.model.conv.parameters(), 'lr': 1e-4},

            {'params': self.model.fc.parameters(), 'lr': 1e-3}], lr=0.001)

        

            scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_ld), epochs=3)

            

            self.model.train()

            

            score_max = 0

            check_step = 0

            loss_min = 1

            check_number = 10

            

            for epoch in range(epochs):

                train_loss, val_loss = 0, 0

                for crop, label in tqdm(train_ld):                

                    crop = crop.permute(0, 3, 1, 2)

                    crop, label = crop.float().to(device), label.to(device)



                    optimizer.zero_grad()

                    output = self.model(crop).squeeze(1)

                    loss = self.creation(output, label)

                    loss.backward()



                    optimizer.step()

                    scheduler.step(train_loss)

                    train_loss += loss.item()



                    del crop, label



                train_loss = train_loss/len(train_ld)

                torch.cuda.empty_cache()



                gc.collect()



                self.model.eval()



                val_score = 0

                with torch.no_grad():

                    for crop, label in tqdm(val_ld):

                        crop = crop.permute(0, 3, 1, 2)

                        crop, label = crop.float().to(device), label.to(device)                    



                        output = self.model(crop).squeeze(1)

                        loss = self.creation(output, label)

                        val_loss += loss.item()

                        val_score += torch.sum((output>0.5).float() == label).item()/len(label)



                    val_loss = val_loss/len(val_ld)

                    val_score = val_score/len(val_ld)



                scheduler.step(val_loss)



                if val_score > score_max:

                    print(f'\tEpoch: {epoch}, train loss: {train_loss:.5f}, val_loss: {val_loss:.5f}.\n\tValidation score increased from {score_max:.5f} to {val_score:.5f}')

                    score_max = val_score

                    loss_min = val_loss

                    torch.save(self.model.state_dict(), f'model_{str(fold)}.pth')

                    print('\tSaving model!')

                    check_step = 0



                elif val_score == score_max:

                    if val_loss < loss_min:

                        print(f'\tEpoch: {epoch}, train loss: {train_loss:.5f}, val_loss: {val_loss:.5f}, val_score: {val_score:.5f}.\n\tValidation loss decreased from {loss_min:.5f} to {val_loss:.5f}')

                        loss_min = val_loss

                        torch.save(self.model.state_dict(), f'model_{str(fold)}.pth')

                        print('\tSaving model!')

                        check_step = 0

                    else:

                        check_step += 1

                        print(f'\tEpoch: {epoch}, train loss: {train_loss:.5f}, val_loss: {val_loss:.5f}, val_score: {val_score:.5f}.\n\tModel not improve in {str(check_step)} step')

                        if check_step > check_number:

                            print('\tStop trainning!')

                            break

                else:

                    check_step += 1

                    print(f'\tEpoch: {epoch}, train loss: {train_loss:.5f}, val_loss: {val_loss:.5f}.\n\tValidation score not increased from {val_score:.5f} in {str(check_step)} step')



                    if check_step > check_number:

                        print('\tStop trainning!')

                        break

                        

            del optimizer, scheduler, train_ld, val_ld

            torch.cuda.empty_cache()

            

            gc.collect()

            

                    

    def predict_process(self, test_ld, submission):

                        

        submission['label'] = 0.6

        self.model.load_state_dict(self.base_model)



        self.model.eval()

        for filename, crops in tqdm(test_ld):



            crops = crops.permute(0, 1, 4, 2, 3)

            crops = crops.float().to(device)

            with torch.no_grad():

                output = self.model(crops)

                output = torch.sigmoid(output).cpu().detach().numpy() 

                    

                submission.loc[(submission[submission['filename']==filename[0]]).index.values[0], 'label'] = np.clip(output, 0.1, 0.9)

                                

        return submission

                    

trainer = Trainer(base_model)        

submission = trainer.predict_process(test_ld=test_ld, submission=submission)

submission.to_csv('submission.csv', index=False)



submission.head()