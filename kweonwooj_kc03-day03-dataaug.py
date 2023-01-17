# 학습 데이터 위치 확인

! ls ../input/state-farm-distracted-driver-detection/
# 학습 데이터 위치 확인

! ls ../input/state-farm-distracted-driver-detection/imgs
# 랜덤하게 배정된 GPU 확인

! nvidia-smi
# 필요한 라이브러리 설치

! pip install torch

! pip install torchvision
# 필요한 라이브러리 import

import pandas as pd

import subprocess

import torch

import torch.optim as optim

from torch import nn

from torch.utils.data import Dataset

from torch.utils.data.sampler import SubsetRandomSampler

import torchvision

from torchvision import transforms



import os

import random

from glob import glob

import cv2

import numpy as np

from tqdm.notebook import tqdm
# 주요 파라미터 지정

EPOCH = 100

BATCH_SIZE = 16

PATIENCE = 5

DATA_PATH = '../input/state-farm-distracted-driver-detection'

MODEL_NAME = './model.baseline.driver_split.data_aug'
# 이미지 로딩시 전처리 함수 정의

# 다양한 데이터 어그멘테이션 함수를 추가로 정의!!

# Normalize는 ImageNet 데이터에 기학습된 모델을 활용하기 위한 함수

transform = transforms.Compose([

        transforms.RandomAffine(30, translate=(0.3, 0.3)),  # 랜덤하게 수평/수직 이동 

        transforms.RandomPerspective(p=0.1),  # 시점 이동

        transforms.RandomRotation(degrees=30),  # 회전

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# 학습 데이터 중, 운전자 기준으로 20%를 검증 데이터로 사용

classes = [f'c{i}' for i in range(10)]

seed = 2020

validation_split = 0.2



# 운전자 정보 읽어오기

driver_list = pd.read_csv(f'{DATA_PATH}/driver_imgs_list.csv')

drivers = np.unique(driver_list['subject'].values)



# 운전자 기준 split

split = int(np.floor(validation_split * len(drivers)))

np.random.seed(seed)

trn_idx, val_idx = drivers[split:], drivers[:split]
# 운전자 기준으로 학습/검증 데이터 분리

split_dir = 'driver_split'

if not os.path.exists(split_dir):

    cmd = f'mkdir {split_dir}'

    subprocess.call(cmd, shell=True)

    for d in ['train', 'valid']:

        cmd = f'mkdir {split_dir}/{d}'

        subprocess.call(cmd, shell=True)

        for cl in classes:

            cmd = f'mkdir {split_dir}/{d}/{cl}'

            subprocess.call(cmd, shell=True)



trn_cnt = 0

val_cnt = 0

for i, driver_info in driver_list.iterrows():

    driver = driver_info['subject']

    label = driver_info['classname']

    img_path = driver_info['img']

    # symlink를 통해서 이미지 파일을 지정

    if driver in trn_idx:

        if not os.path.exists(f'{split_dir}/train/{label}/{img_path}'):

            os.symlink(os.path.abspath(f'{DATA_PATH}/imgs/train/{label}/{img_path}'), f'{split_dir}/train/{label}/{img_path}')

        trn_cnt += 1

    else:

        if not os.path.exists(f'{split_dir}/valid/{label}/{img_path}'):

            os.symlink(os.path.abspath(f'{DATA_PATH}/imgs/train/{label}/{img_path}'), f'{split_dir}/valid/{label}/{img_path}')

        val_cnt += 1
# 학습 데이터 로더 정의

train_dataset = torchvision.datasets.ImageFolder(f'./{split_dir}/train',

                                                 transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset,

                                           batch_size=BATCH_SIZE,

                                           shuffle=True,

                                           num_workers=2)



# 검증 데이터의 이미지 전처리는 Resize + Normalize 만 필요

valid_transform = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

valid_dataset = torchvision.datasets.ImageFolder(f'./{split_dir}/valid',

                                                 transform=valid_transform)

valid_loader = torch.utils.data.DataLoader(valid_dataset,

                                           batch_size=BATCH_SIZE,

                                           num_workers=2)
# GPU 사용을 위한 device 변수 정의

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
# CNN 모델 로딩

# ImageNet에서 기학습된 Resnet50 모델과 파라미터를 그대로 사용하기

model_conv = torchvision.models.resnet50(pretrained=True)



# 최종 layer를 우리 문제에 알맞게 재구성

num_ftrs = model_conv.fc.in_features

model_conv.fc = nn.Sequential(

        nn.Linear(num_ftrs, num_ftrs),

        nn.ReLU(),

        nn.Dropout(0.5),

        nn.Linear(num_ftrs, num_ftrs),

        nn.ReLU(),

        nn.Dropout(0.5),

        nn.Linear(num_ftrs, len(classes)))

print(f'# model : {model_conv}')

model_conv = model_conv.to(device)
# 학습 옵션 : 손실 함 수 및 optimizer 정의

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model_conv.parameters(), lr=0.0001, weight_decay=1e-6, momentum=0.9)

softmax = nn.Softmax(dim=1)

best_valid_score = 999

patience = 0
def train(model_conv, train_loader, optimizer, criterion, trn_cnt):

    running_loss = 0.

    running_acc = 0.



    # 학습 진도 확인을 위한 progress_bar

    pbar = tqdm(total=trn_cnt)

    cnt = 0

    for i, data in enumerate(train_loader, 0):

        # 데이터 로더에서 BATCH_SIZE 만큼 학습 데이터를 로딩

        inputs, labels = data

        # GPU로 데이터 이동

        inputs = inputs.to(device)

        labels = labels.to(device)



        # 학습을 위한 준비

        optimizer.zero_grad()

        model_conv.train()

        outputs = model_conv(inputs)

        probs = softmax(outputs)



        # 정확도 계산

        _, preds = probs.max(axis=1)

        running_acc += sum(labels == preds) / (1. * BATCH_SIZE)



        # 손실 함수 계산

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()  # Gradient Descent HERE!



        running_loss += loss.item()

        cnt += 1



        pbar.update(BATCH_SIZE)

    pbar.close()

    return running_loss / cnt, running_acc / cnt
def evaluate(model_conv, valid_loader, criterion):

    # 1 Epoch마다 검증 데이터에 대하여 평가

    with torch.no_grad():

        model_conv.eval()

        valid_loss = 0.0

        valid_acc = 0.0

        cnt = 0

        pbar = tqdm(total=val_cnt)

        for data in valid_loader:

            inputs, labels = data

            inputs = inputs.to(device)

            labels = labels.to(device)



            outputs = model_conv(inputs)

            probs = softmax(outputs)



            _, preds = probs.max(axis=1)

            valid_acc += sum(labels == preds) / (1. * BATCH_SIZE)



            labels.require_grad = False

            loss = criterion(outputs, labels)

            valid_loss += loss

            cnt += 1

            pbar.update(BATCH_SIZE)

        pbar.close()

    return valid_loss / cnt, valid_acc / cnt
# EPOCH 횟수 만큼 학습 진행

for epoch in range(EPOCH):

    print(f'# Epoch : {epoch}..')

    # 1 Epoch 학습

    trn_loss, trn_acc = train(model_conv, train_loader, optimizer, criterion, trn_cnt)

    print(f'# train loss : {trn_loss} train acc : {trn_acc}')

    

    # 검증 데이터 기준 평가

    valid_loss, valid_acc = evaluate(model_conv, valid_loader, criterion)

    print(f'# valid loss | valid_loss : {valid_loss} valid_acc : {valid_acc}')





    # 검증 데이터의 평가 척도 기준으로 최적의 모델 선정

    if valid_loss < best_valid_score:

        best_valid_score = valid_loss

        print(f'# Saving best model.. epoch {epoch} | valid_loss {valid_loss}')

        torch.save(model_conv, MODEL_NAME)

        patience = 0

    patience += 1



    # early_stopping

    if patience == PATIENCE:

        break



print('Finished Training')
from glob import glob



TEST_SIZE = 79726

BATCH_SIZE = 128



# 캐글 제출을 위한 test_id 읽어오기

test_ids = [os.path.basename(fl) for fl in glob(f'{DATA_PATH}/imgs/test/img_*.jpg')]

test_ids.sort()



# 테스트 데이터의 전처리 함수 정의

transform = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



# 테스트 데이터 로딩을 위한 데이터 로더 정의

test_dataset = torchvision.datasets.ImageFolder(f'{DATA_PATH}/imgs',

                                                transform=transform)

test_loader = torch.utils.data.DataLoader(test_dataset,

                                          batch_size=BATCH_SIZE,

                                          num_workers=2)



# GPU 사용

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)



# 저장된 모델 로딩

model_conv = torch.load(MODEL_NAME)

print(f'# model : {model_conv}')

model_conv = model_conv.to(device)

softmax = nn.Softmax(dim=1)



pbar = tqdm(total=TEST_SIZE)

end_flag = False

with open('submission.csv', 'w') as out:

    # write header

    out.write('img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n')



    for i, data in enumerate(test_loader, 0):

        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data

        inputs = inputs.to(device)

        labels = labels.to(device)



        with torch.no_grad():

            model_conv.eval()

            outputs = model_conv(inputs)

            probs = softmax(outputs)



            # 클래스별 확률 예측값 저장하기

            for j, prob in enumerate(probs):

                if BATCH_SIZE * i + j >= TEST_SIZE:

                    end_flag = True

                    break

                    

                test_id = test_ids[i * BATCH_SIZE + j]

                prob = ','.join([str(round(val, 3)) for val in prob.cpu().detach().numpy()])

                out.write(f'{test_id},{prob}\n')



        pbar.update(BATCH_SIZE)



        if end_flag:

            break

pbar.close()



print('Finished Eval')