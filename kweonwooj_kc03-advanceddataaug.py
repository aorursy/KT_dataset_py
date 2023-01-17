import pandas as pd

import subprocess

import torch

import torch.optim as optim

from torch import nn

import torchvision

from torchvision import transforms



import os

from glob import glob

import numpy as np

from tqdm.notebook import tqdm

#from tqdm import tqdm

import random

import cv2



import sys
# 주요 파라미터 지정

EPOCH = 100

BATCH_SIZE = 16

PATIENCE = 3

cum_patience = 0

best_valid_loss = 999

DATA_PATH = '../input/state-farm-distracted-driver-detection'

WIDTH = 224

HEIGHT = 224

IMGNET_MEAN = [0.485, 0.456, 0.406]

IMGNET_STD = [0.229, 0.224, 0.225]

classes = [f'c{i}' for i in range(10)]



MODE = 'adv_data_aug'

os.mkdir(MODE)



MODEL_NAME = 'resnet50'

assert MODEL_NAME in ['resnet50', 'resnet101', 'vgg19', 'densenet161']
# prepare directory

def prepare_dirs(path):

    cmd = f'rm -rf {path}/*'

    subprocess.call(cmd, shell=True)

    for d in ['train', 'valid']:

        cmd = f'mkdir {path}/{d}'

        subprocess.call(cmd, shell=True)

        for cl in classes:

            cmd = f'mkdir {path}/{d}/{cl}'

            subprocess.call(cmd, shell=True)





def softlink_images(valid_driver, driver_list):

    trn_cnt = 0

    val_cnt = 0

    for i, driver_info in driver_list.iterrows():

        driver = driver_info['subject']

        label = driver_info['classname']

        img_path = driver_info['img']

        # symlink를 통해서 이미지 파일을 지정

        if driver == valid_driver:

            from_ = os.path.abspath(f'{DATA_PATH}/imgs/train/{label}/{img_path}')

            to_ = f'{MODE}/valid/{label}/{img_path}'

            if not os.path.exists(to_):

                os.symlink(from_, to_)

            val_cnt += 1

        else:

            from_ = os.path.abspath(f'{DATA_PATH}/imgs/train/{label}/{img_path}')

            to_ = f'{MODE}/train/{label}/{img_path}'

            if not os.path.exists(to_):

                os.symlink(from_, to_)

            trn_cnt += 1

    return trn_cnt, val_cnt





def load_loaders(path):

    # 이미지 로딩시 전처리 함수 정의

    # Normalize는 ImageNet 데이터에 기학습된 모델을 활용하기 위한 함수

    transform = transforms.Compose([

            transforms.Resize((224, 224)),

            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_dataset = torchvision.datasets.ImageFolder(f'./{path}/train',

                                                     transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,

                                               batch_size=BATCH_SIZE,

                                               shuffle=True,

                                               num_workers=2)



    valid_dataset = torchvision.datasets.ImageFolder(f'./{path}/valid',

                                                     transform=transform)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,

                                               batch_size=BATCH_SIZE,

                                               num_workers=2)

    return train_loader, valid_loader





def get_model(model_name):

    # 2-layer FFN을 만드는 함수

    def get_penultimate_layer(num_ftrs):

        return nn.Sequential(

                nn.Linear(num_ftrs, int(num_ftrs/2)),

                nn.ReLU(),

                nn.Dropout(0.5),

                nn.Linear(int(num_ftrs/2), int(num_ftrs/4)),

                nn.ReLU(),

                nn.Dropout(0.5),

                nn.Linear(int(num_ftrs/4), len(classes)))



    # 모델 이름별, 최종 layer의 이름이 다르기 때문에, 아래와 같이 모델별 최종 layer를 맞춤 제작해야한다

    if model_name == 'resnet50':

        model_conv = torchvision.models.resnet50(pretrained=True)

        num_ftrs = model_conv.fc.in_features

        model_conv.fc = get_penultimate_layer(num_ftrs)

    if model_name == 'resnet101':

        model_conv = torchvision.models.resnet101(pretrained=True)

        num_ftrs = model_conv.fc.in_features

        model_conv.fc = get_penultimate_layer(num_ftrs)

    if model_name == 'vgg19':

        model_conv = torchvision.models.vgg19(pretrained=True)

        num_ftrs = model_conv.classifier[0].in_features  # fc가 아닌 classifier

        model_conv.classifier = get_penultimate_layer(num_ftrs)

    if model_name == 'densenet161':

        model_conv = torchvision.models.densenet161(pretrained=True)

        num_ftrs = model_conv.classifier.in_features  # fc가 아닌 classifier

        model_conv.classifier = get_penultimate_layer(num_ftrs)

    model_conv = model_conv.to(device)

    return model_conv





def train(model_conv, train_loader, optimizer, criterion, trn_cnt):

    running_loss = 0.

    running_acc = 0.



    # 학습 진도 확인을 위한 progress_bar

    pbar = tqdm(total=trn_cnt)

    cnt = 0

    for i, data in enumerate(train_loader, 0):

        # 데이터 로더에서 BATCH_SIZE 만큼 학습 데이터를 로딩

        inputs, labels = data



        # DATA AUGMENTATION

        batch = 0

        for input_, label in zip(inputs, labels):

            # pick random image from class

            imgs = glob(f'{MODE}/train/c{label}/*.jpg')

            rand_img = imgs[random.randint(0, len(imgs) - 1)]



            # read image

            rand_img = cv2.imread(rand_img, cv2.IMREAD_COLOR)

            rand_img = cv2.cvtColor(rand_img, cv2.COLOR_BGR2RGB)

            rand_img = cv2.resize(rand_img, (224, 224), interpolation=cv2.INTER_CUBIC)



            # normalize

            rand_img = rand_img / 255.

            for i in range(3):

                rand_img[:, :, i] = (rand_img[:, :, i] - IMGNET_MEAN[i]) / IMGNET_STD[i]

            rand_img = np.transpose(rand_img, (2, 0, 1))



            # cutmix

            new_img = np.zeros((3, HEIGHT, WIDTH))

            random_cut = random.randint(0, WIDTH - 1)

            new_img[:, :, :random_cut] = input_[:, :, :random_cut]

            new_img[:, :, random_cut:] = rand_img[:, :, random_cut:]



            inputs[batch] = torch.tensor(new_img)

            batch += 1



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

    trn_loss = running_loss / cnt

    trn_acc = running_acc / cnt

    print(f'\n# train loss : {trn_loss} train acc : {trn_acc}')





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



    valid_loss = valid_loss / cnt

    valid_acc = valid_acc / cnt

    print(f'\n# valid loss | valid_loss : {valid_loss} valid_acc : {valid_acc}')

    return valid_loss





def save_best_model(save_path, patience, valid_loss, best_valid_loss):

    # 검증 데이터의 평가 척도 기준으로 최적의 모델 선정

    if valid_loss < best_valid_loss:

        best_valid_loss = valid_loss

        torch.save(model_conv, save_path)

        patience = 0

        print(f'\n# Saving best model.. with valid_loss {valid_loss}')

    patience += 1

    return patience, best_valid_loss





def predict_test(test_save_path, model_save_path, device):

    TEST_BATCH_SIZE = 128



    # 캐글 제출을 위한 test_id 읽어오기

    test_ids = [os.path.basename(fl) for fl in glob(f'{DATA_PATH}/imgs/test/img_*.jpg')]

    test_ids.sort()

    TEST_SIZE = len(test_ids)



    # 테스트 데이터의 전처리 함수 정의

    transform = transforms.Compose([

            transforms.Resize((224, 224)),

            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



    # 테스트 데이터 로딩을 위한 데이터 로더 정의

    test_dataset = torchvision.datasets.ImageFolder(f'{DATA_PATH}/imgs',

                                                    transform=transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,

                                              batch_size=TEST_BATCH_SIZE,

                                              num_workers=2)



    # 저장된 모델 로딩

    model_conv = torch.load(model_save_path)

    model_conv = model_conv.to(device)

    softmax = nn.Softmax(dim=1)



    pbar = tqdm(total=TEST_SIZE)

    pred_cnt = 0

    with open(test_save_path, 'w') as out:

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

                    if pred_cnt >= TEST_SIZE:

                        break



                    test_id = test_ids[i * TEST_BATCH_SIZE + j]

                    prob = ','.join([str(round(val, 3)) for val in prob.cpu().detach().numpy()])

                    out.write(f'{test_id},{prob}\n')

                    pred_cnt += 1



            if pred_cnt >= TEST_SIZE:

                break



            pbar.update(TEST_BATCH_SIZE)

    pbar.close()





def move_test_data_to_train_dir(test_save_path):

    test_results = pd.read_csv(test_save_path)

    test_cnt = 0

    class_cnt = {}

    for i, row in test_results.iterrows():

        img_path = row['img']

        preds = row[classes].values

        pred_class = f'c{np.argmax(preds)}'



        from_ = os.path.abspath(f'imgs/test/{img_path}')

        to_ = f'{MODE}/train/{pred_class}/{img_path}'

        if not os.path.exists(to_):

            os.symlink(from_, to_)



        class_cnt[pred_class] = class_cnt.get(pred_class, 0) + 1

        test_cnt += 1



    print('# Added below images to each class using semi-supervised :')

    for k, v in class_cnt.items():

        print(f'{k} : {v}')

    return test_cnt
# 운전자 25:1 split

driver_list = pd.read_csv(f'{DATA_PATH}/driver_imgs_list.csv')

drivers = np.unique(driver_list['subject'].values)

valid_driver = drivers[0] # just do 1-fold out of 26-folds



# prepare directory for driver-based split

prepare_dirs(MODE)



# split data into trn/val and return cnts

trn_cnt, val_cnt = softlink_images(valid_driver, driver_list)



# Load loaders

trn_loader, val_loader = load_loaders(MODE)



# GPU 사용을 위한 device 변수 정의

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Define model

model_conv = get_model(MODEL_NAME)



# 학습 옵션 : 손실 함 수 및 optimizer 정의

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model_conv.parameters(), lr=0.0001, weight_decay=1e-6, momentum=0.9)

softmax = nn.Softmax(dim=1)



# EPOCH 횟수 만큼 학습 진행

for epoch in range(EPOCH):

    print(f'\n# Epoch : {epoch}..')

    # 1 Epoch 학습

    train(model_conv, trn_loader, optimizer, criterion, trn_cnt)



    # 검증 데이터 기준 평가

    valid_loss = evaluate(model_conv, val_loader, criterion)



    # early_stopping

    model_save_path = f'{MODE}.{MODEL_NAME}.{valid_driver}'

    cum_patience, best_valid_loss = save_best_model(model_save_path, cum_patience, valid_loss, best_valid_loss)

    if cum_patience == PATIENCE:

        break



# make submission

test_save_path = 'submission.csv'

predict_test(test_save_path, model_save_path, device)