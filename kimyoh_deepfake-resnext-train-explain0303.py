import os, sys, random

import numpy as np

import pandas as pd

import cv2

#파이토치임포트

import torch

#파이토치 인공신경망 모델의 재료들을 담고 있는 모듈 임포트

import torch.nn as nn

#위의 nn모듈을 함수화 한 모듈 임포트

import torch.nn.functional as F



from tqdm.notebook import tqdm



%matplotlib inline

import matplotlib.pyplot as plt
#torch.cuda.is_available():현재상태에서 cuda를 사용할 수 있는지여부

#cuda를 사용할 수 있으면  "cuda:0"을 아니면 "cpu" 반환하여 torch.device에 설정한 후 "gpu"라는 변수에 저장 해 놓음

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gpu
image_size = 224

batch_size = 64
crops_dir = "../input/faces-155/"



metadata_df = pd.read_csv("../input/deepfakefaces/metadata.csv")

metadata_df.head()
len(metadata_df)
len(metadata_df[metadata_df.label == "REAL"]), len(metadata_df[metadata_df.label == "FAKE"])
img_path = os.path.join(crops_dir, np.random.choice(os.listdir(crops_dir)))

plt.imshow(cv2.imread(img_path)[..., ::-1])
#정규화를 위해 torchvision.transform에서 정규화 모듈임포트

from torchvision.transforms import Normalize



#역정규화 클래스 선언

class Unnormalize:

    """Converts an image tensor that was previously Normalize'd

    back to an image with pixels in the range [0, 1]."""

    

    #생성자 정의(mean(평균),std(분산)):__init__(생성자):객체가 생성될 때 자동으로 호출되는 메서드

    def __init__(self, mean, std):

        self.mean = mean

        self.std = std



    def __call__(self, tensor):

        #view함수:텐서의 원소개수를 유지하면서 모양을 바꾼다.

        mean = torch.as_tensor(self.mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)

        std = torch.as_tensor(self.std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)

        return torch.clamp(tensor*std + mean, 0., 1.)



#우리가 사용하는 모델에서 요구하는 각 채널의 시퀀스의 분산과 평규니다

mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]

normalize_transform = Normalize(mean, std)

unnormalize_transform = Unnormalize(mean, std)
def random_hflip(img, p=0.5):

    """Random horizontal flip."""

    if random.random() < p:

        return cv2.flip(img, 1)

    else:

        return img
def load_image_and_label(filename, cls, crops_dir, image_size, augment):

    #해당되는 파일의 이미지를 텐서값으로 변환, 그 라벨 값을 가져온다.

    #해당경로에 있는 이미지파일 읽어옴

    img = cv2.imread(os.path.join(crops_dir, filename))

    #openCV컬러를 BGR로 저장하는데 matplotlib등에서는 RGB로 저장하므로, BGR->RGB로 바꾸는 함수

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    

    #확장이 True ,이미지를 수평으로 뒤집는다.

    if augment: 

        img = random_hflip(img)

     #이미지파일을 (224,224)사이즈로 조절한다. 

    img = cv2.resize(img, (image_size, image_size))

    #img파일 파이토치 텐서로 변환하고,차원을(원래이미지파일차원인덱스를 넣음)조절함(괄호안대로 ).이미지는 0~255까지픽셀값이 있으므로 255로 나누어 정규화

    img = torch.tensor(img).permute((2, 0, 1)).float().div(255)

    #앞서 정의 한 평균과 분산을 가지고 이미지 값을 정규화함.

    img = normalize_transform(img)

#cls(라벨)값이 "fake"면 1이고, 아니면 0(real)이다.

    target = 1 if cls == "FAKE" else 0

    return img, target
img, target = load_image_and_label("aabuyfvwrh.jpg", "FAKE", crops_dir, 224, augment=True)

img.shape, target
plt.imshow(unnormalize_transform(img).permute((1, 2, 0)))
from torch.utils.data import Dataset



class VideoDataset(Dataset):

    """Face crops dataset.



    Arguments:

        crops_dir: 이미지자료가 있는 폴더,

        df: 데이터프레임(메타데이터가 있는)

        split: 훈련을 한다면 데이터 확장

        image_size: 사이즈조절한사이즈

        sample_size: evenly samples this many videos from the REAL

            and FAKE subfolders (None = use all videos)

        seed: 랜덤으로 선택하는 샘플링 상태를 저장하는 숫자,

    """

    def __init__(self, crops_dir, df, split, image_size, sample_size=None, seed=None):

        self.crops_dir = crops_dir

        self.split = split

        self.image_size = image_size

        

        if sample_size is not None:#샘플링할 개수가 있다면

            real_df = df[df["label"] == "REAL"]

            fake_df = df[df["label"] == "FAKE"]

            #sample_size와, 진짜와 가짜데이터의 길이중 가장 작은 값을 반환한다.

            sample_size = np.min(np.array([sample_size, len(real_df), len(fake_df)]))

            print("%s: sampling %d from %d real videos" % (split, sample_size, len(real_df)))

            print("%s: sampling %d from %d fake videos" % (split, sample_size, len(fake_df)))

            real_df = real_df.sample(sample_size, random_state=seed)

            fake_df = fake_df.sample(sample_size, random_state=seed)

            self.df = pd.concat([real_df, fake_df])

        else:

            self.df = df



        num_real = len(self.df[self.df["label"] == "REAL"])

        num_fake = len(self.df[self.df["label"] == "FAKE"])

        print("%s dataset has %d real videos, %d fake videos" % (split, num_real, num_fake))



    def __getitem__(self, index):

        row = self.df.iloc[index]

        filename = row["videoname"][:-4] + ".jpg"

        cls = row["label"]

        return load_image_and_label(filename, cls, self.crops_dir, 

                                    self.image_size, self.split == "train")

    def __len__(self):

        return len(self.df)
dataset = VideoDataset(crops_dir, metadata_df, "val", image_size, sample_size=1000, seed=1234)
plt.imshow(unnormalize_transform(dataset[0][0]).permute(1, 2, 0))
del dataset
def make_splits(crops_dir, metadata_df, frac):

    # Make a validation split. Sample a percentage of the real videos, 

    # and also grab the corresponding fake videos.

    real_rows = metadata_df[metadata_df["label"] == "REAL"]

    real_df = real_rows.sample(frac=frac, random_state=666)

    #metadata.csv파일에서 "original"열에서 real_df의 "videoname"이 안에 있으면 가짜로 보고"fake_df"변수에 저장

    fake_df = metadata_df[metadata_df["original"].isin(real_df["videoname"])]

    #진짜와 가짜 합치기

    val_df = pd.concat([real_df, fake_df])



    # The training split is the remaining videos.

    train_df = metadata_df.loc[~metadata_df.index.isin(val_df.index)]



    return train_df, val_df
train_df, val_df = make_splits(crops_dir, metadata_df, frac=0.05)

#assert(가정문:가로안의 값이 오류가나면 aseertatin Error 발생)

#assert()의 조건을 보증한다

#train데이터의 길이와 val데이터의 길이의 앞은 전체 데이터의 길이와 동일하다는 것을 보증

assert(len(train_df) + len(val_df) == len(metadata_df))

#train데이터의 "videoname"안에 val데이터의 "videoname"이 없다는 것을 보증한다.

assert(len(train_df[train_df["videoname"].isin(val_df["videoname"])]) == 0)



del train_df, val_df
#데이터셋으로부터 유의미한 데이터를 뽑아오는 것을 데이터로더라고 함,이 모듈은 임포트해 batch_size지정하면 한 번에 batch_size만큼 불러올 수 있다.

from torch.utils.data import DataLoader



def create_data_loaders(crops_dir, metadata_df, image_size, batch_size, num_workers):

    #트레인데이터와 발리데이션 데이터를 나눔

    train_df, val_df = make_splits(crops_dir, metadata_df, frac=0.05)

    

    #위에서 정의한 VideoDataset()함수를 이용해 train_dataset을 만듦

    train_dataset = VideoDataset(crops_dir, train_df, "train", image_size, sample_size=10000)

    #torch.utils.data에서 제공하는 DataLoader를 사용하여 train_dataset에서 한 번에 batch_size만큼 자료를 가져온다.

    #DataLoader 속성 값: num_workers:데이터프로세싱에 할당하는 cpu코어개수,pin_memory=True이면 메모리에 샘플을 할당하여 데이터 전송속도를 올릴 수 있다.

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 

                              num_workers=num_workers, pin_memory=True)



    val_dataset = VideoDataset(crops_dir, val_df, "val", image_size, sample_size=500, seed=1234)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 

                            num_workers=num_workers, pin_memory=True)



    return train_loader, val_loader
train_loader, val_loader = create_data_loaders(crops_dir, metadata_df, image_size, 

                                               batch_size, num_workers=2)
#train_loader기에서 iter함수로 차례로 읽어들인것을 next함수로 반환한다

X, y = next(iter(train_loader))

plt.imshow(unnormalize_transform(X[0]).permute(1, 2, 0))

print(y[0])
X, y = next(iter(val_loader))

plt.imshow(unnormalize_transform(X[0]).permute(1, 2, 0))

print(y[0])
def evaluate(net, data_loader, device, silent=False):

    net.train(False)



    bce_loss = 0

    total_examples = 0

   #with tqdm(,desc="") as pbar:상태바 나타내는 함수

    with tqdm(total=len(data_loader), desc="Evaluation", leave=False, disable=silent) as pbar:

        for batch_idx, data in enumerate(data_loader):#데이터로더에 있는 값을 인덱스와 함께 펼침

            with torch.no_grad():#평가할때 기울기 자동계산안한다.

                batch_size = data[0].shape[0]

                #자료의 데이터값과 결과값을 해당 메모리에 보냄

                x = data[0].to(device)

                y_true = data[1].to(device).float()

                

#모델 net(x)의 결과값(즉,예측값)을 squeeze()함수 사용하여 차원수를 축소한다.

                y_pred = net(x)

                y_pred = y_pred.squeeze()

               #예측값과 실제값의 오차를 구하여 스칼라(.item())값으로 구한값에 batch_size값을 구한 값을 한 배치마다 더함.

                bce_loss += F.binary_cross_entropy_with_logits(y_pred, y_true).item() * batch_size

            #총자료개수는 한배치가 돌아갈때마다 누적해서 총자료개수가 됨.

            total_examples += batch_size

            #프로그래스바 수정

            pbar.update()

    #오차의 총합을 총 개수로 나누면 오차평균이 나옴.

    bce_loss /= total_examples



    if silent:

        return bce_loss

    else:

        print("BCE: %.4f" % (bce_loss))
def fit(epochs):

    #전역변수 선언

    global history, iteration, epochs_done, lr

    #tqdm 함수와 상태바 설정

    with tqdm(total=len(train_loader), leave=False) as pbar:

        #에포크 수만큼 for문 돌림

        for epoch in range(epochs):

            #상태창을 초기화

            pbar.reset()

            #상태창 출력값 설정

            pbar.set_description("Epoch %d" % (epochs_done + 1))

            #로스값과 자료개수 초기화

            bce_loss = 0

            total_examples = 0

            

            #모델의 훈련을 활성화

            net.train(True)

            #훈련데이터로더의 자료를 인덱스 값과 펼친다

            for batch_idx, data in enumerate(train_loader):

                #batch_size는 데이터의 첫행의 모양의 첫번째

                batch_size = data[0].shape[0]

                #자료값과 실제값을 메모리에 보냄

                x = data[0].to(gpu)

                y_true = data[1].to(gpu).float()

                #경사하강법 기울기(가중치)값 초기화한다.(학습을 시작해야하므로)

                optimizer.zero_grad()

                #예측값을 squeeze()함수 사용하여 차원을 축소한다.

                y_pred = net(x)

                y_pred = y_pred.squeeze()

                #오차값:실제값과 예측값의 오차를 binary_cross_entropy를 사용하여 구한다

                loss = F.binary_cross_entropy_with_logits(y_pred, y_true)

                #역전파:loss의 기울기의 반대방향으로 이동하여 기울기 개선시킨다.

                loss.backward()

                #위의 손실함수가 역전파하는동한 경사하강법으로 기울기 최적화시킨다.

                optimizer.step()

                #batch_bce에 loss의 스칼라 값을 저장한다

                batch_bce = loss.item()

                #for문이 한번 돌때마다 1batch의 loss총합(batch_bce * batch_size)을 bce_loss에 더해 회전이 끝나면 loss의 총합을 구할 수 있다.

                bce_loss += batch_bce * batch_size

                history["train_bce"].append(batch_bce)

                #총자료개수는 한번돌때마다 batch_size를 더해서 총 자료개수 구함

                total_examples += batch_size

                #반복회수 1씩증가

                iteration += 1

                #프로세스바 update됨

                pbar.update()

            #1epoch돌릴때마다 오차의 총합에 총 파일개수를 나누어 평균오차를 구함

            bce_loss /= total_examples

            epochs_done += 1



            print("Epoch: %3d, train BCE: %.4f" % (epochs_done, bce_loss))

            #validation loss는 evaluate함수 사용하여 함께 구함

            val_bce_loss = evaluate(net, val_loader, device=gpu, silent=True)

            history["val_bce"].append(val_bce_loss)

            

            print("              val BCE: %.4f" % (val_bce_loss))

            

            

            



            # TODO: can do LR annealing here

            # TODO: can save checkpoint here

            #학습률 조정

            scheduler.step()

            #에포크마다 모델저장하기

            torch.save(net.state_dict(), "epoch:{} val_bce:{:.4f}.pth".format(epochs_done,val_bce_loss))

            



            print("")
checkpoint = torch.load("../input/externaldata/pretrained-pytorch/resnext50_32x4d-7cdf4587.pth")
import torchvision.models as models

#torchvision.models에 있는 resnet모델에서 ResNet class 상속

class MyResNeXt(models.resnet.ResNet):

    

    def __init__(self, training=True):

        #ResNet 생성자 끌어옴.

        super(MyResNeXt, self).__init__(block=models.resnet.Bottleneck,

                                        layers=[3, 4, 6, 3], 

                                        groups=32, 

                                        width_per_group=4)



        self.load_state_dict(checkpoint)



        # Override the existing FC layer with a new one.

        self.fc = nn.Linear(2048, 1)
net = MyResNeXt().to(gpu)
del checkpoint
out = net(torch.zeros((10, 3, image_size, image_size)).to(gpu))

out.shape
#모델과 파라미터 이름을 입력하면,입력한 파라미터이전 파라미터는 학습 안함(전이학습)

def freeze_until(net, param_name):

    found_name = False

    

    for name, params in net.named_parameters():

        

        if name == param_name:

           

            found_name = True

    

        params.requires_grad = found_name
[k for k,v in net.named_parameters()]
freeze_until(net, "layer4.0.conv1.weight")
[k for k,v in net.named_parameters() if v.requires_grad]
evaluate(net, val_loader, device=gpu)
lr = 0.1



wd=0.000



history = { "train_bce": [], "val_bce": [] }

iteration = 0

epochs_done = 0

#경사하강법 adam,weight_decay=L2규제

optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

#학습률 조절기:stept_size=5(5회전할때마다),gamma=0.1 학습률에 0.1을 곱해준다.

scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)
fit(10)
freeze_until(net, "layer3.0.conv1.weight")
fit(5)
#위의 스케줄러 설정으로 대체한다

# def set_lr(optimizer, lr):

#     for param_group in optimizer.param_groups:

#         param_group["lr"] = lr
#위의 스케줄러 설정으로 대체한다.

# lr /= 10

# set_lr(optimizer, lr)
plt.plot(history["train_bce"])
plt.plot(history["val_bce"])
evaluate(net, val_loader, device=gpu, silent=True)
# torch.save(net.state_dict(), "checkpoint.pth")