# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
train
y = train["label"]
y[:10]
train2 = train.drop("label",axis=1)
train2

X = train2.iloc[0].values
X

X2 = X.reshape(28,28)
import matplotlib.pyplot as plt
plt.imshow(X2)

import cv2
cv2.imwrite("tmp.bmp",X2)
tmp = cv2.imread("tmp.bmp")
tmp.shape
plt.imshow(tmp)
X = train2.iloc[0].values
X2 = X.reshape(28,28)
cv2.imwrite("tmp.bmp",X2)
tmp = cv2.imread("tmp.bmp")
plt.imshow(tmp)
for i in range(10):
    X = train2.iloc[i].values
    X2 = X.reshape(28,28)
    cv2.imwrite("tmp.bmp",X2)
    tmp = cv2.imread("tmp.bmp")
    plt.figure()
    plt.imshow(tmp)
def makeimage(num):
    for i in range(num):
        X = train2.iloc[i].values
        X2 = X.reshape(28,28)
        cv2.imwrite("tmp.bmp",X2)
        tmp = cv2.imread("tmp.bmp")
        plt.figure()
        plt.imshow(tmp)
makeimage(15)
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision.models import resnet18
from albumentations import Normalize, Compose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import os
import glob
import multiprocessing as mp



if torch.cuda.is_available():
    device = 'cuda:0'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
print(f'Running on device: {device}')

preprocess = Compose([
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1)
])

# resnextなどのpre-trainモデルは全て、同じ方法で正規化された入力画像を使用しなければならない。それの変換をこの関数で行う。値はdefault。
# Composeは今回あまり、意味をなさない
# https://betashort-lab.com/%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B5%E3%82%A4%E3%82%A8%E3%83%B3%E3%82%B9/albumentations%E3%81%AE%E3%81%BE%E3%81%A8%E3%82%81/ に詳細は書いてある
preprocess2 = Compose([
    Normalize(mean=[0.5,], std=[0.5, ])
])

# 画像をどれだけ小さくするかの処理
ROWS = 32
COLS = 32
class GLDataset(Dataset):
    
    def __init__(self,df,labels,preprocess=None):
        self.df = df
        self.labels = labels
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        
        # ここからdatasetに食わせる前の前処理の記述。
        
        
        
        label = self.labels[idx]
        
        #img_pass = self.img_pass[idx]
        
        X = self.df.iloc[idx].values
        X2 = X.reshape(28,28)
        cv2.imwrite("tmp.bmp",X2)
        land = cv2.imread("tmp.bmp") 
        
        land = cv2.resize(land,(ROWS,COLS),interpolation = cv2.INTER_CUBIC)
       
        land = cv2.cvtColor(land,cv2.COLOR_BGR2RGB) # augmentを使うときにBGRからRGBにする必要があるのかもしれない。
        
        if self.preprocess is not None: # ここで、前処理を入れてnormalizationしている。
                augmented = self.preprocess(image=land) # preprocessのimageをfaceで読む
                land = augmented['image'] # https://betashort-lab.com/%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B5%E3%82%A4%E3%82%A8%E3%83%B3%E3%82%B9/albumentations%E3%81%AE%E3%81%BE%E3%81%A8%E3%82%81/　に書いてある
                
        return {'landmarks': land.transpose(2, 0, 1), 'label': np.array(label, dtype=int)}  # pytorchはchannnl, x, yの形。これは辞書型で返している。(扱いやすいというだけかも。)
        
        
        
        
        
        
        
        
train2
traindf = train2.iloc[:30000,:]
valdf = train2.iloc[30000:,:]

trainlabel = y[:30000]
vallabel = y[30000:]
# instance化
train_dataset = GLDataset(
    df=traindf,
    labels=trainlabel.to_numpy(),
    preprocess=preprocess
)

val_dataset = GLDataset(
    df=valdf,
    labels=vallabel.to_numpy(),
    preprocess=preprocess
)

BATCH_SIZE = 1028

#NUM_WORKERS = mp.cpu_count()
NUM_WORKERS = 0 # ここを0にしないと動かない。cpuの仕様個数。←実は動くことが判明。classの中身次第！
## DataLoaderはimport torch.utils.data.Datasetでimport済みのもの
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, #https://schemer1341.hatenablog.com/entry/2019/01/06/024605 を参考. idがわからなくなる
    num_workers=NUM_WORKERS
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

encoder = resnet18(pretrained=True) #  pretrained = Trueはimagenetからpre-trainモデルを使う
class LandmarkClassifier(nn.Module): # nn.Moduleが入っているのが、モデルの定義っぽい
    
    def __init__(self, encoder, in_channels=3, num_classes=10): 
        
        super(LandmarkClassifier, self).__init__() # nn.Moduleの__init__を継承。https://blog.codecamp.jp/python-class-2
        
        self.encoder = encoder
        
        # Modify input layer. # 入口のチャンネル数を合わせる。defaultのResnetのclassは64 channelになっているので、それをin_channelsにする。
        # ここの記述がencoderごとに代わるので、efficientnet使うときは変えなければいけない
        
        self.encoder.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Modify output layer.# 出口の個数も合わせる。defaultのResnetの出口は、1000になっているので、num_classesに変更
        # ここの記述がencoderごとに代わるので、efficientnet使うときは変えなければいけない
        
        self.encoder.fc = nn.Linear(512 * 1, num_classes)

    def forward(self, x): # 呼び出されたときに、xの引数があると、sigmoidで返す。ex) 後ほどのclassifier(sample_batched["landmark"]) みたいなところ。
        
        # sigmoidで返すときは以下の感じ　https://aidiary.hatenablog.com/entry/20180203/1517629555
        # return torch.sigmoid(self.encoder(x))
        
        # 多値問題でも、softmaxはここでは使わない。nn.CrossEntropyに既に入っているため
        return self.encoder(x)
    
    
    
    
    
    ### 以下、どこのパラメーターを最適化するか。簡易テスト用は真ん中。フルオプションは下。ここはご参考。###
    
    def freeze_all_layers(self):# 中間層のパラメーター(重みづけ)を全部変えない。
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_middle_layers(self):
        self.freeze_all_layers()
        
        for param in self.encoder.conv1.parameters():# 最初のパラメーター(重みづけ)を変える。
            param.requires_grad = True
            
        for param in self.encoder.fc.parameters():# 最後の(重みづけ)を変える。
            param.requires_grad = True

    def unfreeze_all_layers(self):# 中間層のパラメーター(重みづけ)を全部変える。
        for param in self.encoder.parameters():
            param.requires_grad = True

classifier = LandmarkClassifier(encoder=encoder, in_channels=3, num_classes=10) # classifierはDeepfakeClassifierのインスタンス化

classifier = classifier.to(device) # ここがKerasとは違うところ。GPUに送りますよーという意味。今回はcpuなので、pass.

classifier.train() # ここがKerasとは違うところ。 訓練モードの場合 classifier.train() , 推論の場合 classifier.eval() と書く。Normalizeのプロセスなどが違うらしい。
# 今回は全部最適化
classifier.unfreeze_all_layers()
criterion = nn.CrossEntropyLoss()
# 多値分類はcrossentropy
#optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()),lr=1e-5)
optimizer = optim.Adam(classifier.parameters(),lr=0.01)
#optimizer = optim.SGD(classifier.parameters(),lr=1e-5)

# conv層(model.features)のパラメータを固定し、全結合層(model.classifier)のみをfinetuneする設定です。
# 全層のうち、requires_gradがTrueの層のみをfinetuneするという意味です。
# lr : 学習率
def trainmodel(train_dataloader):
    classifier.train()
    
    for a in train_dataloader:
        
        input1 = a["landmarks"].to(device)
        
        y_pred = classifier(input1) # onenoteでいうoutput = model(train_x)
        label = a["label"].to(device)

        loss = criterion(y_pred, label)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad() # おきまり。初期化。
        loss.backward() # 後方にバック
        optimizer.step() # 重みづけの最適化
        
    return(loss.item())
    
    
def valmodel(val_dataloader):
    
    classifier.eval()
    
    for a in val_dataloader:
        
        input1 = a["landmarks"].to(device)

        y_pred = classifier(input1) # onenoteでいうoutput = model(train_x)
        label = a["label"].to(device)

        loss = criterion(y_pred, label)
        
    return(loss.item())
    
    
epochs = 3
savename = "resnet18.pth"
def savemodel(bestloss,valloss):
    
    #######modelをsave#######
    
    if bestloss is None:
        bestloss = valloss[-1]
        state = {
                'state_dict': classifier.state_dict(),
                'optimizer_dict': optimizer.state_dict(),
                "bestloss":bestloss,
            }

        torch.save(state, savename)
        
        print("save the first model")
    
    elif valloss[-1] < bestloss:
        
        bestloss = valloss[-1]
        state = {
                'state_dict': classifier.state_dict(),
                'optimizer_dict': optimizer.state_dict(),
                "bestloss":bestloss,
            }

        torch.save(state, savename)
        
        print("found a better point")
    
    else:
        pass
    
    return bestloss

trainloss = []
valloss = []

bestloss = None

for epoch in range(epochs):
    
    trainloss.append(trainmodel(train_dataloader))
    valloss.append(valmodel(val_dataloader))
    
    print(str(epoch) + "_end")
    
    bestloss= savemodel(bestloss,valloss)
    
    print(bestloss)
    
    
    

x = np.arange(epochs)
plt.scatter(x,trainloss)
plt.scatter(x,valloss)
testdf = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
testdf
class GLDataset_inf(Dataset):
    
    def __init__(self,df,testid,preprocess=None):
        self.df = df
        self.testid = testid
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        
        # ここからdatasetに食わせる前の前処理の記述。
        
        
        
        testid = self.testid[idx]
        
        #img_pass = self.img_pass[idx]
        
        X = self.df.iloc[idx].values
        X2 = X.reshape(28,28)
        cv2.imwrite("tmp.bmp",X2)
        land = cv2.imread("tmp.bmp") 
        
        land = cv2.resize(land,(ROWS,COLS),interpolation = cv2.INTER_CUBIC)
       
        land = cv2.cvtColor(land,cv2.COLOR_BGR2RGB) # augmentを使うときにBGRからRGBにする必要があるのかもしれない。
        
        if self.preprocess is not None: # ここで、前処理を入れてnormalizationしている。
                augmented = self.preprocess(image=land) # preprocessのimageをfaceで読む
                land = augmented['image'] # https://betashort-lab.com/%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B5%E3%82%A4%E3%82%A8%E3%83%B3%E3%82%B9/albumentations%E3%81%AE%E3%81%BE%E3%81%A8%E3%82%81/　に書いてある
                
        return {'landmarks': land.transpose(2, 0, 1), 'testid': testid}  # pytorchはchannnl, x, yの形。これは辞書型で返している。(扱いやすいというだけかも。)
        
        
        
        
        
        
        
        
testid2 = testdf.index.values + 1
# instance化
test_dataset = GLDataset_inf(
    df=testdf,
    testid=testid2,
    preprocess=preprocess
)

## DataLoaderはimport torch.utils.data.Datasetでimport済みのもの
test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, #https://schemer1341.hatenablog.com/entry/2019/01/06/024605 を参考. idがわからなくなる
    num_workers=NUM_WORKERS
)

classifier.eval()
presub = []

    
for a in test_dataloader:

    input1 = a["landmarks"].to(device)

    y_pred = classifier(input1) # onenoteでいうoutput = model(train_x)
    
    soft = F.softmax(y_pred).cpu().detach().numpy()
    
    y_pred = y_pred.cpu().detach().numpy()
    
    for b in range(len(y_pred)):
        
        predid = np.argmax(soft[b])
        conf = np.max(soft[b])
        
        testid = a["testid"][b]
        
        presub.append([testid,predid,conf])
    
    
    

    
submission = pd.DataFrame(presub,columns = ["ImageId","labels","conf"])
submission
sample = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
sample
sample["Label"]=submission["labels"]
sample
sample.to_csv("submission.csv",index=False)