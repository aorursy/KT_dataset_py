DEBUG = False
#before import process

import sys

sys.path.append('../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master')



#imports

import os, warnings, random, time

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import cv2



#評価指標(ROCスコア)を算出してくれるモジュール

from sklearn.metrics import accuracy_score, roc_auc_score



#trainingデータ分割を工夫するモジュール

from sklearn.model_selection import StratifiedKFold



#プログレスバーを表示してくれる

from tqdm import tqdm_notebook as tqdm



#pytorchのライブラリ

import torch

from torch import nn, optim

from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.data import Dataset, DataLoader



#Pytorch用EfficientNetを読み込むためのモジュール

from efficientnet_pytorch import model as enet



#Data Augmentation用ライブラリ

import albumentations as A



#モデルの保存

import pickle



#描画設定

%matplotlib inline



#警告文を全て無視

warnings.filterwarnings('ignore')
SEED = 32 #69

def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True



seed_everything(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
#params

enet_type = 'efficientnet-b0' #b0~b8まであります。パラメータ数が大きくなります。

model_name = 'v1'

n_epochs = 2 if DEBUG else 5

cosine_t = 5

n_fold = 5



batch_size = 64

image_size = 224



num_workers = 2 #サーバ用gpu特有の概念です。並行処理をいくつ行うかを指定します。



init_lr = 1e-3

TTA = 1



Progress_Bar = True
train_df = pd.read_csv('../input/melanoma-merged-external-data-512x512-jpeg/folds_13062020.csv')

test_df = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

train_images = '../input/melanoma-merged-external-data-512x512-jpeg/512x512-dataset-melanoma/512x512-dataset-melanoma'

test_images = '../input/melanoma-merged-external-data-512x512-jpeg/512x512-test/512x512-test'



if DEBUG:

    train_df = train_df[:100]

    test_df = test_df[:50]

else:

    train_df = train_df[:10000]

    test_df = test_df[:]

# One-hot encoding of anatom_site_general_challenge feature

concat = pd.concat([train_df['anatom_site_general_challenge'], test_df['anatom_site_general_challenge']], ignore_index=True)

dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')

train_df = pd.concat([train_df, dummies.iloc[:train_df.shape[0]]], axis=1)

test_df = pd.concat([test_df, dummies.iloc[train_df.shape[0]:].reset_index(drop=True)], axis=1)



# Sex features

train_df['sex'] = train_df['sex'].map({'male': 1, 'female': 0})

test_df['sex'] = test_df['sex'].map({'male': 1, 'female': 0})

train_df['sex'] = train_df['sex'].fillna(-1)

test_df['sex'] = test_df['sex'].fillna(-1)



# Age features

train_df['age_approx'] /= train_df['age_approx'].max()

test_df['age_approx'] /= test_df['age_approx'].max()

train_df['age_approx'] = train_df['age_approx'].fillna(0)

test_df['age_approx'] = test_df['age_approx'].fillna(0)



train_df['patient_id'] = train_df['patient_id'].fillna(0)
train_df.head()
pretrained_model = {

        'efficientnet-b0': '../input/efficientnet-pytorch/efficientnet-b0-08094119.pth',

        'efficientnet-b1': '../input/efficientnet-pytorch/efficientnet-b1-dbc7070a.pth',

        'efficientnet-b2': '../input/efficientnet-pytorch/efficientnet-b2-27687264.pth',

        'efficientnet-b3': '../input/efficientnet-pytorch/efficientnet-b3-c8376fa2.pth',

        'efficientnet-b4': '../input/efficientnet-pytorch/efficientnet-b4-e116e8b3.pth',

        'efficientnet-b5': '../input/efficientnet-pytorch/efficientnet-b5-586e6cc6.pth',

        

    }

model_save_path = False





class enetv2(nn.Module):

    def __init__(self, backbone, out_dim=1):

        super(enetv2, self).__init__()

        self.enet = enet.EfficientNet.from_name(backbone)

        self.enet.load_state_dict(torch.load(pretrained_model[backbone]))



        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)

        self.enet._fc = nn.Identity()

        self.sigmoid = nn.Sigmoid()



    def extract(self, x):

        return self.enet(x)



    def forward(self, x):

        x = self.extract(x)

        x = self.myfc(x)

        #x = self.sigmoid(x)

        return x
transforms_train = A.Compose([

    A.Transpose(p=0.5),

    A.VerticalFlip(p=0.5),

    A.HorizontalFlip(p=0.5),

    A.Resize(image_size, image_size), 

    #A.Normalize()

])



transforms_val = A.Compose([

    A.Resize(image_size, image_size),

    #A.Normalize()

])
class MelanomaDataset(Dataset):

    def __init__(self, df: pd.DataFrame, imfolder: str, train: bool = True, transforms = None):



        self.df = df

        self.imfolder = imfolder

        self.transforms = transforms

        self.train = train

        

    def __getitem__(self, index):

        if self.train:

            im_path = os.path.join(self.imfolder, self.df.iloc[index]['image_id'] + '.jpg')

        else:

            im_path = os.path.join(self.imfolder, self.df.iloc[index]['image_name'] + '.jpg')



        x = cv2.imread(im_path)



        if self.transforms:

            x = self.transforms(image = x) #albumentationsに画像を投げます

            x = x['image'].astype(np.float32) #帰ってきたデータから画像データを取り出します(SSDのような場合には矩形領域データも合わせて帰ってきたりするため、このような仕様になっています)

            

        x = x.transpose(2, 0, 1) #channel first

        

        if self.train:

            y = self.df.iloc[index]['target']

            return x, y

        else:

            return x

    

    def __len__(self):

        return len(self.df)
dataset_show = MelanomaDataset(train_df, train_images, train=True,transforms = transforms_train)

from pylab import rcParams

rcParams['figure.figsize'] = 20,10

for i in range(2):

    f, axarr = plt.subplots(1,5)

    for p in range(5):

        idx = np.random.randint(0, len(dataset_show))

        img, label = dataset_show[idx]

        img = np.asarray(img)

        img = img.transpose(1,2,0)

        img = img.astype(np.uint8)

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #rgb→bgr

        axarr[p].imshow(img) 

        axarr[p].set_title(str(label))
def train(model, iterator, optimizer, criterion, device):

    

    epoch_loss = 0

    model.train()

    

    #プログレスバーを表示するか否か

    bar = tqdm(iterator) if Progress_Bar else iterator

    

    for (x, y) in bar:

        x = torch.tensor(x, device=device, dtype=torch.float32)

        y = torch.tensor(y, device=device, dtype=torch.float32)

        

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y.unsqueeze(1)) #ここで、yにy.unsqeeze()次元拡張を挟む必要がある？(BCEだったら、だと思われる)

        loss.backward()

        optimizer.step()

        loss_np = loss.detach().cpu().numpy()

        epoch_loss += loss_np

        

        if Progress_Bar:

            bar.set_description('Training loss: %.5f' % (loss_np))

        

    return epoch_loss/len(iterator)
def evaluate(model, iterator, criterion, device):

    

    epoch_loss = 0

    preds = np.array([])

    targets = np.array([])

    model.eval()

    

    bar = tqdm(iterator) if Progress_Bar else iterator

    

    with torch.no_grad(): #validation時には学習を行いません

        

        for (x, y) in bar:

        

            x = torch.tensor(x, device=device, dtype=torch.float32)

            y = torch.tensor(y, device=device, dtype=torch.float32)

            

            y_pred = model(x)

            loss = criterion(y_pred, y.unsqueeze(1))

            loss_np = loss.detach().cpu().numpy()

            epoch_loss += loss_np

            y_pred = torch.sigmoid(y_pred)

            preds = np.append(preds, y_pred.detach().cpu().numpy())

            targets = np.append(targets, y.detach().cpu().numpy())

            

            if Progress_Bar:

                bar.set_description('Validation loss: %.5f' % (loss_np))

    

    val_acc = accuracy_score(targets, np.round(preds))

    

    try:

       val_roc = roc_auc_score(targets, preds)

    except ValueError:

       val_roc = -1

    

            

    return epoch_loss/len(iterator), val_acc,val_roc
def fit_model(model, model_name, train_iterator, valid_iterator, optimizer, loss_criterion, device, epochs):

    """ Fits a dataset to model"""

    best_valid_score = float('inf')

    

    train_losses = []

    valid_losses = []

    valid_roc_scores = []

    

    for epoch in range(epochs):

        scheduler.step(epoch)

        start_time = time.time()

    

        train_loss = train(model, train_iterator, optimizer, loss_criterion, device)

        valid_loss, valid_acc_score, valid_roc_score = evaluate(model, valid_iterator, loss_criterion, device)

        

        train_losses.append(train_loss)

        valid_losses.append(valid_loss)

        valid_roc_scores.append(valid_roc_score)



        if valid_roc_score < best_valid_score:

            best_valid_score = valid_roc_score

            if model_save_path:

                torch.save(model.state_dict(), os.path.join(model_save_path,f'{model_name}.pt'))

            else:

                torch.save(model.state_dict(), f'{model_name}_best.pt')

        

        #schedulerの処理 cosineannealingは別

        #if scheduler != None:

        #    scheduler.step(valid_loss)

        end_time = time.time()



        epoch_mins, epoch_secs = (end_time-start_time)//60,round((end_time-start_time)%60)

    

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')

        print(f'Train Loss: {train_loss:.3f}')

        print(f'Val. Loss: {valid_loss:.3f} | Val. ACC Score: {valid_acc_score:.3f} | Val. Metric Score: {valid_roc_score:.4f}')

        print(f'lr:{optimizer.param_groups[0]["lr"]:.7f}')

        

        torch.save(model.state_dict(), f'{model_name}_final.pt')

        #pickle

        with open(f'{model_name}_final.pickle', mode = "wb") as fp:

            pickle.dump(model,fp)

        

    return train_losses, valid_losses, valid_roc_scores
tr_loss=[]

val_loss=[]

val_roc=[]

models = []

for fold in range(2):

    print(f"Fitting on Fold {fold+1}")

    #Make Train and Valid DataFrame from fold

    train_df_fold = train_df[train_df['fold'] != fold].reset_index(drop=True)

    valid_df_fold = train_df[train_df['fold'] == fold].reset_index(drop=True)

    

    #Build and load Dataset

    train_data = MelanomaDataset(train_df_fold, train_images, train=True, transforms = transforms_train) 

    valid_data = MelanomaDataset(valid_df_fold, train_images, train=True, transforms = transforms_val) 

    

    train_iterator = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    valid_iterator = DataLoader(valid_data, shuffle=False, batch_size=16, num_workers=num_workers)

    

    #モデルの呼び出し(設計図からインスタンスへ)

    model = enetv2(enet_type, out_dim=1).to(device) #to(device)：gpuで計算を行うことを宣言

    

    #損失関数の定義(BCEはBinary Cross Entropyの略で、通常のCross Entropy Lossをちょっと工夫したものです)

    loss_criterion = nn.BCEWithLogitsLoss()

    

    #最適化手法の定義(現在ではAdam1強です)

    opt= optim.Adam(model.parameters(), lr=init_lr)

    

    #学習率を徐々に下げていくためのスケジューラーを定義します。

    scheduler = CosineAnnealingLR(opt, n_epochs)

    

    name = model_name + "_" + enet_type + "_f" + str(fold)

    

    #全ての情報をfit_modelに入れて、学習を開始します

    temp_tr_loss, temp_val_loss, temp_val_roc = fit_model(model, name, train_iterator, valid_iterator, opt, loss_criterion, device, epochs=n_epochs)

    

    #lossと評価指標に対するスコアを記録します

    tr_loss.append(temp_tr_loss)

    val_loss.append(temp_val_loss)

    val_roc.append(temp_val_roc)

    

    #foldごとにモデルを定義する為、学習し終わったモデルはリストに保持しておきます

    models.append(model)
for i in range(len(tr_loss)):

    fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(20,5))

    ax[0].plot(tr_loss[i])

    ax[0].set_title('Training and Validation Loss')

    ax[0].plot(val_loss[i])

    ax[0].set_xlabel('Epoch')



    ax[1].plot(val_roc[i])

    ax[1].set_title('Val ROC Score')

    ax[1].set_xlabel('Epoch')





    ax[0].legend();

    ax[1].legend();
test = MelanomaDataset(df=test_df,

                       imfolder=test_images, 

                       train=False,

                       transforms=transforms_train)
#(model数*TTA数)回すので注意

def get_predictions(model, iterator, device):

    

    preds = np.array([0.]*len(test_df))

    model.eval()

    bar = tqdm(iterator) if Progress_Bar else iterator

    

    with torch.no_grad():

        for tta in range(TTA):

            res = np.array([])

            for x in bar:

                x = torch.tensor(x, device=device, dtype=torch.float32)

                y_pred = model(x)

                y_pred = torch.sigmoid(y_pred)

                res = np.append(res, y_pred.detach().cpu().numpy())

            preds += res

    preds /= TTA

    return preds
prediction = np.array([0.]*len(test_df))

for i in range(len(models)):

    test_iterator = DataLoader(dataset=test, batch_size=16, shuffle=False, num_workers=num_workers)

    preds = get_predictions(models[i], test_iterator, device)

    prediction += preds

prediction /= len(models)
sub_df = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')

sub_df = sub_df[:50] if DEBUG else sub_df

sub_df['target'] = prediction



sub_df.to_csv('submission.csv', index=False) #indexをfalseにしないと、先頭列にindex情報が付加されたcsvファイルが出力されるので注意

sub_df.head()