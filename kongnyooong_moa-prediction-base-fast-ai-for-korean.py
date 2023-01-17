import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

import warnings



%matplotlib inline

warnings.filterwarnings("ignore")

plt.style.use("seaborn-whitegrid")
df_train = pd.read_csv("../input/lish-moa/train_features.csv")

df_test = pd.read_csv("../input/lish-moa/test_features.csv")

train_labels = pd.read_csv("../input/lish-moa/train_targets_scored.csv")

submission = pd.read_csv("../input/lish-moa/sample_submission.csv")

data = pd.concat([df_train, df_test])
df_train.head()
df_test.head()
train_labels.head()
print("학습데이터 rows:", df_train.shape[0], "\t학습데이터 columns:", df_train.shape[1])

print("테스트데이터 rows:", df_test.shape[0], "\t테스트데이터 columns:", df_test.shape[1])

print("전체데이터 rows:", data.shape[0], "\t전체데이터 columns:", data.shape[1])
df_train.info()



# dtypes: float64(872), int64(1), object(3)
df_train.isnull().sum()



# missingno로 확인해본결과 nullvalue 없음
msno.matrix(df=df_train.iloc[:, 800:], color=(0.1, 0.6, 0.8))
print("cp_type의 유니크 값 확인:", df_train["cp_type"].unique())

print("cp_dose의 유니크 값 확인:", df_train["cp_dose"].unique())
f, ax = plt.subplots(1, 3, figsize = (20, 6))



plot1 = sns.countplot(x = "cp_type", data = data, palette = "cool", edgecolor='black', alpha=0.7, linewidth=0.8, ax=ax[0])

plot2 = sns.countplot(x = "cp_time", data = data, palette = "cool", edgecolor='black', alpha=0.7, linewidth=0.8, ax=ax[1])

plot3 = sns.countplot(x = "cp_dose", data = data, palette = "cool", edgecolor='black', alpha=0.7, linewidth=0.8, ax=ax[2])



# ctl_vehicle은 매우 적다.

# cp_time, cp_dose는 고르게 분포되어 있다.
f = plt.subplots(figsize = (12, 12))



ax = plt.subplot2grid((2,2),(0,0))

plt.hist(data["c-10"], bins=4, color='skyblue', alpha=0.7, edgecolor='black', linewidth = 0.8)

plt.title("c-10", weight='bold', fontsize=18)



ax = plt.subplot2grid((2,2),(0,1))

plt.hist(data["c-30"], bins=4, color='lightcoral', alpha=0.7, edgecolor='black', linewidth = 0.8)

plt.title("c-30", weight='bold', fontsize=18)



ax = plt.subplot2grid((2,2),(1,0))

plt.hist(data["c-60"], bins=4, color='purple', alpha=0.7, edgecolor='black', linewidth = 0.8)

plt.title("c-60", weight='bold', fontsize=18)



ax = plt.subplot2grid((2,2),(1,1))

plt.hist(data["c-90"], bins=4, color='orange', alpha=0.7, edgecolor='black', linewidth = 0.8)

plt.title("c-90", weight='bold', fontsize=18)



# 0~1, %와 같이 정량화되어 있지 않아보임

# 눈으로 봐서는 -10~10이 아닐까 하지만 확실히 알아봐야 할듯
treated= data[data['cp_type']=='trt_cp']

control= data[data['cp_type']=='ctl_vehicle']



f = plt.subplots(figsize = (12, 5))



ax = plt.subplot2grid((1,2),(0,0))

plt.hist(control["c-30"], bins=4, color='mediumpurple', alpha=0.7, edgecolor='black', linewidth = 0.8)

plt.title("Control", weight='bold', fontsize=18)



ax = plt.subplot2grid((1,2),(0,1))

plt.hist(treated["c-30"], bins=4, color='darkcyan', alpha=0.7, edgecolor='black', linewidth = 0.8)

plt.title("Treated with Compound", weight='bold', fontsize=18)



# 대체적으로 살펴보면, trt_cp의 생존력이 더 강한것을 알 수 있다.
hours_24= data[data['cp_time']==24]

hours_48= data[data['cp_time']==48]

hours_72= data[data['cp_time']==72]



f = plt.subplots(figsize = (18, 5))



ax = plt.subplot2grid((1,3),(0,0))

plt.hist(hours_24["c-30"], bins=4, color='forestgreen', alpha=0.7, edgecolor='black', linewidth = 0.8)

plt.title("Treatment Duration 24 Hours", weight='bold', fontsize=18)



ax = plt.subplot2grid((1,3),(0,1))

plt.hist(hours_48["c-30"], bins=4, color='tomato', alpha=0.7, edgecolor='black', linewidth = 0.8)

plt.title("Treatment Duration 48 Hours", weight='bold', fontsize=18)



ax = plt.subplot2grid((1,3),(0,2))

plt.hist(hours_72["c-30"], bins=4, color='slateblue', alpha=0.7, edgecolor='black', linewidth = 0.8)

plt.title("Treatment Duration 72 Hours", weight='bold', fontsize=18)
!pip uninstall fastai -y

!pip install /kaggle/input/fast-v2-offline/dataclasses-0.6-py3-none-any.whl

!pip install /kaggle/input/fast-v2-offline/torch-1.6.0-cp37-cp37m-manylinux1_x86_64.whl

!pip install /kaggle/input/fast-v2-offline/torchvision-0.7.0-cp37-cp37m-manylinux1_x86_64.whl

!pip install /kaggle/input/fast-v2-offline/fastcore-1.0.1-py3-none-any.whl

!pip install /kaggle/input/fast-v2-offline/fastai-2.0.8-py3-none-any.whl



from fastai.tabular.all import *
n_splits = 5

seed = 1337

test_size = 0.15

layers = [1024, 512, 256]

bs = 4096

epochs = 48

lr = slice(8e-4, 8e-3)
for column in train_labels.columns:

    print(column)
cat_names = ['cp_type', 'cp_time', 'cp_dose']

cont_names = [c for c in df_train.columns if c not in cat_names and c != 'sig_id']

y_names = [c for c in train_labels.columns if c != 'sig_id']
train = pd.concat([df_train, train_labels], axis=1)
from sklearn.model_selection import KFold



sss = KFold(n_splits=n_splits, shuffle=True, random_state=seed)



out = np.zeros((len(submission), len(submission.columns) - 1))



for _, val_index in sss.split(df_train):

    splits = IndexSplitter(val_index)(train)

    procs = [Categorify, Normalize]

    tab_pan = TabularPandas(train, procs=procs, cat_names=cat_names, cont_names=cont_names, y_names=y_names, splits=splits)

    dls = tab_pan.dataloaders(bs=bs)

    learn = tabular_learner(dls, y_range=(0,1), layers=layers, loss_func=BCELossFlat())

    learn.fit_one_cycle(epochs, lr)

    test_dl = learn.dls.test_dl(df_test)

    sub = learn.get_preds(dl=test_dl)

    out += sub[0].numpy()

    

out /= n_splits
moa_cols = [c for c in submission.columns if c != 'sig_id']

dummy = np.zeros(len(moa_cols))

ctl_cp_ids = df_train.query('cp_type == "trt_cp"')["sig_id"].values

submission[moa_cols] = out
submission.head()
submission.to_csv('submission.csv', index=False)