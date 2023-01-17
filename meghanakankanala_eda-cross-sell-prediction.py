# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("/kaggle/input/janatahack-crosssell-prediction/train.csv")
test=pd.read_csv("/kaggle/input/janatahack-crosssell-prediction/test.csv")
print(train.shape)
print(test.shape)
train.info()
train.head()
train.describe()
bins=[20,40,60,80,100]
train["cut_age"]=pd.cut(train["Age"],bins)
train["cut_age"].value_counts()
train.nunique()
figure,axisarr=plt.subplots(1,2,figsize=(8,8))
a=sns.countplot(train.Gender,ax=axisarr[0]).set_title("SEX COUNT")
b=sns.barplot(x="Gender",y="Response",data=train,ax=axisarr[1]).set_title("response according to sex")
ax=axisarr[1].set_ylabel("response")
plt.subplots_adjust(wspace=0.8)
train.isnull().sum()

train["binned_age"]=pd.qcut(train["Age"],q=4)
train["binned_age"].value_counts()
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,8))
plt.subplots_adjust(wspace=0.5)
sns.countplot(train["binned_age"],ax=ax[0]).set_title("Age distribution")
sns.barplot(x="binned_age",y="Response",data=train,ax=ax[1]).set_title("response according to binned_age" )
sns.barplot(x="binned_age",y="Response",data=train,hue="Gender").set_title("response according to binned_age" )
# fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,8))
# plt.subplots_adjust(wspace=0.5)
# sns.countplot(train["Driving_License"],ax=ax[0]).set_title("Age distribution")
sns.barplot(x="Vehicle_Age",y="Response",data=train,hue="Vehicle_Damage").set_title("response according to vehicle" )
train["binned_annualpremium"]=pd.qcut(train["Annual_Premium"],q=4)
print(train["binned_annualpremium"].unique())
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,8))
plt.subplots_adjust(wspace=0.5)
sns.countplot(train["binned_annualpremium"],ax=ax[0]).set_title("binned_annualpremium distribution")
sns.barplot(x="binned_annualpremium",y="Response",data=train,hue="Vehicle_Age",ax=ax[1]).set_title("response according to annualpremium" )
ax[1].set_xlabel(' ', fontsize=8)
from sklearn.preprocessing import OneHotEncoder
encode=OneHotEncoder()
Train=encode.fit_transform(train)
Trains=np.array(Train)
df=pd.DataFrame(Trains)
df

bins=[]