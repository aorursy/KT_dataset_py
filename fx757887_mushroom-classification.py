# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from sklearn import preprocessing

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.feature_selection import RFE
mushroom=pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")
mushroom.head()
mushroom[1:]
mushcol=mushroom.iloc[:,1:].columns



plt.figure(figsize=(25,21*5))

gs=gridspec.GridSpec(21,4)



for i, col in enumerate(mushroom[mushcol]):

    ax=plt.subplot(gs[i])

    sns.countplot(x=col, hue="class", palette="Greens_d", data=mushroom)
mushroom.info()
mushroom=pd.get_dummies(mushroom, drop_first=True)
mushroom.head()
#欠損値の確認

mushroom.isnull().sum()
mushroomcol=mushroom.columns[1:]
mushroom.describe()
mushroom.shape
train_set,test_set=train_test_split(mushroom, test_size=0.2, random_state=42)
y_train=train_set["class_p"]

X_train=train_set.drop("class_p",axis=1)



y_test=test_set["class_p"]

X_test=test_set.drop("class_p",axis=1)
#モデルの学習

Log_model=LogisticRegression()



#RFE(再帰的特徴消去)を用いた特徴量抽出

rfe=RFE(Log_model, 5, verbose=1)
#RFEの実践

rfe=rfe.fit(X_train,y_train)
X_train=X_train[X_train.columns[rfe.support_]]

X_test=X_test[X_test.columns[rfe.support_]]



X_train.head()
X_test.head()
#モデルの訓練と評価

Log_model2=LogisticRegression()



#抽出した特徴量にてモデルの学習

Log_model2.fit(X_train,y_train)
#訓練データ(X_train)にて予測(y_pred_train)

y_pred_train=Log_model2.predict(X_train)
#正解率

accuracy_score(y_train,y_pred_train)
#テストデータ(X_test)にて予測(y_pred_test)

y_pred_test=Log_model2.predict(X_test)
#正解率

accuracy_score(y_test,y_pred_test)