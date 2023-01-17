# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/eval-lab-2-f464/train.csv")

test = pd.read_csv("/kaggle/input/eval-lab-2-f464/test.csv")
train = train.drop(['id'], axis=1)
train.info()
train.describe()
train.head(10)
test_id = test.drop(['id'],axis=1)
import pandas as pd

import numpy as np

import seaborn as sns



#get correlations of each features in dataset

corrmat = train.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
np.random.seed(0)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()

X_lda = lda.fit_transform(train.drop(['class'],axis=1), train['class'])

trainlda = pd.DataFrame(lda.transform(train.drop(['class'],axis=1)))

train['lda0'] = trainlda[0]

train['lda1'] = trainlda[1]

train['lda2'] = trainlda[2]

train['lda3'] = trainlda[3]

train['lda4'] = trainlda[4]
train.head()
test_lda = pd.DataFrame(lda.transform(test_id))

test_id['lda0'] = test_lda[0]

test_id['lda1'] = test_lda[1]

test_id['lda2'] = test_lda[2]

test_id['lda3'] = test_lda[3]

test_id['lda4'] = test_lda[4]
test_id.head()
train['create'] = (train['attribute']*train['chem_3']*train['chem_4'])

import pandas as pd

import numpy as np

import seaborn as sns



#get correlations of each features in dataset

corrmat = train.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")

test_id['create'] = (test_id['attribute']*test_id['chem_3']*test_id['chem_4'])
X = train.drop(['class'],axis=1)

y = train['class']
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()

Xlda = lda.fit_transform(X, y)
plt.scatter(Xlda[:,0],Xlda[:,1],c=y,cmap='rainbow',alpha=0.7,edgecolors='b')
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler

ss = StandardScaler()

cols = test_id.columns

st = train.copy()

st2 = test_id.copy()

st[cols]=ss.fit_transform(train[cols])

st2[cols]=ss.transform(test_id[cols])
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()

Xlda = lda.fit_transform(X, y)

plt.xlabel('LDA2 X Axis')

plt.ylabel('LDA2 Y Axis')

plt.scatter(Xlda[:,0],Xlda[:,1],c=y,cmap='rainbow',alpha=0.7,edgecolors='b')
st.head()
st2.head()
from sklearn.decomposition import PCA

model=PCA(n_components=2)

model_data = model.fit(st.drop('class',axis=1)).transform(st.drop('class',axis=1))
plt.title('PCA')

plt.legend()

plt.scatter(model_data[:,0],model_data[:,1],label = train['class'],c=train['class'])

plt.show()
X = st.drop(['class'],axis=1)

y = st['class']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.metrics import accuracy_score

from sklearn.ensemble import ExtraTreesClassifier

extratree = ExtraTreesClassifier(n_estimators=1000, random_state=42, class_weight="balanced")

extratree.fit(X_train,y_train)

pred = extratree.predict(X_test)

accuracy_score(y_test,pred)
extratree = ExtraTreesClassifier(n_estimators=1000, random_state=42, class_weight="balanced")

extratree.fit(X,y)

test['class'] = extratree.predict(st2)

final = test[['id','class']]

final.info()
pd.set_option('display.max_rows',121)

final.head(100)