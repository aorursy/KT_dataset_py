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
df = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")
df.head()
df.isnull().sum()
df1=df
from sklearn.preprocessing import LabelEncoder

lc = LabelEncoder()   # Created a object of label encoder
l1 = []

l1 = df.columns
#for i in range(0,len(l1)):

for i in l1:

    df[i] = lc.fit_transform(df[i])
y = df["class"]

x = df.drop("class",axis=1)
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2,random_state=0,stratify=y)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score, auc

from sklearn.metrics import roc_auc_score
logreg = LogisticRegression()

logreg.fit(train_x,train_y)
train_predict = logreg.predict(train_x)

test_predict = logreg.predict(test_x)
print(f1_score(train_predict,train_y))
print(f1_score(test_predict,test_y))
from sklearn.metrics import confusion_matrix

confusion_matrix(train_y,train_predict,labels=[0,1])
train_predict_prob = logreg.predict_proba(train_x)
train_predict_prob
train_predict_prob = train_predict_prob[:,1]
train_predict_prob_1 = train_predict_prob

train_predict_prob_1
for i in range(0,len(train_predict_prob)):

    if(train_predict_prob[i] < 0.6):

        train_predict_prob_1[i] = 0

    else:

        train_predict_prob_1[i] = 1
print(f1_score(train_predict_prob_1,train_y))
confusion_matrix(train_y,train_predict_prob_1)