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
import os

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import pickle as pkl

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.metrics import confusion_matrix,f1_score

from sklearn.model_selection import train_test_split
req_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca',

       'target', 'sex_0', 'sex_1', 'cp_0', 'cp_1', 'cp_2', 'cp_3',

       'fbs_0', 'fbs_1', 'restecg_0', 'restecg_1', 'restecg_2', 'exang_0',

       'exang_1', 'thal_0', 'thal_1', 'thal_2', 'thal_3']
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

df.head()
df.shape
corr = df.corr()

corr
def preprocessing(df):

    dummy_cols = ['sex','cp','fbs','restecg','exang','thal']

    df = pd.get_dummies(df,columns=dummy_cols)

    df['trestbps_risk'] = df['trestbps'].apply(lambda x : 1 if x>150 else 0)

    df['ca'] = df['ca'].apply(lambda x : 0 if x==3 else 1 if x==2 else 2 if x==1 else 3)

    cols = df.columns.values

    for x in req_cols:

        if x not in cols:

            df[x]=0

    return df



def scaling(df,scaler=None):

    if scaler==None:

        sc = StandardScaler()

        sc.fit(df)

        df = sc.transform(df)

        pkl.dump(sc,open("heart_scaler.pkl",'wb'))

    else:

        df = scaler.transform(df)

    return df
y = df['target']

X = df.drop(columns=['target'])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)
X_train = preprocessing(X_train)

X_train = scaling(X_train)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(penalty='l1',random_state=0)

logreg.fit(X_train,y_train)
X_test = preprocessing(X_test)

X_test = scaling(X_test)
y_pred = logreg.predict(X_test)
confusion_matrix(y_test,y_pred)
f1_score(y_test,y_pred)
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.grid(True)
auc(fpr, tpr)