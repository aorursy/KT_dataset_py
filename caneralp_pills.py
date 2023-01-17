# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/drug-classification/drug200.csv")

df
from sklearn.preprocessing import LabelEncoder

df["BP"]=LabelEncoder().fit_transform(df["BP"])

df["Sex"]=LabelEncoder().fit_transform(df["Sex"])

df["Cholesterol"]=LabelEncoder().fit_transform(df["Cholesterol"])

df["Drug"]=LabelEncoder().fit_transform(df["Drug"])

df
y=df["Drug"]

x=df.drop(["Drug"],axis=1)
import pandas_profiling as pp

pp.ProfileReport(df)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
from sklearn.preprocessing import StandardScaler

model=StandardScaler().fit(x_train)

x_train=model.transform(x_train)

x_test=model.transform(x_test)
from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression(C=50).fit(x_train,y_train)

print("Train score:",logreg.score(x_train,y_train))

print("Test score:",logreg.score(x_test,y_test))
from sklearn.linear_model import LogisticRegression

logr=LogisticRegression(C=100)

from sklearn.model_selection import cross_val_score

mean=np.mean(cross_val_score(estimator=logr,X=x_train,y=y_train,cv=50))

mean
logreg=LogisticRegression(C=100).fit(x_train,y_train)

y_pred=logreg.predict(x_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True)

plt.ylabel("Real")

plt.xlabel("Predict")
Drug=pd.DataFrame({"Real":y_test,"Predict":y_pred})

Drug