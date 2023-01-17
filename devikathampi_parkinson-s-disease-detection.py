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
import numpy as np

import pandas as pd

import os,sys

from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
df = pd.read_csv("../input/detection-of-parkinson-disease/parkinsons.csv")

df.head(20)
df.info()
features = df.drop(columns="status")

labels=df.status

features.head(5)
labels
labels =pd.DataFrame(labels)

labels.head(5)
features=features.drop(columns="name")

features.head(5)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(3,4))

sns.set(style="darkgrid")

sns.countplot(x="status",data=df,palette=["Red","Blue"])

plt.title("Status",size=20)

plt.show()
df.status.value_counts()
scaler = MinMaxScaler((-1,1))

x=scaler.fit_transform(features)

y=labels
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=7)
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=7, sampling_strategy=1.0)

X_train, y_train = sm.fit_sample(X_train, y_train)

X_train.shape, y_train.shape
y_train = pd.DataFrame(y_train, columns = ['status'])

y_train.status.value_counts()
model_1 =XGBClassifier()

model_1.fit(X_train,y_train)
xgb_predict = model_1.predict(X_test)

print(accuracy_score(xgb_predict,y_test))
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,n_jobs=-1,train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training Examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="b")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
g=plot_learning_curve(model_1, "XGBoost Learning Curve",X_train,y_train)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

model_2= rfc.fit(X_train,y_train)

rfc_predict = model_2.predict(X_test)

print(accuracy_score(rfc_predict,y_test))
g=plot_learning_curve(model_2, "Random forest classifier Learning Curve",X_train,y_train)