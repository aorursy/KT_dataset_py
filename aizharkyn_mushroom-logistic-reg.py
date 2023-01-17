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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

datapath = pd.read_csv("../input/mushroom-classification/mushrooms.csv")

test = pd.read_csv("../input/mushroom-classification/mushrooms.csv")

train = pd.read_csv("../input/mushroom-classification/mushrooms.csv")
train.head()
train.isnull().sum()
train.shape

train.describe()
a=train['class']

b=sns.countplot(x=a,data=train)
def plot_data(hue, train):

    for i,col in enumerate(train.columns):

        plt.figure(i)

        sns.set(rc={'figure.figsize':(11.7,8.27)})

        ax=sns.countplot(x=train[col], hue=hue, data=train)
hue=train['class']

data_for_plot=train.drop('class',1)

plot_data(hue,data_for_plot)
le=LabelEncoder()

train['class']=le.fit_transform(train['class'])

train.head()
encoded_data=pd.get_dummies(train)

encoded_data.head()
from sklearn.model_selection import train_test_split

y=train['class'].values.reshape(-1,1)

X=encoded_data

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn import metrics
logistic_reg=LogisticRegression()

logistic_reg.fit(X_train,y_train.ravel())
y_prob=logistic_reg.predict_proba(X_test)[:,1]

y_pred=np.where(y_prob>0.5,1,0)
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)

confusion_matrix
auc_roc=metrics.roc_auc_score(y_test,y_pred)

auc_roc
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, threshold=roc_curve(y_test,y_prob)

roc_auc=auc(false_positive_rate,true_positive_rate)

def plot_roc(roc_auc):

    plt.figure(figsize=(10,10))

    plt.title('Operating Characteristic')

    plt.plot(false_positive_rate,true_positive_rate,color='red',label='AUC=%0.2f'%roc_auc)

    plt.legend(loc='lower right')

    plt.plot([0,1],[0,1],linestyle='--')

    plt.axis('tight')

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

plot_roc(roc_auc)

    
