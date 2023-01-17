# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/heart.csv')

df.head(5)
df.describe()
df.info()
sns.set_palette('Set2')

sns.countplot(x='target',data=df)
print('Male :',df.sex.value_counts().tolist()[0])

print('Female :',df.sex.value_counts().tolist()[1])

sns.countplot(x='sex',data=df)
df.shape
continuous = ['age','trestbps','chol','thalach','oldpeak']

color = ['blue','green','red','yellow','orange']

for i,j in zip(continuous,color):

    sns.distplot(df[i],color=j)

    plt.show()
categorical = [cat for cat in df.columns.tolist() if cat not in continuous]

categorical.remove('sex')

for i,j in zip(categorical,color):

    sns.countplot(x=i,data=df,color=j,hue=df.sex)

    plt.show()
corr = df.corr()

plt.figure(figsize=(20,10))

sns.heatmap(corr,annot=True,linewidths=5)
categorical.remove('target')

for i in categorical:

    df[i] = df[i].astype('object')

df.info()
y = df.iloc[:,13].values

df.drop(labels='target',axis=1,inplace=True)

df = pd.get_dummies(df,drop_first=True)

df.head()
from sklearn.model_selection import train_test_split

np.random.seed(42)

x = df.iloc[:,:-1].values

xtr, xtst, ytr, ytst = train_test_split(x,y,test_size=0.20,random_state=0)

print('X train : {}\tY train :{}\nX test : {}\tY test : {}'.format(xtr.shape,ytr.shape,xtst.shape,ytst.shape))
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier

from sklearn.naive_bayes import BernoulliNB, GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier



models = []

models.append(('XGBoost',XGBClassifier()))

models.append(('LightGBM',LGBMClassifier()))

models.append(('AdaBoostClassifier',AdaBoostClassifier()))

models.append(('Bagging',BaggingClassifier()))

models.append(('Extra Trees Ensemble', ExtraTreesClassifier()))

models.append(('Gradient Boosting',GradientBoostingClassifier()))

models.append(('Random Forest', RandomForestClassifier()))

models.append(('BNB',BernoulliNB()))

models.append(('GNB',GaussianNB()))

models.append(('KNN',KNeighborsClassifier()))

models.append(('MLP',MLPClassifier()))

models.append(('DTC',DecisionTreeClassifier()))

models.append(('ETC',ExtraTreeClassifier()))
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

%matplotlib inline



best_model = None

best_model_name = ""

best_valid = 0



for name, model in models:

    model.fit(xtr,ytr)

    proba = model.predict_proba(xtst)[:,1]

    score = roc_auc_score(ytst, proba)

    fpr, tpr, _  = roc_curve(ytst, proba)

    plt.figure()

    plt.plot(fpr, tpr, color='darkorange', label=f"ROC curve (auc = {score})")

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    plt.title(f"{name} Results")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.legend(loc="lower right")

    plt.show()

    if score > best_valid:

        best_valid = score

        best_model = model

        best_model_name = name



print(f"Best model is {best_model_name}")