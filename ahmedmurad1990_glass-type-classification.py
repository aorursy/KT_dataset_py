# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import numpy as np  # linear algebra

import pandas as pd  # read and wrangle dataframes

import matplotlib.pyplot as plt # visualization

import seaborn as sns # statistical visualizations and aesthetics

from sklearn.base import TransformerMixin # To create new classes for transformations

from sklearn.preprocessing import (FunctionTransformer, StandardScaler) # preprocessing 

from sklearn.decomposition import PCA # dimensionality reduction

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from scipy.stats import boxcox # data transform

from sklearn.model_selection import (train_test_split, KFold , StratifiedKFold, 

                                     cross_val_score, GridSearchCV, 

                                     learning_curve, validation_curve) # model selection modules

from sklearn.pipeline import Pipeline # streaming pipelines

from sklearn.base import BaseEstimator, TransformerMixin # To create a box-cox transformation class

from collections import Counter

import warnings

# load models

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from xgboost import (XGBClassifier, plot_importance)

from sklearn.svm import SVC

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from time import time



%matplotlib inline 

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/glass/glass.csv')

features = df.columns[:-1].tolist()

print(df.shape)
df
df.dtypes
df.describe()

df['Type'].value_counts()

attribute=df.columns.tolist()

attribute
for atr in attribute:

    #a=df[atr].skew()

    plt.hist(df[atr])

    plt.show()
for i in range(len(attribute)-1):

    figure = plt.figure()

    ax = sns.boxplot(x='Type', y=attribute[i], data=df)

plt.figure(figsize=(9,9))

sns.pairplot(df[attribute],palette='coolwarm')

plt.show()
l=['Type']



x=df.drop(l,axis=1)

x.head()
#TARGET

y=df['Type']

y.head()
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state =42)

dt= DecisionTreeClassifier().fit(X_train, y_train)#overfitting

print(dt.score(X_train, y_train))

print(dt.score(X_test, y_test))
r= RandomForestClassifier().fit(X_train, y_train)#overfitting

print(r.score(X_train, y_train))

print(r.score(X_test, y_test))
scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)

X_test_s= scaler.transform(X_test)
lg1=LogisticRegression().fit(X_train_s, y_train)

print(lg1.score(X_train_s, y_train))

print(lg1.score(X_test_s, y_test))

clf1 = SVC().fit(X_train_s, y_train)

print(clf1.score(X_train_s, y_train))

print(clf1.score(X_test_s, y_test))
from sklearn.model_selection import cross_val_score

all_accuracies = cross_val_score(estimator=r, X=X_train, y=y_train, cv=5)

np.mean(all_accuracies)
from sklearn.metrics import accuracy_score

def score_dataset(X_train, X_valid, y_train, y_valid, n_estimators ):

    model = RandomForestClassifier(n_estimators = n_estimators)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return accuracy_score(y_valid, preds)





n_es_range = list(range(200,2000,20))



for n_es in n_es_range:

    my_acc = score_dataset( X_train, X_test, y_train, y_test,n_es )

    print(f" n_estimators: {n_es}  \t\t my_acc: {my_acc}")
