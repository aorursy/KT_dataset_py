#linear algebra
import numpy as np

#dataframe
import pandas as pd

#data visualization
import matplotlib.pyplot as plt
import seaborn as sns

#regex
import re

#machine learning
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report,f1_score,accuracy_score
from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from vecstack import stacking

#neural network
import tensorflow as tf
from tensorflow import keras

import missingno
test = pd.read_csv("../input/prsits/test.csv")
train = pd.read_csv("../input/prsits/train.csv")
missingno.matrix(train, figsize = (15,8))

missingno.matrix(test, figsize = (15,8))
train['BILL_AMT6']-train['PAY_AMT5']
train.head()
train.BILL_AMT3



def facto(a):
    for i in train:
        train[a] = train[a].factorize()[0]    
        
facto('SEX')
facto('EDUCATION')
facto('MARRIAGE')

train.rename({'default.payment.next.month' : 'nunggak'}, axis=1, inplace=True)

train.head(10)
def facto(a):
    for i in test:
        test[a] = test[a].factorize()[0]    
        
facto('SEX')
facto('EDUCATION')
facto('MARRIAGE')

test.head()
train['Selisih 9']=train['BILL_AMT1']-train['PAY_AMT1']


train.head()
train.PAY_6
train.corr()["nunggak"].sort_values(ascending = False)
# Correlation heatmap between numerical values (SibSp Parch Age and Fare values) and Survived 
g = sns.heatmap(train[["EDUCATION","SEX","MARRIAGE","LIMIT_BAL","AGE","nunggak"]].corr(),annot=True
                , fmt = ".2f", cmap = "coolwarm")
def prob_box_plot_with_annotation(x, y, ax_X, ax_Y):
    """ Docstring
    Fungsi ini digunakan untuk melakukan plotting boxplot dan memberikan annotasi berupa nilai probability pada masing-masing bar
    
    Parameter yang dibutuhkan ada 4, yaitu x, y, dan nilai axes x dan y. x dan y merupakan input parameter barplot dari seaborn 
    yang dapat dibaca pada dokumentasinya.
    """
    ax = sns.barplot(x, y, ax = axes[ax_X, ax_Y])
    ax.set_title('\n\nProbabilitas terjadi nunggak berdasarkan \n\n {}\n\n'.format(x.name))
    for p in ax.patches:
        ax.annotate(np.round(p.get_height(),decimals= 3), (p.get_x() + p.get_width()/2., p.get_height()), 
                       ha = 'center', va = 'center', xytext = (0, 25), textcoords = 'offset points')
fig, axes = plt.subplots(5, 3, figsize=(18, 14))

# Menggunakan fungsi yang telah dibuat untuk membuat boxplot dengan mudah
prob_box_plot_with_annotation(train['SEX'], train['nunggak'], 0, 0)
prob_box_plot_with_annotation(train['EDUCATION'], train['nunggak'], 0, 1)
prob_box_plot_with_annotation(train['MARRIAGE'], train['nunggak'], 0, 2)

fig, axes=plt.subplots(2,3,figsize=(18,14))
prob_box_plot_with_annotation(train['PAY_0'],train['nunggak'],0 , 0)
prob_box_plot_with_annotation(train['PAY_2'],train['nunggak'],0 , 1 )
prob_box_plot_with_annotation(train['PAY_3'],train['nunggak'],0 , 2)
prob_box_plot_with_annotation(train['PAY_4'],train['nunggak'],1 ,0  )
prob_box_plot_with_annotation(train['PAY_5'],train['nunggak'],1 ,1  )
prob_box_plot_with_annotation(train['PAY_6'],train['nunggak'],1 , 2 )
sns.jointplot("LIMIT_BAL", "AGE", data=train, kind="reg")
train.corr()["nunggak"].sort_values(ascending = False)
train.drop(["BILL_AMT6"],axis=1,inplace=True)
train.drop(["ID"],axis=1,inplace=True)
train.drop(["BILL_AMT5"],axis=1,inplace=True)
train.drop(["BILL_AMT4"],axis=1,inplace=True)
train.drop(["BILL_AMT3"],axis=1,inplace=True)
train.drop(["BILL_AMT2"],axis=1,inplace=True)
train.drop(["BILL_AMT1"],axis=1,inplace=True)
train.drop(["PAY_AMT1"],axis=1,inplace=True)
train.drop(["PAY_AMT3"],axis=1,inplace=True)
train.drop(["PAY_AMT5"],axis=1,inplace=True)
train.drop(["PAY_AMT6"],axis=1,inplace=True)
train.drop(["PAY_AMT2"],axis=1,inplace=True)
train.drop(["PAY_AMT4"],axis=1,inplace=True)
train.drop(["PAY_0"],axis=1,inplace=True)
train.drop(["PAY_2"],axis=1,inplace=True)
train.drop(["PAY_3"],axis=1,inplace=True)
train.drop(["PAY_4"],axis=1,inplace=True)
train.drop(["PAY_5"],axis=1,inplace=True)
train.drop(["PAY_6"],axis=1,inplace=True)

train
X = train.drop(["nunggak"],axis=1)
y = train["nunggak"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
lr_clf = LogisticRegression(max_iter=1000)
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)
f1_score(lr_clf.predict(X_test),y_test)
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train,y_train)
knn_clf.score(X_test,y_test)
f1_score(knn_clf.predict(X_test),y_test)
train["LIMIT_BAL"].describe()