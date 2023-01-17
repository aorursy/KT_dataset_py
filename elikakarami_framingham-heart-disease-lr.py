# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





#required libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pylab as pl

import scipy.optimize as opt

from sklearn import preprocessing

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report, confusion_matrix

import itertools



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#load data from CSV



df = pd.read_csv("/kaggle/input/logistic-regression-heart-disease-prediction/framingham_heart_disease.csv")

df.drop(['education'],axis=1,inplace=True)

df.drop(['currentSmoker'],axis=1,inplace=True)

df.drop(['cigsPerDay'],axis=1,inplace=True)

df.drop(['male'],axis=1,inplace=True)

df.head()
#remove null



df = df.dropna(axis=0)

df.isnull().sum()
#correlation matrix



def correlation_heatmap(df):

    correlations = df.corr()



    fig, ax = plt.subplots(figsize=(10,10))

    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',cmap = 'twilight_shifted',

                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})

    plt.show();

    

correlation_heatmap(df)
#data preprocessing



df = df[[ 'age','BPMeds','prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',

       'diaBP', 'BMI', 'heartRate', 'glucose', 'TenYearCHD']]



df['TenYearCHD'] = df['TenYearCHD'].astype(int)





df.head()
#define x



x = np.asarray(df[['age','BPMeds','prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',

       'diaBP', 'BMI', 'heartRate', 'glucose',]])



x[0:5]
#define y



y = np.asarray(df['TenYearCHD'])

y [0:5]
#normalization



x = preprocessing.StandardScaler().fit(x).transform(x)

x[0:5]
#train and test split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

print('Train set:', x_train.shape, y_train.shape)

print('Test set:', x_test.shape, y_test.shape)
#using lofistic regression



LR = LogisticRegression(C=0.01 , solver='liblinear').fit(x_train, y_train)

LR
#predict



y_hat = LR.predict(x_test)
#accuracy



accuracy_score(y_test,y_hat)
#confiusion matrix



def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.cool):



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

print(confusion_matrix(y_test, y_hat, labels=[1,0]))
#compute confusion matrix



cnf_matrix = confusion_matrix(y_test, y_hat, labels=[1,0])

np.set_printoptions(precision=2)





#plot non-normalized confusion matrix



plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['CHD=1','CHD=0'],normalize= False,  title='Confusion matrix')