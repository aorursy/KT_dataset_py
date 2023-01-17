import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
ir=pd.read_csv("/kaggle/input/iris/Iris.csv")
ir.sample(10)
#Knowing data types and information



ir.info()
ir.drop(['Id'],axis=1,inplace=True)
# Show random rows

ir.sample(10)
#Checking NaN values 

ir.isna().any()
## Data description

ir.describe()
# data exploration 

# check class distributions

import plotly.graph_objects as go

fig = go.Figure(data=[

    go.Pie(labels=['Iris-setosa','Iris-virginica', 'Iris-versicolor'],

           values=ir['Species'].value_counts())

])

fig.update_layout(title_text='class distributions')

fig.show()
# first we plot histogram of numerical fatures

num_features=ir[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

num_features.hist(bins=50,figsize=(20,15))

plt.show()
sns.boxplot(x='Species', y='PetalLengthCm', data=ir)
sns.boxplot(x='Species', y='PetalWidthCm', data=ir)
ir['Species'].value_counts()
ir.sample(10)
# we need to one hot encode all categorical features

label=ir['Species']

ir.drop(['Species'],inplace=True,axis=1)

ir=pd.get_dummies(ir)

ir.sample(10)
# we will SVM

from sklearn.svm import SVC #import svm as classifier

from sklearn.model_selection import train_test_split



xtrain,xtest,ytrain,ytest=train_test_split(ir,label,test_size=0.25)
# train model

svm = SVC(class_weight='balanced') # create new svm classifier with default parameters

svm.fit(xtrain,ytrain)
from sklearn.metrics import accuracy_score

predictions = svm.predict(xtest) # test model against test set

preds_train=svm.predict(xtrain)

print("Model Acurracy in testing = {}".format(accuracy_score(ytest, predictions))) # print test accuracy

print("Model Acurracy in train = {}".format(accuracy_score(ytrain, preds_train))) # print train accuracy
# confusion matrix

from sklearn.metrics import plot_confusion_matrix # only valid in sklearn versions above or equal scikit-learn==0.22.0

plot_confusion_matrix(svm, xtest, ytest)

plt.show()
# evaluate performance on train set

plot_confusion_matrix(svm, xtrain, ytrain)

plt.show()