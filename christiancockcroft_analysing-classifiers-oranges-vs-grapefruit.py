import pandas as pd

from pandas import DataFrame

import numpy as np

from sklearn import preprocessing

import math

from sklearn.model_selection import train_test_split

from sklearn import cross_validation, metrics     

import seaborn as sns

import matplotlib.pyplot as plt

from ggplot import *

import itertools

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('../input/oranges-vs-grapefruit/citrus.csv') ## Modified input data from original

le = preprocessing.LabelEncoder()

df['name'] = le.fit_transform(df['name']) ## Modified name column contains only object values.

df.head()
df = df.select_dtypes(exclude=['object'])
# Dividing the data into quantiles and doing the outlier analysis.



Q1 = df.quantile(0.25)

Q3 = df.quantile(0.75)

IQR = Q3 - Q1



((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
# Heatmap showing the correlation of various columns with each other.



ax = plt.figure(figsize = (8,8)) ## Modify figsize from original

ax = sns.heatmap(df.corr(),cmap = 'gist_heat')
# The features with skewed or non-normal distribution.

skew_df = df[['diameter','red','green','blue']] ## Adapt to the data columns we are interested in.

fig , ax = plt.subplots(2,2,figsize = (20,10)) ## Modify subplot array to 2x2

col = skew_df.columns

for i in range(2): ## Range is now 2

    for j in range(2): ## Range is now 2

        ax[i][j].hist(skew_df[col[2 * i + j]] , color = 'k') ## Modified to match subplot

        ax[i][j].set_title(str(col[2 * i + j])) ## Modified to match subplot

        ax[i][j].set_axis_bgcolor((1, 0, 0))
target = df['name'] ##Adapt to our data

train = df.drop('name',axis = 1) ##Adapt to our data

train.shape
pd.value_counts(target).plot(kind = 'bar',cmap = 'BrBG')

plt.rcParams['axes.facecolor'] = 'blue'

plt.title("Count of classes")
train_accuracy = []

test_accuracy = []

models = ['SVM' , 'XgBoost'] ## Choose the models we work with
#Defining a function which will give us train and test accuracy for each classifier.

def train_test_error(y_train,y_test):

    train_error = ((y_train==Y_train).sum())/len(y_train)*100

    test_error = ((y_test==Y_test).sum())/len(Y_test)*100

    train_accuracy.append(train_error)

    test_accuracy.append(test_error)

    print('{}'.format(train_error) + " is the train accuracy")

    print('{}'.format(test_error) + " is the test accuracy")
X_train, X_test, Y_train, Y_test = train_test_split(

    train, target, test_size=0.33, random_state=42)
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

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
from sklearn import svm

SVM = svm.SVC(probability=True)

SVM.fit(X_train,Y_train)

train_predict = SVM.predict(X_train)

test_predict = SVM.predict(X_test)

train_test_error(train_predict , test_predict)
class_names = ['0', '1']

confusion_matrix=metrics.confusion_matrix(Y_test,test_predict)

plot_confusion_matrix(confusion_matrix, classes=class_names,

                      title='Confusion matrix')
probs = SVM.predict_proba(X_test)

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)

roc_auc = metrics.auc(fpr, tpr)

df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))

ggplot(df, aes(x = 'fpr', y = 'tpr'))+ geom_line(aes(y = 'tpr')) + geom_abline(linetype = 'dashed') + geom_area(alpha = 0.1) + ggtitle("ROC Curve w/ AUC = %s" % str(roc_auc))
import xgboost

from xgboost import XGBClassifier

XgB = XGBClassifier(max_depth=1,min_child_weight=1,gamma=0.0,subsample=0.8,colsample_bytree=0.75,reg_alpha=1e-05)

XgB.fit(X_train,Y_train)

train_predict = XgB.predict(X_train)

test_predict = XgB.predict(X_test)

train_test_error(train_predict,test_predict)
confusion_matrix=metrics.confusion_matrix(Y_test,test_predict)

plot_confusion_matrix(confusion_matrix, classes=class_names,

                      title='Confusion matrix')
probs = XgB.predict_proba(X_test)

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)

roc_auc = metrics.auc(fpr, tpr)

df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))

ggplot(df, aes(x = 'fpr', y = 'tpr'))+ geom_line(aes(y = 'tpr')) + geom_abline(linetype = 'dashed') + geom_area(alpha = 0.1) + ggtitle("ROC Curve w/ AUC = %s" % str(roc_auc))
roc_score = np.asarray([0.9182,0.9799]) ##Modify values to match roc graph values
results = DataFrame({"Roc score" : roc_score, "Test Accuracy" : test_accuracy , "Train Accuracy" : train_accuracy} , index = models)
results