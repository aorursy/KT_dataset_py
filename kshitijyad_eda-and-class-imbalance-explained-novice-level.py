from sklearn.preprocessing import binarize

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import confusion_matrix,auc,roc_auc_score,recall_score,classification_report,precision_recall_curve, roc_curve

from subprocess import check_output

import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import itertools
data=pd.read_csv("../input/creditcard.csv")

data.head()
plt.rcParams['figure.figsize']=(10,10)

sns.heatmap(data.corr())

sns.plt.show()
data.isnull().any().sum()
sns.countplot(data['Class'])

sns.plt.show()

print('Percent of fraud transaction: ',len(data[data['Class']==1])/len(data['Class'])*100,"%")
sns.distplot(data.Amount)

sns.plt.show()

sns.distplot(data[data.Class==1].Amount)

sns.plt.show()

population = data[data.Class == 0].Amount

sample = data[data.Class == 1].Amount

sampleMean = sample.mean()

populationStd = population.std()

populationMean = population.mean()

z_score = (sampleMean - populationMean) / (populationStd / sample.size ** 0.5)

z_score
train= data.drop(['Time'], axis=1)



X= train.drop('Class',axis=1)

y= train['Class']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=15)

    plt.yticks(tick_marks, classes, rotation=15)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #print("Normalized confusion matrix")

    

        #print(cm)

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
class_set = [0, 1]

lr = LogisticRegression()

lr.fit(X_train, y_train.values.ravel())

y_pred = lr.predict(X_test)

cnf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

np.set_printoptions(precision=2)

print ("Confusion matrix undersampled")

plt.rcParams['figure.figsize']=(4,4)

plot_confusion_matrix(cm=cnf_matrix, classes=class_set)

plt.show()

print('cr:', classification_report(y_test,y_pred))
y_predprob = lr.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test,y_predprob)

roc_auc = auc(fpr,tpr)

plt.plot(fpr,tpr)

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('ROC curve for fraud classifier')

plt.grid(True)

plt.show()

roc_auc_score(y_test, y_predprob)
fraud_count = len(train[train.Class == 1])



fraud_indices = train[train.Class == 1].index

normal_indices = train[train.Class == 0].index



r_normal_indices = np.random.choice(normal_indices, fraud_count, replace = False)



undersample_indices = np.concatenate([fraud_indices,r_normal_indices])

undersample_train = train.iloc[undersample_indices,:]



X_undersample = undersample_train.drop('Class',axis=1)

y_undersample = undersample_train['Class']



X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X_undersample,y_undersample,test_size = 0.3,random_state = 0)
class_set = [0, 1]

lr_und = LogisticRegression()

lr_und.fit(X_train_u, y_train_u.values.ravel())

y_pred_u = lr_und.predict(X_test_u)

cnf_matrix_und = confusion_matrix(y_true=y_test_u, y_pred=y_pred_u)

np.set_printoptions(precision=2)

print ("Confusion matrix undersampled")

plt.rcParams['figure.figsize']=(4,4)

plot_confusion_matrix(cm=cnf_matrix_und, classes=class_set)

plt.show()

print('cr:', classification_report(y_test_u,y_pred_u))
y_predprob_u = lr.predict_proba(X_test_u)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test_u,y_predprob_u)

roc_auc = auc(fpr,tpr)

plt.plot(fpr,tpr)

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('ROC curve for fraud classifier')

plt.grid(True)

plt.show()

roc_auc_score(y_test_u, y_predprob_u)