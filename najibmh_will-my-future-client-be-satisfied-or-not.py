import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import scipy.optimize as opt

from sklearn import preprocessing

import pylab as pl

df = pd.read_csv('../input/satisfaction_v2.csv')

df.head()
df.info()
df.columns
df.describe().round(2)
df['satisfaction_v2'].unique()
sns.countplot(x = 'satisfaction_v2', data = df, order = df['satisfaction_v2'].value_counts().index)

plt.xticks(rotation=0)
pd.crosstab(df['satisfaction_v2'], df['Age'], dropna=True, normalize='columns')
k = pd.crosstab(df['Age'],df['satisfaction_v2'], dropna=True, normalize='columns')

k.plot.bar(stacked=False, figsize=(25, 5))

plt.show()
df['Customer Type'].unique()
sns.countplot(x = 'Customer Type', data = df, order = df['Customer Type'].value_counts().index)

plt.xticks(rotation=0)
pd.crosstab(df['satisfaction_v2'], df['Customer Type'], dropna=True, normalize='columns')
k = pd.crosstab(df['Customer Type'],df['satisfaction_v2'], dropna=True, normalize='columns')

k.plot.bar(stacked=False)

plt.show()
k = pd.crosstab(df['Age'],df['Customer Type'], dropna=True, normalize='columns')

k.plot.bar(stacked=False, figsize=(25, 5))

plt.show()
df['Type of Travel'].unique()
sns.countplot(x = 'Type of Travel', data = df, order = df['Type of Travel'].value_counts().index)

plt.xticks(rotation=0)
pd.crosstab(df['satisfaction_v2'], df['Type of Travel'], dropna=True, normalize='columns')
k = pd.crosstab(df['Type of Travel'],df['satisfaction_v2'], dropna=True, normalize='columns')

k.plot.bar(stacked=False)

plt.show()
k = pd.crosstab(df['Age'],df['Type of Travel'], dropna=True, normalize='columns')

k.plot.bar(stacked=False, figsize=(25, 5))

plt.show()
df['Class'].unique()
sns.countplot(x = 'Class', data = df, order = df['Class'].value_counts().index)

plt.xticks(rotation=0)
pd.crosstab(df['satisfaction_v2'], df['Class'], dropna=True, normalize='columns')
k = pd.crosstab(df['Class'],df['satisfaction_v2'], dropna=True, normalize='columns')

k.plot.bar(stacked=False)

plt.show()
k = pd.crosstab(df['Age'],df['Class'], dropna=True, normalize='columns')

k.plot.bar(stacked=False, figsize=(25, 5))

plt.show()
df['Gender'].unique()
sns.countplot(x = 'Gender', data = df, order = df['Gender'].value_counts().index)

plt.xticks(rotation=0)
pd.crosstab(df['satisfaction_v2'], df['Gender'], dropna=True, normalize='columns')
k = pd.crosstab(df['Gender'],df['satisfaction_v2'], dropna=True, normalize='columns')

k.plot.bar(stacked=False)

plt.show()
k = pd.crosstab(df['Age'],df['Gender'], dropna=True, normalize='columns')

k.plot.bar(stacked=False, figsize=(25, 5))

plt.show()
plt.figure(figsize=(20,10))

sns.countplot(x = 'Age', data = df, order = df['Age'].value_counts().index)

plt.xticks(rotation=90)
df.Gender[df.Gender == 'Male'] = 1

df.Gender[df.Gender == 'Female'] = 0



df.satisfaction_v2[df.satisfaction_v2 == 'satisfied'] = 1

df.satisfaction_v2[df.satisfaction_v2 == 'neutral or dissatisfied'] = 0



df['Type of Travel'][df['Type of Travel'] == 'Personal Travel'] = 1

df['Type of Travel'][df['Type of Travel'] == 'Business travel'] = 0



df['Customer Type'][df['Customer Type'] == 'Loyal Customer'] = 1

df['Customer Type'][df['Customer Type'] == 'disloyal Customer'] = 0



df.Class[df.Class == 'Eco'] = 1

df.Class[df.Class == 'Business'] = 2

df.Class[df.Class == 'Eco Plus'] = 3



pd.options.mode.chained_assignment = None



df = df.apply(pd.to_numeric, errors='coerce')
df.head(1000)
sat_df = df[['Age','Gender','Flight Distance','Customer Type','Class','satisfaction_v2']]

sat_df['satisfaction_v2'] = sat_df['satisfaction_v2'].astype('int')

sat_df.head()
sat_df.shape
X = np.asarray(sat_df[['Age','Gender','Flight Distance','Customer Type','Class']])

X[0:5]
y = np.asarray(sat_df['satisfaction_v2'])

y[0:5]
from sklearn import preprocessing

X = preprocessing.StandardScaler().fit(X).transform(X)

X[0:5]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4 )

print('Train set:', X_train.shape, y_train.shape)

print('Test set:', X_test.shape, y_test.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

LR
yhat = LR.predict(X_test)

yhat
yhat_prob = LR.predict_proba(X_test)

yhat_prob
# Jaccard index

from sklearn.metrics import jaccard_similarity_score

jaccard_similarity_score(y_test, yhat)
# Confusion matrix

from sklearn.metrics import classification_report, confusion_matrix

import itertools

def plot_confusion_matrix(cm, classes, 

                         normalize=False,

                         title='Confusion matrix',

                         cmap=plt.cm.Blues):

    

    if normalize : 

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

    plt.yticks(tick_marks,classes)

    

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i,j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i,j] > thresh else "black")

    

    plt.tight_layout()

    plt.ylabel("True label")

    plt.xlabel("Predicted label")



print(confusion_matrix(y_test, yhat, labels=[1,0]))
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])

np.set_printoptions(precision=2)



plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False, title='Confusion matrix')
print(classification_report(y_test, yhat))
# Log Loss

from sklearn.metrics import log_loss

log_loss(y_test, yhat_prob)
a = 19

k =  0

ds = 2051

ct = 1

cs = 2
from sklearn.preprocessing import Normalizer

# g = np.reshape(1,-1)

g= np.array([[a,k,ds,ct,cs]])

# g = preprocessing.StandardScaler().fit(g).transform(g)

# g = preprocessing.StandardScaler().fit(g.reshape(1, -1)).transform(g.reshape(1, -1))

Z = preprocessing.Normalizer().fit(g.reshape(1, -1))

Z
B = Z.transform(g)

B
y_prob = LR.predict(B)

y_prob
y_prob = LR.predict_proba(B)

y_prob
