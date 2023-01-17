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

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

mpl.style.use('ggplot')
#read the data

pd.set_option('display.max_columns', None)

df = pd.read_csv("../input/seattle-sdot-collisions-data/Collisions.csv")

print("Data read into dataframe!") 
df.head()
#shape of the data

df.shape
#info about data

df.info()
#columns

df.columns.values
df.dtypes.value_counts().plot(kind='bar')
Null_values = df.isnull().sum()

Null_values[0:40]
df1= df.drop(['X','Y','INTKEY','COLDETKEY','REPORTNO','LOCATION','EXCEPTRSNCODE','EXCEPTRSNDESC','SEVERITYDESC','INJURIES','SERIOUSINJURIES','FATALITIES','INCDATE','INCDTTM','SDOT_COLCODE','SDOT_COLDESC','INATTENTIONIND','UNDERINFL','PEDROWNOTGRNT','SDOTCOLNUM','SPEEDING','ST_COLDESC','SEGLANEKEY','CROSSWALKKEY','HITPARKEDCAR'],axis=1)
df1.dtypes.value_counts().plot(kind='bar')
sns.countplot(df1['STATUS'], data=df1)
sns.countplot(df1['ADDRTYPE'], data=df1)
df = df1
df.shape
Null_values = df.isnull().sum()

Null_values[0:15]
df['ST_COLCODE'] = df['ST_COLCODE'].fillna(0)

df['SEVERITYCODE'] = df['SEVERITYCODE'].fillna(0)



df['ADDRTYPE'] = df['ADDRTYPE'].fillna(0)

df['ADDRTYPE'] = df['ADDRTYPE'].replace(0,'others')



df['WEATHER'] = df['WEATHER'].fillna(0)

df['WEATHER'] = df['WEATHER'].replace(0,'others')



df['ROADCOND'] = df['ROADCOND'].fillna(0)

df['ROADCOND'] = df['ROADCOND'].replace(0,'others')



df['LIGHTCOND'] = df['LIGHTCOND'].fillna(0)

df['LIGHTCOND'] = df['LIGHTCOND'].replace(0,'others')



df['COLLISIONTYPE'] = df['COLLISIONTYPE'].fillna(0)

df['COLLISIONTYPE'] = df['COLLISIONTYPE'].replace(0,'others')



df['JUNCTIONTYPE'] = df['JUNCTIONTYPE'].fillna(0)

df['JUNCTIONTYPE'] = df['JUNCTIONTYPE'].replace(0,'others')
Null_values = df.isnull().sum()

Null_values[0:15]
#addrtype

from sklearn import preprocessing

addrtype = preprocessing.LabelEncoder()

addrtype.fit(['Intersection','Block','Alley','others'])

df['ADDRTYPE'] = addrtype.transform(df['ADDRTYPE'])
#status

from sklearn import preprocessing

status = preprocessing.LabelEncoder()

status.fit(['Unmatched','Matched'])

df['STATUS'] = status.transform(df['STATUS'])
#weather

from sklearn import preprocessing

weathercond = preprocessing.LabelEncoder()

weathercond.fit(df['WEATHER'])

df['WEATHER'] = weathercond.transform(df['WEATHER'])
#Road

from sklearn import preprocessing

roadcond = preprocessing.LabelEncoder()

roadcond.fit(df['ROADCOND'])

df['ROADCOND'] = roadcond.transform(df['ROADCOND'])
#light

from sklearn import preprocessing

light = preprocessing.LabelEncoder()

light.fit(df['LIGHTCOND'])

df['LIGHTCOND'] = light.transform(df['LIGHTCOND'])
#collision type

from sklearn import preprocessing

coll = preprocessing.LabelEncoder()

coll.fit(df['COLLISIONTYPE'])

df['COLLISIONTYPE'] = coll.transform(df['COLLISIONTYPE'])
#junction type

from sklearn import preprocessing

jun = preprocessing.LabelEncoder()

jun.fit(df['JUNCTIONTYPE'])

df['JUNCTIONTYPE'] = jun.transform(df['JUNCTIONTYPE'])
df.head()
sns.countplot(df['SEVERITYCODE'], data=df)
fig, axes = plt.subplots(2, 2, sharex=True, figsize=(10,5))

sns.boxplot(ax=axes[0, 0], data=df, x='SEVERITYCODE', y='ADDRTYPE')

sns.boxplot(ax=axes[0, 1], data=df, x='SEVERITYCODE', y='STATUS')

sns.boxplot(ax=axes[1, 0], data=df, x='SEVERITYCODE', y='WEATHER')

sns.boxplot(ax=axes[1, 1], data=df, x='SEVERITYCODE', y='ROADCOND')
sns.pairplot(df[['SEVERITYCODE','ADDRTYPE','WEATHER','ROADCOND']])
sns.heatmap(df.corr(),cmap="Blues", linewidth=0.3, cbar_kws={"shrink": .8})
df[['ADDRTYPE', 'SEVERITYCODE','COLLISIONTYPE','JUNCTIONTYPE', 'WEATHER', 'ROADCOND', 'LIGHTCOND']].plot(kind='hist', figsize=(10,6), alpha=0.5, stacked=False)
#import libraries

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score, r2_score, confusion_matrix, classification_report
df2 = pd.read_csv('../input/collision-data/new_data_collisions.csv')
#split the data



X = df2[['OBJECTID','INCKEY','STATUS','ADDRTYPE','COLLISIONTYPE','PERSONCOUNT','PEDCOUNT','PEDCYLCOUNT','VEHCOUNT','JUNCTIONTYPE', 'WEATHER', 'ROADCOND', 'LIGHTCOND']]

y = df2['SEVERITYCODE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("Train set size")

print(X_train.shape)

print(y_train.shape)

print('')

print("Test set size")

print(X_test.shape)

print(y_test.shape)
#K Nearest Neighbors

k=17

knn = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

knn
knn_pred = knn.predict(X_test)

knn_pred
print('Score:',accuracy_score(knn_pred, y_test))

print('F1-Score:',f1_score(knn_pred, y_test))

print('')

print('Confusion Martix:')

print(confusion_matrix(y_test, knn_pred))

print('')

print('Classification Report:')

print (classification_report(y_test, knn_pred))


Ks = 20

mean_acc = np.zeros((Ks-1))

std_acc = np.zeros((Ks-1))

ConfustionMx = [];

for n in range(1,Ks):

    

    #Train Model and Predict  

    knn = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)

    pred=knn.predict(X_test)

    mean_acc[n-1] = accuracy_score(y_test, pred)



    

    std_acc[n-1]=np.std(pred==y_test)/np.sqrt(pred.shape[0])



mean_acc
plt.plot(range(1,Ks),mean_acc,'g')

plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)

plt.legend(('Accuracy ', '+/- 3xstd'))

plt.ylabel('Accuracy ')

plt.xlabel('Number of Nabors (K)')

plt.tight_layout()

plt.show()
#Decision Tree

tree = DecisionTreeClassifier(criterion="gini", max_depth = 4).fit(X_train, y_train)

tree
tree_pred = tree.predict(X_test)

tree_pred
print('Score:',accuracy_score(tree_pred, y_test))

print('F1-Score:',f1_score(tree_pred, y_test))

print('')

print('Confusion Martix:')

print(confusion_matrix(y_test, tree_pred))

print('')

print('Classification Report:')

print (classification_report(y_test, tree_pred))
#Logistic Regression

lr = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

lr
lr_pred = lr.predict(X_test)

lr_pred
print('Score:',accuracy_score(lr_pred, y_test))

print('F1-Score:',f1_score(lr_pred, y_test))

print('')

print('Confusion Martix:')

print(confusion_matrix(y_test, lr_pred))

print('')

print('Classification Report:')

print (classification_report(y_test, lr_pred))
#Navie Bayes

nb = GaussianNB(priors=None, var_smoothing=1e-09).fit(X_train,y_train)

nb
nb_pred = nb.predict(X_test)

nb_pred
print('Score:',accuracy_score(nb_pred, y_test))

print('F1-Score:',f1_score(nb_pred, y_test))

print('')

print('Confusion Martix:')

print(confusion_matrix(y_test, knn_pred))

print('')

print('Classification Report:')

print (classification_report(y_test, nb_pred))
#Random Forest

rf = RandomForestClassifier(n_estimators=20).fit(X_train,y_train)

rf
rf_pred = rf.predict(X_test)

rf_pred
print('Score:',accuracy_score(rf_pred, y_test))

print('F1-Score:',f1_score(rf_pred, y_test))

print('')

print('Confusion Martix:')

print(confusion_matrix(y_test, knn_pred))

print('')

print('Classification Report:')

print (classification_report(y_test, rf_pred))