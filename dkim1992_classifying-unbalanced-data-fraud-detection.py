# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.









data = pd.read_csv("../input/creditcard.csv")



data.sample(5)
data.info()
import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 

import numpy as np 

%matplotlib inline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
sns.countplot(data.Class)

d = data.Class.value_counts()

print(d)

print('Fraud cases are only {:f}% of all cases.'.format(d[1]*100/len(data)))
sns.kdeplot(data = data[data.Class == 1].Amount, label = 'Fraud', bw=50)

sns.kdeplot(data = data[data.Class == 0].Amount, label = 'Normal', bw=50)

plt.legend();

data.drop("Time", axis = 1, inplace = True)

data.columns
X = data.drop("Class", axis = 1)

y = data.Class



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=333, stratify = y)



print(len(y_train[y_train == 1])/len(y_train))

print(len(y_test[y_test == 1])/len(y_test))
X_train.Amount.shape
scaler = StandardScaler()

scaler.fit(X_train.Amount.reshape(-1, 1))

X_train.Amount = scaler.transform(X_train.Amount.reshape(-1,1))

X_test.Amount = scaler.transform(X_test.Amount.reshape(-1,1))
X_train.Amount.describe(), X_test.Amount.describe()


from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report


clf = LogisticRegression()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)




cm = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(cm, range(2), range(2))

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True)



print(classification_report(y_test,y_pred))


clf = RandomForestClassifier(n_estimators = 10)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


cm = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(cm, range(2), range(2))

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True)



print(classification_report(y_test,y_pred))
from sklearn.utils import resample



X_train_normal = X_train[y_train == 0]

y_train_normal = y_train[y_train == 0]

X_train_fraud = X_train[y_train == 1]

y_train_fraud = y_train[y_train == 1]



X_train_normal, y_train_normal = resample(X_train_normal, y_train_normal, n_samples = len(y_train_fraud), replace = False, random_state = 333)



X_train_undersample = pd.concat([X_train_normal, X_train_fraud], ignore_index=True)

y_train_undersample = pd.concat([y_train_normal, y_train_fraud], ignore_index=True)

print(type(y_train_undersample))

print(y_train_undersample.value_counts())

print(len(X_train_undersample))


clf = LogisticRegression()

clf.fit(X_train_undersample, y_train_undersample)

y_pred = clf.predict(X_test)


cm = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(cm, range(2), range(2))

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True)



print(classification_report(y_test,y_pred))


clf = RandomForestClassifier(n_estimators = 100)

clf.fit(X_train_undersample, y_train_undersample)

y_pred = clf.predict(X_test)





cm = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(cm, range(2), range(2))

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True)



print(classification_report(y_test,y_pred))
X_train_normal = X_train[y_train == 0]

y_train_normal = y_train[y_train == 0]

X_train_fraud = X_train[y_train == 1]

y_train_fraud = y_train[y_train == 1]



X_train_fraud, y_train_fraud = resample(X_train_fraud, y_train_fraud, n_samples = len(y_train_normal), replace = True, random_state = 333)



X_train_oversample = pd.concat([X_train_normal, X_train_fraud], ignore_index=True)

y_train_oversample = pd.concat([y_train_normal, y_train_fraud], ignore_index=True)

print(y_train_oversample.value_counts())

print(len(X_train_oversample))
clf = LogisticRegression()

clf.fit(X_train_oversample, y_train_oversample)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(cm, range(2), range(2))

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True)



print(classification_report(y_test,y_pred))


clf = RandomForestClassifier(n_estimators = 100)

clf.fit(X_train_oversample, y_train_oversample)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(cm, range(2), range(2))

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True)



print(classification_report(y_test,y_pred))
from sklearn.cluster import KMeans



X_train_normal = X_train[y_train == 0]

y_train_normal = y_train[y_train == 0]

X_train_fraud = X_train[y_train == 1]

y_train_fraud = y_train[y_train == 1]



len(y_train_normal), len(y_train_fraud)
n_clusters = len(y_train_fraud)

kmeans = KMeans(n_clusters = n_clusters, random_state = 333).fit(X_train_normal)

X_train_normal = kmeans.cluster_centers_



X_train_normal.shape
X_train_normal = kmeans.cluster_centers_

X_train_normal = pd.DataFrame(X_train_normal, columns = X_train.columns)

X_train_normal.sample(5)
y_train_normal = y_train_normal[:n_clusters]

y_train_normal.shape
X_train_kmeans = pd.concat([X_train_normal, X_train_fraud], ignore_index=True)

y_train_kmeans = pd.concat([y_train_normal, y_train_fraud], ignore_index=True)



print(y_train_kmeans.value_counts())

print(len(X_train_kmeans))



X_train_kmeans.isnull().values.any(), y_train_kmeans.isnull().values.any()
clf = LogisticRegression()

clf.fit(X_train_kmeans, y_train_kmeans)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(cm, range(2), range(2))

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True)



print(classification_report(y_test,y_pred))
clf = RandomForestClassifier(n_estimators = 100)

clf.fit(X_train_kmeans, y_train_kmeans)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(cm, range(2), range(2))

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True)



print(classification_report(y_test,y_pred))
from imblearn.over_sampling import SMOTE

os = SMOTE(random_state=0)
X_train_smote, y_train_smote = os.fit_sample(X_train,y_train)

type(X_train_smote), type(y_train_smote)



X_train_smote = pd.DataFrame(X_train_smote, columns= X_train.columns)

y_train_smote= pd.Series(y_train_smote)
#check the number of each class 

y_train_smote.value_counts()
clf = LogisticRegression()

clf.fit(X_train_smote, y_train_smote)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(cm, range(2), range(2))

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True)



print(classification_report(y_test,y_pred))


clf = RandomForestClassifier(n_estimators = 100)

clf.fit(X_train_smote, y_train_smote)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(cm, range(2), range(2))

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True)



print(classification_report(y_test,y_pred))