# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import recall_score

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler,RobustScaler, PowerTransformer, QuantileTransformer, Normalizer, FunctionTransformer

sns.set()

%matplotlib inline





df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

print(df.shape)

df.head()
df.info()
df.describe()
class_names = {0:'Not Fraud', 1:'Fraud'}

print(df.Class.value_counts().rename(index = class_names))
fig = plt.figure(figsize = (15, 12))



plt.subplot(5, 6, 1) ; plt.plot(df.V1) ; plt.subplot(5, 6, 15) ; plt.plot(df.V15)

plt.subplot(5, 6, 2) ; plt.plot(df.V2) ; plt.subplot(5, 6, 16) ; plt.plot(df.V16)

plt.subplot(5, 6, 3) ; plt.plot(df.V3) ; plt.subplot(5, 6, 17) ; plt.plot(df.V17)

plt.subplot(5, 6, 4) ; plt.plot(df.V4) ; plt.subplot(5, 6, 18) ; plt.plot(df.V18)

plt.subplot(5, 6, 5) ; plt.plot(df.V5) ; plt.subplot(5, 6, 19) ; plt.plot(df.V19)

plt.subplot(5, 6, 6) ; plt.plot(df.V6) ; plt.subplot(5, 6, 20) ; plt.plot(df.V20)

plt.subplot(5, 6, 7) ; plt.plot(df.V7) ; plt.subplot(5, 6, 21) ; plt.plot(df.V21)

plt.subplot(5, 6, 8) ; plt.plot(df.V8) ; plt.subplot(5, 6, 22) ; plt.plot(df.V22)

plt.subplot(5, 6, 9) ; plt.plot(df.V9) ; plt.subplot(5, 6, 23) ; plt.plot(df.V23)

plt.subplot(5, 6, 10) ; plt.plot(df.V10) ; plt.subplot(5, 6, 24) ; plt.plot(df.V24)

plt.subplot(5, 6, 11) ; plt.plot(df.V11) ; plt.subplot(5, 6, 25) ; plt.plot(df.V25)

plt.subplot(5, 6, 12) ; plt.plot(df.V12) ; plt.subplot(5, 6, 26) ; plt.plot(df.V26)

plt.subplot(5, 6, 13) ; plt.plot(df.V13) ; plt.subplot(5, 6, 27) ; plt.plot(df.V27)

plt.subplot(5, 6, 14) ; plt.plot(df.V14) ; plt.subplot(5, 6, 28) ; plt.plot(df.V28)

plt.subplot(5, 6, 29) ; plt.plot(df.Amount)

plt.show()
from sklearn.model_selection import train_test_split
feature_names = df.iloc[:, 1:30].columns

target = df.iloc[:1, 30: ].columns

print(feature_names)

print(target)
data_features = df[feature_names]

data_target = df[target]
X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.70, test_size=0.30, random_state=1)

print("Length of X_train is: {X_train}".format(X_train = len(X_train)))

print("Length of X_test is: {X_test}".format(X_test = len(X_test)))

print("Length of y_train is: {y_train}".format(y_train = len(y_train)))

print("Length of y_test is: {y_test}".format(y_test = len(y_test)))
scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
from sklearn.svm import SVC



svm = SVC()

svm.fit(X_train_scaled, y_train)



print("Train set score: {:.2f}".format(svm.score(X_train_scaled, y_train)))

print("Test set score: {:.2f}".format(svm.score(X_test_scaled, y_test)))
df['new_amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)

df.head()
train_set_percentage = 0.5

###################################################

# select 30% of the entire class 1 (fraudulent transactions) data in order to train the model 

fraud_series = df[df['Class'] == 1]

idx = fraud_series.index.values

np.random.shuffle(idx)

fraud_series.drop(idx[:int(idx.shape[0]*train_set_percentage)], inplace=True)

df.drop(fraud_series.index.values, inplace=True)

###################################################
###################################################

# normal dataset with the same size of the fraud_series (training dataset)

normal_series = df[df['Class'] == 0] 

idx = normal_series.index.values

np.random.shuffle(idx)

normal_series.drop(idx[fraud_series.shape[0]:], inplace=True)

df.drop(normal_series.index.values, inplace=True)

###################################################
# build the training dataset

new_dataset = pd.concat([normal_series, fraud_series])

new_dataset.reset_index(inplace=True, drop=True)

y = new_dataset['Class'].values.reshape(-1, 1)

new_dataset.drop(['Class'], axis=1, inplace=True)
X = new_dataset.values
attr={'C': [0.1, 1, 2, 5, 10, 25, 50, 100],

      'gamma': [1e-1, 1e-2, 1e-3]

     }



X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), test_size=0.3, random_state=10)



model = SVC()

classif = GridSearchCV(model, attr, cv=5)

classif.fit(X_train, y_train)

y_pred = classif.predict(X_test)

print('Accuracy: ',accuracy_score(y_pred, y_test))
#y_all = df['Class'].values.reshape(-1, 1)

#df.drop(['Class'], axis=1, inplace=True)

#X_all = df

#y_pred_all = classif.predict(X_all)
#print(recall_score(y_all, y_pred_all))
# Fitting K Nearest Neighbor Classification to the Training Set

from sklearn.neighbors import KNeighborsClassifier

from sklearn import neighbors

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values= np.nan, strategy= 'mean')

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier.fit(X_train, y_train)
# Predicting the Test Set results

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
X_test
y_test
from sklearn.model_selection import cross_val_score
#create a new KNN model

knn_cv = KNeighborsClassifier(n_neighbors=3)
#train model with cv of 5 

cv_scores = cross_val_score(knn_cv, X, y, cv=5)
#print each cv score (accuracy) and average them

print(cv_scores)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

X_train, X_validate, y_train, y_validate = train_test_split(X_train,y_train,test_size=0.25, random_state=42)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,y_train)
print("X_train: ",X_train.shape)

print("y_train: ",y_train.shape)

print("X_test: ",X_test.shape)

print("y_test: ",y_test.shape)

print("X_validate: ",X_test.shape)

print("y_validate: ",y_test.shape)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,y_train)
y_pred = nb.predict(X_test)

print("Tahmin Edilen Deger: ",y_pred)
from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

print("confusion_matrix: ",confusion_matrix(y_test, y_pred))

print()

print("accuracy: ",metrics.accuracy_score(y_test,y_pred))

print("f1 Score(macro): ",f1_score(y_test, y_pred, average='macro'))

print("f1 Score(micro): ",f1_score(y_test, y_pred, average='micro'))

print("f1 Score(weighted): ",f1_score(y_test, y_pred, average='weighted'))

print("precision(macro): ",precision_score(y_test, y_pred, average='macro'))

print("precision(micro): ",precision_score(y_test, y_pred, average='micro'))

print("precision (weighted): ",precision_score(y_test, y_pred, average='weighted'))