# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
for df in ("../input"):
    df=pd.read_csv("../input/preprocessing.csv").fillna(0)

# Any results you write to the current directory are saved as output.
df.head()
df.info()
df.corr()
df['EMOTION'].unique()
plt.figure(figsize = (10, 8))
sns.countplot(df['EMOTION'])
plt.show()
df['EMOTION'].value_counts()
df.isnull().sum().sum() #no missing values
#split into features and labels sets
X = df.drop(['EMOTION','ID'], axis = 1) #features
y = df['EMOTION'] #labels


X.head()
X.info()
print("Total number of labels: {}".format(df.shape[0]))
target = df.ID
X.dtypes.sample(104)

one_hot_encoded_training_predictors = pd.get_dummies(X)
one_hot_encoded_test_predictors = pd.get_dummies(y)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,join='left', axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
m1 = LogisticRegression()
m1.fit(X_train, y_train)
pred1 = m1.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, pred1))
labels = ['ANGRY','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE']
cm1 = pd.DataFrame(confusion_matrix(y_test, pred1), index = labels, columns = labels)
plt.figure(figsize = (10, 8))
sns.heatmap(cm1, annot = True, cbar = False, fmt = 'g')
plt.ylabel('Actual values')
plt.xlabel('Predicted values')
plt.show()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
grid = {'n_estimators': [10, 50, 100, 300]}

m2 = GridSearchCV(RandomForestClassifier(), grid)
m2.fit(X_train, y_train)
m2.best_params_  #I got n_estimators = 300
pred2 = m2.predict(X_test)
print(classification_report(y_test, pred2)) #much better, but recall is still low
cm2 = pd.DataFrame(confusion_matrix(y_test, pred2), index = labels, columns = labels)

plt.figure(figsize = (10, 8))
sns.heatmap(cm2, annot = True, cbar = False, fmt = 'g')
plt.ylabel('Actual values')
plt.xlabel('Predicted values')
plt.show()
from sklearn.ensemble import GradientBoostingClassifier
grid = {
    'learning_rate': [0.03, 0.1, 0.5], 
    'n_estimators': [100, 300], 
    'max_depth': [1, 3, 9]
}

m3 = GridSearchCV(GradientBoostingClassifier(), grid, verbose = 2)
m3.fit(X_train, y_train) 
m3.best_params_
pred3 = m3.predict(X_test)

print(classification_report(y_test, pred3))
cm3 = pd.DataFrame(confusion_matrix(y_test, pred3), index = labels, columns = labels)

plt.figure(figsize = (10, 8))
sns.heatmap(cm3, annot = True, cbar = False, fmt = 'g')
plt.ylabel('Actual values')
plt.xlabel('Predicted values')
plt.show()
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler()
scaler.fit(X_train)
X_sc_train = scaler.transform(X_train)
X_sc_test = scaler.transform(X_test)
pca = PCA(n_components=104)
pca.fit(X_train)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
NCOMPONENTS = 104

pca = PCA(n_components=NCOMPONENTS)
X_pca_train = pca.fit_transform(X_sc_train)
X_pca_test = pca.transform(X_sc_test)
pca_std = np.std(X_pca_train)

print(X_sc_train.shape)
print(X_pca_test.shape)
inv_pca = pca.inverse_transform(X_pca_train)
inv_sc = scaler.inverse_transform(inv_pca)
from sklearn.svm import SVC
grid = {
    'C': [1,5,50],
    'gamma': [0.05,0.1,0.5,1,5]
}

m5 = GridSearchCV(SVC(), grid)
m5.fit(X_train, y_train)

m5.best_params_ #I got C = 1, gamma = 0.05
pred5 = m5.predict(X_test)

print(classification_report(y_test, pred5))
cm5 = pd.DataFrame(confusion_matrix(y_test, pred5), index = labels, columns = labels)

plt.figure(figsize = (10, 8))
sns.heatmap(cm5, annot = True, cbar = False, fmt = 'g')
plt.ylabel('Actual values')
plt.xlabel('Predicted values')
plt.show()
