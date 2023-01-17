# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import zscore
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC 
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import time
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
all_data = "/kaggle/input/pump-sensor-data/sensor.csv"
data =  pd.read_csv(all_data)
data.describe()
data.info()
# deleting column named as sensor 15.
data = data.drop('sensor_15', 1)
data = data.drop('Unnamed: 0', 1)
data.shape
#creating new columns named as data and time from timestamp and deleting the column timestamp.
data['date'] = data['timestamp'].apply(lambda x: x.split(' ')[0])
data['time'] = data['timestamp'].apply(lambda x: x.split(' ')[1])
data = data.drop(['timestamp'], 1)
# imputting missing values with median of each column
data_imputed = data.fillna(data.median())
#checking out histogram for outlier data
data_imputed.hist(figsize=(15,15))
#removing outliers using zscore
z_scores = zscore(data_imputed.iloc[:,:51])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
data_outlir = data_imputed[filtered_entries]
# how much data has been lost after removing outliers
((220320 - 165201)/220320) * 100
data_outlir['machine_status'].value_counts()
# standarizing the dataset using minmax scalar
scaler = MinMaxScaler()
data_std = scaler.fit_transform(data_outlir.iloc[:,:51])
data_std = pd.DataFrame(data_std)
# checking for variance of each feature
data_va = data_std.var(axis= 0)
data_vas = data_va.sort_values(ascending=False)
y = data_vas.values 
x = range(len(y))
plt.figure(figsize = (20,20))
plt.plot(x, y)
plt.show()
corrmatrix = data_std.corr()
# doing multicollinearity test
def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (abs(corr_matrix.iloc[i, j]) >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

    return dataset
le = LabelEncoder()
Y = le.fit_transform(data_outlir['machine_status'])
X = correlation(data_std, 0.7)
X
for i in data_std.columns:
    plt.scatter( data_outlir['machine_status'] , data_std[i] )
    plt.xlabel('machine_status')
    plt.ylabel(i)
    plt.show()
plt.scatter( data_outlir['time'] , data_outlir['machine_status'] )
plt.xlabel('time')
plt.ylabel('machine status')
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
knn = KNeighborsClassifier(n_neighbors=15)
clf = knn.fit(X_train, y_train)
y_pred = clf.predict(X_test)
roc_auc_score(y_test, y_pred)
y_pred_proba = clf.predict_proba(X_test)
#applying logistic regression
lr = LogisticRegression(C = 0.2)
clf1 = lr.fit(X_train, y_train)
y_pred1 = clf1.predict(X_test)
roc_auc_score(y_test, y_pred1)
#applying naive bayes
clf2 = GaussianNB().fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)
roc_auc_score(y_test, y_pred2)
#applying decision tree
clf3 = tree.DecisionTreeClassifier().fit(X_train, y_train)
y_pred3 = clf3.predict(X_test)
roc_auc_score(y_test, y_pred3)
#applying random forest
clf4 = RandomForestClassifier(max_depth=5, random_state=0).fit(X_train, y_train)
y_pred4 = clf4.predict(X_test)
roc_auc_score(y_test, y_pred4)
#applying support vector machine
clf5 = SVC(gamma='auto').fit(X_train, y_train)
y_pred5 = clf5.predict(X_test)
roc_auc_score(y_test, y_pred5)