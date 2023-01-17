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
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import metrics
data3 = pd.read_csv('/kaggle/input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv')
data2 = pd.read_csv('/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')
data2.head()

def result (y_pred,y_test):
    print('Confusion_matrix')
    print(metrics.confusion_matrix(y_test,y_pred))
    print('Accuracy')
    print(metrics.accuracy_score(y_test,y_pred))
x= data2.iloc[:,0:6]
y = data2.iloc[:,6]
encoder = LabelEncoder()
y = encoder.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0 )

from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier()
classifier_dt.fit(x_train, y_train)
y_pred_dt = classifier_dt.predict(x_test)
result(y_pred_dt,y_test)
from sklearn.neighbors import KNeighborsClassifier
classifier_KNN = KNeighborsClassifier()
classifier_KNN.fit(x_train, y_train)
y_pred_KNN = classifier_KNN.predict(x_test)
result(y_pred_KNN,y_test)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled = train_test_split(x_scaled, y, test_size = 0.2, random_state = 0 )
classifier_dt_s = DecisionTreeClassifier()
classifier_dt_s.fit(x_train, y_train)
y_pred_dt_s = classifier_dt.predict(x_test)
result(y_pred_dt_s,y_test)
classifier_KNN_s = KNeighborsClassifier()
classifier_KNN_s.fit(x_train, y_train)
y_pred_KNN_s = classifier_KNN.predict(x_test)
result(y_pred_KNN_s,y_test)
plt.scatter(data2.iloc[:,4], data2.iloc[:,5])
data2.groupby('class').pelvic_incidence.mean().plot(kind = 'bar')
from sklearn.ensemble import RandomForestClassifier
forest_model = RandomForestClassifier()
forest_model.fit(x_train_scaled,y_train_scaled)
y_pred_rf = forest_model.predict(x_test_scaled)
result(y_pred_rf,y_test_scaled)
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
classifer_svc = SVC()
voting_model = VotingClassifier(estimators = [('random_forest',forest_model),('KNN',classifier_KNN_s),('svm',classifer_svc)],voting = 'hard')
voting_model.fit(x_test_scaled,y_test_scaled)
y_pred_voting = voting_model.predict(x_test_scaled)
result(y_pred_voting,y_test_scaled)
metrics.precision_score(y_test_scaled, y_pred_voting)
metrics.recall_score(y_test_scaled,y_pred_voting)
print(metrics.classification_report(y_test_scaled, y_pred_voting))
