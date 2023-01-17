# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
#The data has a column called  'Unnamed: 32' with nothing but NaN values.
df.drop(['Unnamed: 32'], axis = 1, inplace = True)
df.head()
print(df.isnull().sum().sum())
print(df.shape)
df['diagnosis'].replace({'B' : 0, 'M' : 1}, inplace = True)
df['diagnosis'].value_counts()
# Allocating 69 rows to the test data and rest to the train data. 
df_train = df.iloc[:500]
df_test = df.iloc[500:]

# Re-indexing of the test data
df_test.reset_index(inplace = True)
df_test.drop('index', axis = 1,inplace = True)
df_corr = df_train.corr()
df_corr = df_corr.where(abs(df_corr['diagnosis']) > 0.5)
df_corr.dropna(inplace=True)
cols_to_drop = df_corr.columns.drop(df_corr.index.drop('diagnosis'))
df_corr.drop(columns = cols_to_drop, inplace = True)
print("Important features  are - ", df_corr.columns.values)
#defining the features
features = ['radius_mean', 'concave points_mean', 'radius_se', 'radius_worst', 'concave points_worst']
from sklearn.preprocessing import StandardScaler
X = df_train[features]
y = df_train.diagnosis
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test = df_test[features]
y_test = df_test.diagnosis
scale = StandardScaler()
X_test_scaled = scale.fit_transform(X_test)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

#developing baseline model
RF_model = RandomForestClassifier(n_estimators = 1000, max_depth = 10, random_state = 7, n_jobs = -1)
scores = cross_val_score(RF_model, X_scaled, y, cv = 5, scoring = 'f1')
scores.mean()*100
#hyperparamter tuning
for n_est in [200,400,600,800]:
    for m_d in [5,6,7,8,12]:
        RF_model = RandomForestClassifier(n_estimators = n_est, max_depth = m_d, random_state = 7, n_jobs = -1)
        scores = cross_val_score(RF_model, X_scaled, y, cv = 5, scoring = 'f1')
        print('estimators = ',n_est," max depth = ",m_d," score = ",scores.mean()*100)
#developing the optimized model
RF_model = RandomForestClassifier(n_estimators = 600, max_depth = 10, random_state = 7, n_jobs = -1)
RF_model.fit(X,y)
scores = cross_val_score(RF_model, X_scaled, y, cv = 5, scoring = 'f1')
scores.mean()*100
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

#developing baseline model
knn_model = KNeighborsClassifier(n_neighbors = 5, leaf_size = 20, n_jobs = -1)
scores = cross_val_score(knn_model, X_scaled, y, cv = 5, scoring = 'f1')
scores.mean()*100
#hyperparameter tuning
for ne in [4,5,6,7,8,9,10]:
    for ls in [10,20,30,40,50,60,80]:
        knn_model = KNeighborsClassifier(n_neighbors = ne, leaf_size = ls, n_jobs = -1)
        scores = cross_val_score(knn_model, X_scaled, y, cv = 5, scoring = 'f1')
        print('n_neighbors = ', ne, ' leaf_size = ', ls, 'score = ', scores.mean()*100)            
#developing the optimized model
knn_model = KNeighborsClassifier(n_neighbors = 5, leaf_size = 20, n_jobs = -1)
scores = cross_val_score(knn_model, X_scaled, y, cv = 5, scoring = 'f1')
scores.mean()*100
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

#developing the baseline model and tuning it
for c in [0.1,0.2,0.3,0.4,0.5,1]:
    LoReg_model = LogisticRegression(C = c, random_state = 3, n_jobs = -1)
    scores = cross_val_score(LoReg_model, X_scaled ,y , cv=5, scoring='f1')
    print(scores.mean()*100)
#developing the optimized model
LoReg_model = LogisticRegression(C = 0.5, random_state = 3, n_jobs = -1)
scores = cross_val_score(LoReg_model, X_scaled ,y , cv=5, scoring='f1')
scores.mean()*100
#Fit the final model on the train data
final_model = RandomForestClassifier(n_estimators = 600, max_depth = 10, random_state = 7, n_jobs = -1)
final_model.fit(X_scaled, y);
#Scoring on test data
y_true = y_test
y_pred = final_model.predict(X_test_scaled)
#defining labels for prognosis
labels = ['Benign', 'Malignant']

#using yellowbricks Classification Report
from yellowbrick.classifier import ClassificationReport

report = ClassificationReport(final_model, size=(480,240), classes = labels, cmap = 'PuBu')
report.score(X_test_scaled, y_test)
report.show()  
#using yellowbrick class prediction error
from yellowbrick.classifier import ClassPredictionError

error = ClassPredictionError(final_model, size=(540,360), classes = labels)
error.score(X_test_scaled, y_test)
error.show()