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
#from keras.utils import np_utils
#from keras.models import Sequential
#from keras.layers import Dense, LSTM, Dropout
#from keras.callbacks import EarlyStopping
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, mean_absolute_error,accuracy_score, classification_report
kfold = KFold(n_splits=10, random_state=7)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Train = '/kaggle/input/datatraining.txt'
Test = '/kaggle/input/datatest.txt'


import matplotlib.pyplot as plt

parse_dates = ['date']
index_col = 'ObservationID'

Train_df = pd.read_csv(Train, parse_dates=parse_dates, index_col=index_col)
Train_df.index.names = [index_col]
Train_df.head()
Y = np.array(Train_df['Occupancy'].values)
X = np.array(Train_df[['CO2', 'Light', 'Temperature', 'Humidity']])

modelDT = DecisionTreeClassifier()
resultsDT = cross_val_score(modelDT,X,Y,cv=kfold)
print("Decison Tree",resultsDT.mean()*100)

num_trees = 200
max_features = 3
modelRF = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
resultsRF = cross_val_score(modelRF,X,Y,cv=kfold)
print("Random Forest",resultsRF.mean()*100)

modelGB = GradientBoostingClassifier()
resultsGB = cross_val_score(modelGB,X,Y,cv=kfold)
print("GradientBoosting",resultsGB.mean()*100)

modelET = ExtraTreesClassifier()
resultsET = cross_val_score(modelET,X,Y,cv=kfold)
print("Extratree",resultsET.mean()*100)

modelNB = KNeighborsClassifier()
resultsNB = cross_val_score(modelNB,X,Y,cv=kfold)
print("Knearestneighbor",resultsNB.mean()*100)

modelSVM = SVC(gamma='auto')
resultsSVM = cross_val_score(modelSVM, X, Y, cv = kfold)
#clf = svm.SVC(kernel='linear', C=1).fit(X, Y)
print("SupportVectormachine",resultsSVM.mean()*100)

modelLR= LogisticRegression(solver='liblinear')
resultsLR = cross_val_score(modelLR,X,Y,cv=kfold)
print("LogisticRegression",resultsLR.mean()*100)

modelDA = LinearDiscriminantAnalysis()
resultsDA = cross_val_score(modelDA,X,Y,cv=kfold)
print("LinearDiscriminant",resultsDA.mean()*100)

#modelXGB= XGBClassifier(objective ='reg:linear', 
                             #max_depth = 3,
                             #silent = 1,
                             #learning_rate = 0.3,
                             #n_estimators = 200)
#resultsXGB = cross_val_score(modelXGB,X,Y,cv=kfold)
#print("XGBoosting",resultsXGB.mean()*100)
modelLR=modelLR.fit(X, Y)
Test_df = pd.read_csv(Test, parse_dates=parse_dates, index_col=index_col)
Test_df.head()
Y_Test = np.array(Test_df['Occupancy'].values)
X_Test = np.array(Test_df[['CO2', 'Light', 'Temperature', 'Humidity']])
Y_Predict=modelLR.predict(X_Test)
#print("Classification Report: \n", classification_report(Y_Test, Y_Predict))
#print("Confusion Matrix: \n", confusion_matrix(Y_Test, Y_Predict))
accuracy = accuracy_score(Y_Test, Y_Predict)
print("\nAccuracy",accuracy*100)