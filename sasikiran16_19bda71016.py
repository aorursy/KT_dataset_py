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
Train_Data = pd.read_csv("/kaggle/input/bda-2019-ml-test/Train_Mask.csv")
Test_Data = pd.read_csv("/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv")
Sample_Submission = pd.read_csv("/kaggle/input/bda-2019-ml-test/Sample Submission.csv")
Train_Data.head() # Gives First five rows
Train_Data.info() # Gives the information of a train data
Train_Data.drop(["timeindex"],axis=1,inplace=True) #Droping two columns from Train Data becuase its basically for classification .The flag column has binary (0 & 1)
Y_Train_Data = Train_Data[['flag']]
X_Train_Data = Train_Data.drop(['flag'],axis=1)
Train_Data['flag'].value_counts() #It gives the counts of flag variables
from sklearn.model_selection import train_test_split # Importing train_test_split - which is used to split the data data into train and test
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Train_Data,Y_Train_Data, test_size=0.2)#Splitting the data
print('Training Set:')                                # Gives the shape of train and test dataset
print('Number of datapoints: ', X_Train.shape[0])
print('Number of features: ', Y_Train.shape[1])
print('\n')
print('Test Set:')
print('Number of datapoints: ', X_Test.shape[0])
print('Number of features: ', Y_Test.shape[1])
from sklearn.metrics import f1_score                      # Importing Classification modules for finding the f1 score for each algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
Model_1 = DecisionTreeClassifier() # fitting model
Model_1.fit(X_Train,Y_Train)
Predictions = Model_1.predict(X_Test)
f1_score(Predictions,Y_Test)
Model_2 = KNeighborsClassifier()
Model_2.fit(X_Train,Y_Train)
Predictions = Model_2.predict(X_Test)
f1_score(Predictions,Y_Test)
Model_3 = LogisticRegression()
Model_3.fit(X_Train,Y_Train)
Predictions = Model_3.predict(X_Test)
f1_score(Predictions,Y_Test)
Model_4 = RandomForestClassifier(n_estimators = 1000)
Model_4.fit(X_Train,Y_Train)
Predictions = Model_4.predict(X_Test)
f1_score(Predictions,Y_Test)
Test_Data.drop(['timeindex'],axis=1,inplace=True)
Test_Data['Anomaly_Flag'] = Model_4.predict(Test_Data)
Sample_submission['flag'] = Test_Data['Anomaly_Flag']
Sample_submission.to_csv(Test_data)