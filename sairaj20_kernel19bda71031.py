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
# importing packages..
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#importing Train data
Train1 = pd.read_csv("../input/bda-2019-ml-test/Train_Mask.csv")
#importing Test data
Test = pd.read_csv("../input/bda-2019-ml-test/Test_Mask_Dataset.csv")
# import sample data
SampleSubmission = pd.read_csv("../input/bda-2019-ml-test/Sample Submission.csv")
# reading the Train data
Train.head()
#reading the Test data
Test.head()
#droping flag column beacuse it is an dependent variable of the data
Train = Train1.drop('flag',axis = 1)
Train.head(10)
##DATA PREPROCESSING
#let us find the data contains null values or not
print("Percentage null or na values in data")
((Train.isnull() | Train.isna()).sum() * 100 / Train.index.size).round(2)
# here we can se the number of rows,cloumns avilabel in the data
Train1.shape
#which the gives basic satistics of the data
Train1.describe()
# which shows the information of the data
Train1.info()
#which shows the data types availabel in the data
Train1.dtypes
#finding the correlation of the data
correlation = Train.corr()
correlation
#Splting the data into x,y variable
data = Train
data1 = Test
x = Train
y = Train1['flag']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42) 
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
#fitting normailisation to data
def normalize(data):
    for feature in x.data:
        train[feature] -= data[feature].mean()
        train[feature] /= data[feature].std()
        return data
## FITTING A MODEL
#import logistic regression from sklearn
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train,y_train)
# predecting the test values
y_pred = logreg.predict(x_test)
# import metrics from sklearn to use confusion matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test,y_pred)
cnf_matrix
#let us find Accuracy,precison,Recall
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
print("Precison:",metrics.precision_score(y_test,y_pred))
print("Recall:",metrics.recall_score(y_test,y_pred))

prediction = logreg.predict(Test)
Samle_Submission = pd.read_csv("../input/bda-2019-ml-test/Sample Submission.csv")
Sample_Submission['flag'] = prediction
Sample_Submission.to_csv("submit_1.csv",index=False)
