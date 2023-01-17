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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#reading training data
data=pd.read_csv("/kaggle/input/bda-2019-ml-test/Train_Mask.csv")
#head gives first five data
data.head()
#tail gives last values of the dataset
data.tail()
#columns of the data
data.columns
#size the of the data
data.size
#information the data
data.info()
#describing the data
data.describe()
#duplicates of the data
data = data.drop_duplicates()
#To find the missing values
data.isnull().sum()
#using heat map To find missing values
sns.heatmap(data.isnull(), cbar='Flase')
#gives the flag count
sns.countplot(x="flag",data=data)
#dropping the flag
data.drop('flag',axis=1)
#correlation matrix
data.corr()
#taking variables for training 
x=data.drop('flag',axis=1)
x
y =  data['flag']
y

#model fitting
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)


#randomForest model
from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier(n_estimators = 75,criterion="entropy",random_state=42)
ranfor.fit(x_train,y_train)
#predicting
y_pred=ranfor.predict(x_test)
y_pred
#finding accuracy
from sklearn.metrics import accuracy_score
accuracy_classifier=accuracy_score(y_test,y_pred)
accuracy_classifier
#Loading the testdata
data2=pd.read_csv("/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv")

#this shoes the first 5 from the dataset
data2.head()
#it showes the last part of test data
data.tail()
#columns of the test data
data2.columns
#Size of the data
data2.size
#information of the data
data.info()
#dropping duplicates in test data set
data2 = data2.drop_duplicates()
#missing values
data2.isnull().sum()
#finding missing values with heat map
sns.heatmap(data.isnull(), cbar='Flase')
#predictng the classifier 
predict=classifier.predict(data2)
#predict
predict
#for submission
Sample=pd.read_csv("/kaggle/input/bda-2019-ml-test/Sample Submission.csv")
Sample['flag']=predict
Sample.to_csv('submit_10.csv',index=False)
#for this dataset i tried all algorithms 
#but randomforest is best accuarcy
#getting high accuracy 89%


