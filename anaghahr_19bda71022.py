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
#Importing basic libraries

import pandas as pd

import numpy as np
#Importing dataset

data = pd.read_csv('/kaggle/input/bda-2019-ml-test/Train_Mask.csv')
#Knowing the data

data.describe()
#Checking for missing values

data.isnull().sum()

data.isna().sum()

#As the data is clean proceeding with further steps
#Splitting the data into train and test

from sklearn.model_selection import train_test_split  #Importing library for splitting data

x_train, x_test, y_train, y_test = train_test_split(data.drop('flag',axis=1),data['flag'], test_size=0.25)

#Splitting the data into train and test with 75:25 ratio and dropiing the dependent variable 'flag'
#Checking for Data Dimensions

x_train.shape,x_test.shape,y_train.shape,y_test.shape
#Feature Scaling 

from sklearn.preprocessing import StandardScaler #Importing relevant library to standardize values 

sc = StandardScaler() #Creating an object which scales the data

x_train = sc.fit_transform(x_train) #Scaling trainig data
#Applying Support Vector Machine model

from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0) # creating an object which contains Support vector machine algorithm

classifier.fit(x_train, y_train) #Application of the model to the training set
#Getting the test set data

test = pd.read_csv('/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv')
#Scaling the test set data

x_test = sc.transform(test) 
#Applying model to test data

result = classifier.predict(x_test)
#Reading sample file

sample = pd.read_csv('/kaggle/input/bda-2019-ml-test/Sample Submission.csv')
#Updating the result values to 

sample = sample.assign(flag=result)

sample.to_csv('submit1.csv',index= False)