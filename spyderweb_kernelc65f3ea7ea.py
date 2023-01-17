# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/train_2v.csv')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Imputer
labelEncoder = LabelEncoder()
#removing the ID which may not be required for analysing and prediting 
x = dataset.iloc[:,1:-1]
#getting the labels
y = dataset.iloc[:,-1].values
# filling the missing values in smoking_staus column with the maximum values 
x.smoking_status.fillna(x.smoking_status.dropna().max(), inplace=True)
#convering muptiple categorical variables(>2) of a column to numerical
work_type = pd.get_dummies(x.work_type)
smoking_status = pd.get_dummies(x.smoking_status)
#droping the categorical variables. Will be adding the converted numerical values of the column
train = x.drop(['work_type', 'smoking_status'], axis =1)
#joining the converted numerical values of the columns to the actual dataframe
train= train.join(pd.DataFrame(work_type, index=train.index))
train = train.join(pd.DataFrame(smoking_status, index=train.index))
#converting the dataframe in numpy array
x = train.values
#converting the categorical variables to numerical variables
x[:,0] = labelEncoder.fit_transform(x[:,0])
x[:,4] = labelEncoder.fit_transform(x[:,4])
x[:,5] = labelEncoder.fit_transform(x[:,6])
#handling the missing numerical data by filling it with the mean of the column
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:,7:8])
x[:,7:8] = imputer.transform(x[:,7:8])
#splitting the dataset into train and validation
x_train, x_test, y_train, y_test = train_test_split(x , y , test_size=0.2)
#using decision tree to classify the heart_stroke will come or not
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
# checking the accuracy on validation set
count = 0
for i in range(len(x_test)):
    pre = clf.predict([x_test[i]])
    if pre == y_test[i]:
        count += 1
print("Accuracy: " , count/len(x_test)*100)
#handing the missing and categorical data for test set
test_dataset = pd.read_csv('../input/test_2v.csv')
x_test = test_dataset.iloc[:,1:]
x_test.smoking_status.fillna(x_test.smoking_status.dropna().max(), inplace=True)
test_work_type = pd.get_dummies(x_test.work_type)
test_smoking_status = pd.get_dummies(x_test.smoking_status)
test = x_test.drop(['work_type', 'smoking_status'], axis =1)
test = test.join(pd.DataFrame(test_work_type, index=test.index))
test = test.join(pd.DataFrame(test_smoking_status, index=test.index))
x_test = test.values
x_test[:,0] = labelEncoder.fit_transform(x_test[:,0])
x_test[:,4] = labelEncoder.fit_transform(x_test[:,4])
x_test[:,5] = labelEncoder.fit_transform(x_test[:,6])
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x_test[:,7:8])
x_test[:,7:8] = imputer.transform(x_test[:,7:8])
#writing the prediction on test set on a csv file
file = open('../submission.csv','w')
file.write('Prediction'+'\n')
for i in x_test:
    prediction = clf.predict([i])
    file.write(str(prediction).replace("[","").replace("]","") + '\n')
file.close()



