# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split # train, test split

from sklearn.linear_model import LogisticRegression # regression algo

from sklearn.preprocessing import LabelEncoder # for converting data to categorical

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dfEmployees = pd.read_csv('/kaggle/input/employee-analysis/employee.csv')
dfEmployees.isnull().sum()

label_encoder = LabelEncoder()

attrition_encoded = label_encoder.fit_transform(dfEmployees['Attrition'].values)

business_travel_encoded = label_encoder.fit_transform(dfEmployees['BusinessTravel'].values)

department_encoded = label_encoder.fit_transform(dfEmployees['Department'].values)

education_encoded = label_encoder.fit_transform(dfEmployees['EducationField'].values)

gender_encoded = label_encoder.fit_transform(dfEmployees['Gender'].values)

MaritalStatus_encoded = label_encoder.fit_transform(dfEmployees['MaritalStatus'].values)

JobRole_encoded = label_encoder.fit_transform(dfEmployees['JobRole'].values)

Over18_encoded = label_encoder.fit_transform(dfEmployees['Over18'].values)

OverTime_encoded = label_encoder.fit_transform(dfEmployees['OverTime'].values)

XData = dfEmployees.drop(['Attrition'],axis=1)

XData['BusinessTravel'] = pd.DataFrame(data=business_travel_encoded.flatten(), columns=['BusinessTravel'])

XData['Department'] = pd.DataFrame(data=department_encoded.flatten(), columns=['Department'])

XData['EducationField'] = pd.DataFrame(data=education_encoded.flatten(), columns=['EducationField'])

XData['Gender'] = pd.DataFrame(data=gender_encoded.flatten(), columns=['Gender'])

XData['MaritalStatus'] = pd.DataFrame(data=MaritalStatus_encoded.flatten(), columns=['MaritalStatus'])

XData['JobRole'] = pd.DataFrame(data=JobRole_encoded.flatten(), columns=['JobRole'])

XData['Over18'] = pd.DataFrame(data=Over18_encoded.flatten(), columns=['Over18'])

XData['OverTime'] = pd.DataFrame(data=Over18_encoded.flatten(), columns=['OverTime'])

YData=pd.DataFrame(data=attrition_encoded.flatten(), columns=['Attrition'])
XData.isnull().sum()

YData.isnull().sum()

trainX,testX,trainY,testY = train_test_split(XData,YData,test_size=0.3,random_state=1) 
model = LogisticRegression(max_iter=1000)

print(trainX.shape)

print(trainY.values.ravel().shape)

model.fit(trainX,trainY.values.ravel())
model.predict(testX)