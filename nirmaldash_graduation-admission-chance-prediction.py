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
#Load all libraries requires to create this model

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
#Before Collection of Data we need to verify the dataset is present in the drive or not

for dirname,_,filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname,filename))
#Collect the data for further analysis

Admission_df=pd.read_csv('/kaggle/input/admission-predict/Admission_Predict.csv',delimiter=",")

#Change the Column name with better access

Admission_df.rename(

    {

        'Serial No.':'S.N.',

        'GRE Score':'GRE_Score',

        'TOEFL Score':'TOEFL_Score',

        'University Rating':'University_Rating',

        'Chance of Admit ':'Chance_of_Admit'

    },axis=1,inplace=True

)

Admission_df.set_index('S.N.',inplace=True)
#Now it's time do EDA before work on model creation

#Check is there any missing value in the dataset or not

Admission_df.isnull().sum()

#Check statistical data which gives the idea about the average and SD for this all features

Admission_df.describe()

#Check the Dataset information

Admission_df.info()

#Check the shape of the dataset

Admission_df.shape

#Check is there any outlier in the dataset or not, identify through box plot

for column_name in Admission_df.columns:

    plt.figure()

    Admission_df.boxplot(column_name)

#Filter the outlier from dataset

Admission_df[((Admission_df['LOR ']<1.5)|(Admission_df['CGPA']<7.25)|(Admission_df['Chance_of_Admit']<0.2))]

#These are the outlier data, we need to remove after discussing with SME
X=Admission_df.drop('Chance_of_Admit',axis=1)

y=Admission_df['Chance_of_Admit']

print(X.columns)

print(y.keys)
#Split the X and y into train and test dataset with 80:20 ratio

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=21)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
#Building the Model

model_lr=LinearRegression() #Instantiate the estimator object, here we are using the intercept for model building

model_lr.fit(X_train,y_train)

print('Coefficient value is:',model_lr.coef_)

print('Intercept value is:',model_lr.intercept_)
#Evaluate the y_predict using the above model

y_predict=model_lr.predict(X_test)

pd.DataFrame({'Y Actual':y_test,'Y Predict':y_predict})
#Find out the RMSE to check how good the model is

print('RMSE score of this test model is %f' %np.sqrt(mean_squared_error(y_test,y_predict)))
#Now we do the similar exsercise on train dataset and compare with RMSE score for test dataset



#Evaluate the model on X_train to predict Y_train



y_predict_train=model_lr.predict(X_train)

pd.DataFrame({'Actual Y train Data':y_train,'Predicted Y train Data':y_predict_train})
#Find out the RMSE score to validate how good the model do the prediction

print('RMSE score for train model is %f'%np.sqrt(mean_squared_error(y_train,y_predict_train)) )