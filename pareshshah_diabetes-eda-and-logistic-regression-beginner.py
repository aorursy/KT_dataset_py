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
# Numerical libraries

import numpy as np   



# to handle data in form of rows and columns 

import pandas as pd    



# importing ploting libraries

import matplotlib.pyplot as plt   



#importing seaborn for statistical plots

import seaborn as sns



# Import Logistic Regression machine learning library

from sklearn.linear_model import LogisticRegression



#Sklearn package's data splitting function which is based on random function

from sklearn.model_selection import train_test_split





# calculate accuracy measures and confusion matrix

from sklearn import metrics



# To scale the dimensions we need scale function which is part of sckikit preprocessing libraries

from sklearn import preprocessing



# To enable plotting graphs in Jupyter notebook

%matplotlib inline 
# reading the CSV file into pandas dataframe

diab_df = pd.read_csv('/kaggle/input/diabetes-dataset/diabetes2.csv')
# Let's eyeball the data

diab_df.head()
diab_df.info()
diab_df.shape

# There are 768 rows and 9 columns
diab_df.describe().T

# Insulin and DiabetesPedigreeFunction has mean and median has big diff
# check missing values count

missing_values=diab_df.columns[diab_df.isnull().any()]

diab_df[missing_values].isnull().sum()
# as part of eyeballing data we found there are 0 values in many columns, 

# let's find out how many 0 values are there in all columns



(diab_df == 0).sum(axis=0)
#We will use 'median' to replace 0 for all columns except for 'Insulin' as diff between mean and median was big

diab_df['Pregnancies'].replace(0,diab_df['Pregnancies'].median(),inplace=True)

diab_df['Glucose'].replace(0,diab_df['Glucose'].median(),inplace=True)

diab_df['BloodPressure'].replace(0,diab_df['BloodPressure'].median(),inplace=True)

diab_df['SkinThickness'].replace(0,diab_df['SkinThickness'].median(),inplace=True)

diab_df['BMI'].replace(0,diab_df['BMI'].median(),inplace=True)



diab_df['Insulin'].replace(0,diab_df['Insulin'].mean(),inplace=True)
# let's check if all 0 values are replaced now



(diab_df == 0).sum(axis=0)
# Let us look at the target column which is 'Outcome' to understand how the data is distributed amongst the various values

diab_df.groupby(['Outcome']).count()



# The ratio is almost 1:2 in favor of class 0.  The model's ability to predict class 0 will 

# be better than predicting class 1. 
# Pairplot using sns



sns.pairplot(diab_df)
# data for Insulin and DiabetesPedigreeFunction looks skewed



# the mean for Insulin is 80(rounded) while the median is 30.5 which clearly indicates an extreme long tail on the right

# the mean for DiabetesPedigreeFunction is 0.47 while the median is 0.37 which clearly indicates a long tail on the right

diab_df.corr()

# there is no strong correlation between any columns

# Data for BMI, Glucose and BloodPressure has normal distribution

sns.distplot(diab_df['Glucose'],kde=True)

sns.distplot(diab_df['BloodPressure'],kde=True)

sns.distplot(diab_df['BMI'],kde=True)
# most of the data columns has some outliers, We will see Insulin separetly due to max value 

plt.subplots(figsize=(20,15))

sns.boxplot(data=diab_df.drop(['Insulin','Outcome'],axis=1))
# Insulin has high number of outliers compared to other columns

plt.subplots(figsize=(20,15))

sns.boxplot(data=diab_df['Insulin'])
x=diab_df.drop('Outcome',axis=1)

y=diab_df['Outcome']



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=1)

type(x_train)



model=LogisticRegression()

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

model_score=model.score(x_test,y_test)

print('Accuracy = ',model_score)

print(metrics.confusion_matrix(y_test,y_predict))

# To scale the dimensions we need scale function which is part of sckikit preprocessing libraries

x_train_scaled = preprocessing.scale(x_train)

x_test_scaled = preprocessing.scale(x_test)



model=LogisticRegression()

model.fit(x_train_scaled,y_train)

y_predict=model.predict(x_test_scaled)

model_score=model.score(x_test_scaled,y_test)

print('Accuracy = ',model_score)

print(metrics.confusion_matrix(y_test,y_predict))
