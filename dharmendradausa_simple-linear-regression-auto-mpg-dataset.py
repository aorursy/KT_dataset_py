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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
mpg_df = pd.read_csv('../input/autompg-dataset/auto-mpg.csv')#loading the dataset in the notebook
mpg_df.head(10)
mpg_df = mpg_df.drop('car name',axis=1)#dropping car name column
mpg_df.head(10)
mpg_df['origin'].unique()
#replace the number in the categorical variables to the acutal country names

mpg_df['origin']=mpg_df['origin'].replace({1:'america',2:'europe',3:'asia'})
mpg_df.head()
#changing categorical variables into dummy variables

mpg_df = pd.get_dummies(mpg_df,columns=['origin'])
mpg_df.head(10)
mpg_df.describe().transpose()
#describe function is very important in building the models.

#lets analyse the data by describe function - 'Median' is the central value or 50% of the column.

#we can know the right tail by Max-Q3(75%) and Left tail by Q1(25%)-Min, this will show us the potential outliers in the data.

#if the tails are long on either side then the mean gets easily impacted by the extreme values, extreme values are on the extreme end of the tails.

#Alogorithm expect the data input as symetric however the skew can happen in the data because of the extreme outliers therefore it becomes

#very important to analyse the range of data well.

#for example 'displacement' column right tail is for 193 and left is for 36.25, which shows that the right tail is five times more than the left tail,

#and the data is rightly skewed because of outliers.

#in 'acceleration' column right tail is 7.6 and the left is 5.83 which shows that no big difference in the data and it's seems symetric.
#Note - HP column is missing in the describe output means something is wrong with the column.

#we will check if horsepower column has some value other than digit.

temp = pd.DataFrame(mpg_df.horsepower.str.isdigit())

temp[temp['horsepower']==False]#False values show the values which are not digits.
#we will check what value is there in the row and find that there is '?'

mpg_df['horsepower'].iloc[[32,126,330,336,354,374]]
#we will replace the '?' with 'nan'

mpg_df = mpg_df.replace('?',np.nan)
#lets check if the values has been replaced or not

mpg_df[mpg_df.isnull().any(axis=1)]

#great, the values has been replaced with nan
#we must know the data types 

mpg_df.dtypes

#we get to know that 'horsepower' datatype is a object means string hence we have to change it to numerical data to make calculations.
#let's convert the data type from string to float64 now

mpg_df['horsepower']=mpg_df['horsepower'].astype('float64')

mpg_df.dtypes
#to check null values in 'horsepower' column

mpg_df['horsepower'].isnull().sum()
#Handling missing values - if the data is random we can use strategy to fill the missing values with the Median,let's find out median

mpg_df.median()
#filling null values by median

mpg_df = mpg_df.apply(lambda x: x.fillna(x.median()),axis=0 )
#lets check if there are still null values in the dataset, it's solved, the column has no null values now.

mpg_df['horsepower'].isnull().sum()
#Bi-variate analysis

#As we see that all columns are numerical, we can proceed to bivaritate analysis of the data, for that we can use 

#pairplot in seaborn library.

mpg_df_attr = mpg_df.iloc[:,0:10]

sns.pairplot(mpg_df_attr)#it will show the histogram of the data
#Copy all the predictor variables in X dataframe.

X = mpg_df.iloc[:,1:10]

X
#since mpg is the dependent(target) variable so take it in y dataframe

y = mpg_df.iloc[:,0:1]

y
#split the data into training and test dataset

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=1)

#random state can be any number, it will keep the random sample data same irrecpective of multiple runs.
y_train
#invoke the linear regression function and find the best fit model 

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train,y_train)
#lets find out the coefficient for all independent variables

for idx, col_name in enumerate(X_train.columns):

    print("the coefficient for {} is {}".format(col_name,reg.coef_[0][idx]))
#now let's find out the intercept for the model, intercept is the value that remains constant when all the independent variables are zero.

intercept = reg.intercept_[0]

print("The intercept for our model is {}".format(intercept))
#checking the accuracy of the model

print("The accuracy of the model is",(reg.score(X_test,y_test)))
from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model

poly = PolynomialFeatures(degree=2, interaction_only=True)

X_train_ = poly.fit_transform(X_train)

X_test_ = poly.fit_transform(X_test)

poly_clf = linear_model.LinearRegression()

poly_clf.fit(X_train_,y_train)

y_pred = poly_clf.predict(X_test_)

#let's check if the accuracy of the model has been increased, great it's now 0.86 which was earlier 0.84, a slight improvment in the

#accuracy since we have taken the true interactions into account.

print("The improved accuracy of the model is",(poly_clf.score(X_test_,y_test)))
#thank you for reading and i welcome your suggestions for the improvement in above model.