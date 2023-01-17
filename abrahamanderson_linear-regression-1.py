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
import matplotlib as plt

%matplotlib inline

import seaborn as sns

# we get the data we will use

df=pd.read_csv("../input/Ecommerce")

df.head()
# Firstly I want to get more information about the data

df.describe() # I get all the statistical information about the numerical columns in the data
df.info() #It seem that the data has three nonnumerical data and five numerical data
#I want to visualize the data in order to get some relations and better insights about the data

sns.pairplot(df)
#It seems that there is a correlation between "Time on App" and "Yearly Amount Spent"

#I want to compare them closely

sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=df)
sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=df)

#There is better correlation between time on app and yearly amount spent than between time in website and yearly amount spent
#But Length of Membership correlates most with yearly amount spent

sns.lmplot(x="Length of Membership", y="Yearly Amount Spent", data=df)
#After learning some details about the data I will use linear regression model for machine learning

# First of all I will split the data into training and testing sets.

# I set a variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column

X=df[["Avg. Session Length","Time on App","Time on Website","Length of Membership"]]

y=df["Yearly Amount Spent"] #I want to predict yearly amount spend 
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)# I will use %30 of the data for the test size

from sklearn.linear_model import LinearRegression

lm=LinearRegression()

lm.fit(X_train,y_train)#I implement Linear Regression for the training set of the data
#In order to evaluate the performance 

predictions=lm.predict(X_test) #here we predict the y_test values from X_test data according to the our trained Linear Regression data

predictions
# here I will visualize the real test values(y_test) versus the predicted values.

sns.scatterplot(y_test,predictions)

#It seems that our linear regression model predict ver well

# I will evaluate our model performance by calculating the residual sum of squares and the explained variance score

from sklearn import metrics

print("MAE:",metrics.mean_absolute_error(y_test,predictions))

print ("MSE:",metrics.mean_squared_error(y_test,predictions))

print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,predictions)))
#Evaluation of  the explained variance score (R^2)

metrics.explained_variance_score(y_test,predictions) #This shows our model predict %99 of the variance
sns.distplot(y_test-predictions,bins=50) #this figure also proves that our model fits very good

#There is no huge differences between our predictions and actual y data
cdf=pd.DataFrame(lm.coef_,X.columns,columns=["Coefficients"])

cdf
#This shows that one unit increase in Average session length causes 25 dolars more yearly spent if all other features fixed

#one unit increase in time on app causes 38 dolar spent money yearly

# one unit increase in time on website causes 19 cent increase spent money yearly

# one unit increase in length of membership causes 19 cent increase spent money yearly
