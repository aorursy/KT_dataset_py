# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#importing Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as st



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#This is to display whole of the dataframe

pd.options.display.max_rows = None

pd.options.display.max_columns = None
# Importing housing.csv

df = pd.read_csv('/kaggle/input/delhi-housing-dataset/Delhi Housing.csv')
# Looking at the first five rows

df.head()
# What type of values are stored in the columns?

df.info()
# Converting Yes to 1 and No to 0

df['mainroad'] = df['mainroad'].map({'yes': 1, 'no': 0})

df['guestroom'] = df['guestroom'].map({'yes': 1, 'no': 0})

df['basement'] = df['basement'].map({'yes': 1, 'no': 0})

df['hotwaterheating'] = df['hotwaterheating'].map({'yes': 1, 'no': 0})

df['airconditioning'] = df['airconditioning'].map({'yes': 1, 'no': 0})

df['prefarea'] = df['prefarea'].map({'yes': 1, 'no': 0})
# Now let's see the head

df.head()
# Creating a dummy variable for 'furnishingstatus'()

df_1 = pd.get_dummies(df['furnishingstatus'])
# The result has created three variables that are not needed.

df_1.head()
# we don't need 3 columns.

# we can use drop_first = True to drop the first column from status df.

df_1 = pd.get_dummies(df['furnishingstatus'],drop_first=True)
#Adding the results to the master dataframe

df = pd.concat([df,df_1],axis=1)
# Now let's see the head of our dataframe.

df.head()
# Dropping furnishingstatus as we have created the dummies for it

df.drop(['furnishingstatus'],axis=1,inplace=True)
# Now let's see the head of our dataframe.

df.head()
# Let us create the new metric and assign it to "areaperbedroom"

df['areaperbedroom'] = df['area']/df['bedrooms']
# Metric:bathrooms per bedroom

df['bbratio'] = df['bathrooms']/df['bedrooms']
df.head()
#defining a normalisation function

def normalize(x):

    return((x-np.min(x))/(max(x)-min(x)))



# applying normalize ( ) to all columns 

df = df.apply(normalize) 
y = df['price']

X = df.drop('price',axis = 1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=100)
import statsmodels.api as sm   # Importing statsmodels



Xc = sm.add_constant(X)        # Adding a constant column to our dataframe



# create a first fitted model

model = sm.OLS(y,Xc).fit()



#Let's see the summary of our first linear model

model.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
pd.DataFrame([vif(Xc.values,i) for i in range(Xc.shape[1])],index = Xc.columns,columns=['VIF'])
# Let's see the correlation matrix 

plt.figure(figsize = (15,15))     # Size of the figure

sns.heatmap(df.corr(),annot = True)
# Dropping highly correlated variables and insignificant variables 

Xc = Xc.drop(['bedrooms','semi-furnished','areaperbedroom','bbratio'],axis = 1)

model = sm.OLS(y,Xc).fit()

model.summary()
y = df['price']    #dependent

X = df.drop(['price','bedrooms','semi-furnished','areaperbedroom','bbratio'],axis = 1)    #independent
# Model building

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train,y_train)
# Making predictions

y_test_pred = lr.predict(X_test)

y_train_pred = lr.predict(X_train)
from sklearn.metrics import r2_score,mean_squared_error

print('r-square for train: ', r2_score(y_train,y_train_pred))

print('RMSE for train: ',np.sqrt(mean_squared_error(y_train,y_train_pred)))



print('\n')

print('r-square for test: ', r2_score(y_test,y_test_pred))

print('RMSE for test: ', np.sqrt(mean_squared_error(y_test,y_test_pred)))