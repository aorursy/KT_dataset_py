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
df = pd.read_csv('/kaggle/input/delhi-housing-dataset/Delhi Housing.csv')
#This is to display whole of the dataframe

pd.options.display.max_rows = None

pd.options.display.max_columns = None
# Looking at the first five rows

df.head()
# What type of values are stored in the columns?

df.info()
# Converting Yes to 1 and No to 0

df['mainroad'] = df['mainroad'].map({'yes':1 , 'no':0})

df['guestroom'] = df['guestroom'].map({'yes': 1, 'no': 0})

df['basement'] = df['basement'].map({'yes': 1, 'no': 0})

df['hotwaterheating'] = df['hotwaterheating'].map({'yes': 1, 'no': 0})

df['airconditioning'] = df['airconditioning'].map({'yes': 1, 'no': 0})

df['prefarea'] = df['prefarea'].map({'yes': 1, 'no': 0})
df.head()
# Creating a dummy variable for 'furnishingstatus'

df_1 = pd.get_dummies(df['furnishingstatus'])
# The result has created three variables that are not needed.

df_1.head()
# we don't need 3 columns.

# we can use drop_first = True to drop the first column from status df.

df_1 = pd.get_dummies(df['furnishingstatus'],drop_first=True)
#Adding the results to the master dataframe

df = pd.concat([df,df_1],axis=1)
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

def normalize (x): 

    return ( (x-np.min(x))/ (max(x) - min(x)))

                                            

                                              

# applying normalize ( ) to all columns 

df = df.apply(normalize)
y = df['price']

X = df.drop('price',axis = 1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=100)
# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 9

lr = LinearRegression()

rfe = RFE(lr,n_features_to_select= 9)             # running RFE

rfe = rfe.fit(X_train, y_train)

print(rfe.support_)           # Printing the boolean results

print(rfe.ranking_)
cols = X_train.columns[rfe.support_]
# Creating X_test dataframe with RFE selected variables

X_train_rfe = X_train[cols]
# Adding a constant variable 

import statsmodels.api as sm  

X_train_rfe = sm.add_constant(X_train_rfe)
model = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model



#Let's see the summary of our linear model

model.summary()
# Now let's use our model to make predictions.



# Creating X_test_6 dataframe by dropping variables from X_test

X_test_rfe = X_test[cols]



# Adding a constant variable 

X_test_rfe = sm.add_constant(X_test_rfe)



# Making predictions

y_pred = model.predict(X_test_rfe)
# Now let's check the Root Mean Square Error of our model.

import numpy as np

from sklearn import metrics

print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))