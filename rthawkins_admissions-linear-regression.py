# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
# Correlation heat map



corr = df.corr()

top_corr_features = corr.index[abs(corr["Chance of Admit "])>0.5]

plt.figure(figsize=(10,10))

g = sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")



# View highest correlation for the chance of admission



corr["Chance of Admit "].loc[abs(corr["Chance of Admit "])>=.5].sort_values(ascending=False)
# Keep all variables due to high correlation



df = df.drop(['Serial No.'],axis = 1)

x = df.drop(['Chance of Admit '],axis = 1)

y = df['Chance of Admit ']
# Review N/As in dataframe via function

# Source: https://towardsdatascience.com/cleaning-missing-values-in-a-pandas-dataframe-a88b3d1a66bf



def assess_NA(data):

    """

    Returns a pandas dataframe denoting the total number of NA values and the percentage of NA values in each column.

    The column names are noted on the index.

    

    Parameters

    ----------

    data: dataframe

    """

    # pandas series denoting features and the sum of their null values

    null_sum = data.isnull().sum()# instantiate columns for missing data

    total = null_sum.sort_values(ascending=False)

    percent = ( ((null_sum / len(data.index))*100).round(2) ).sort_values(ascending=False)

    

    # concatenate along the columns to create the complete dataframe

    df_NA = pd.concat([total, percent], axis=1, keys=['Number of NA', 'Percent NA'])

    

    # drop rows that don't have any missing data; omit if you want to keep all rows

    df_NA = df_NA[ (df_NA.T != 0).any() ]

    

    return df_NA



assess_NA(df)
# Creating a test and training dataset



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
# Linear regression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

print(regressor.intercept_)

print(regressor.coef_)
#Displaying the difference between the actual and the predicted

y_pred = np.round(regressor.predict(X_test),2)

df_output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

print(df_output)
# Review residuals



plt.scatter(y_pred, y_pred - y_test)

plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max())

plt.show()
from sklearn.metrics import mean_squared_error, r2_score

import sklearn.metrics as metrics



#Checking the accuracy of Linear Regression

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('R Squared:', r2_score(y_test, y_pred))