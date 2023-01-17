# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



data = pd.read_csv("../input/IPL2013.csv")  # Step 1. Load the dataset as a Python dataframe object

data.info()  # Step 2. Look into the basic info about the dataset - attributes and their types
data.iloc[0:5, 0:10]  # Step 3. Show the initial 10 columns of the first 5 rows of the dataset
# Step 4. Remove the irrelevant features



print(data.columns)



data = data[['AGE', 'COUNTRY', 'PLAYING ROLE',

       'T-RUNS', 'T-WKTS', 'ODI-RUNS-S', 'ODI-SR-B', 'ODI-WKTS', 'ODI-SR-BL',

       'CAPTAINCY EXP', 'RUNS-S', 'HS', 'AVE', 'SR-B', 'SIXERS', 'RUNS-C',

       'WKTS', 'AVE-BL', 'ECON', 'SR-BL', 'SOLD PRICE']]

print(data.columns)
# Step 5. Encode the categorical features

categorical_features = ['COUNTRY', 'PLAYING ROLE']

encoded_data = pd.get_dummies( data, columns = categorical_features, drop_first = True)

predictors = encoded_data.drop('SOLD PRICE', axis = 1)

target = encoded_data['SOLD PRICE']
# Step 6. Identify similar features and take a subset of similar features

from statsmodels.stats.outliers_influence import variance_inflation_factor

import seaborn as sns 

import matplotlib.pyplot as plt



def get_vif_factors(X):

    X_matrix = X.as_matrix()

    vif = [variance_inflation_factor( X_matrix, i ) for i in range( X_matrix.shape[1] ) ]

    vif_factors = pd.DataFrame()

    vif_factors['column'] = X.columns

    vif_factors['vif'] = vif

    return vif_factors



vif_factors = get_vif_factors(predictors)

vif_factors

columns_with_large_vif = vif_factors[vif_factors.vif > 4].column

print(columns_with_large_vif)

plt.figure(figsize = (12,10))

sns.heatmap(abs(predictors[columns_with_large_vif].corr()), annot = True)

plt.title( "Heatmap depicting correlation between features")
# Step 7. Remove the similar features (other than the representative features identified in step 6b



columns_to_be_removed = ['T-RUNS', 'T-WKTS', 'RUNS-S', 'HS', 'AVE', 'RUNS-C', 'SR-B', 'AVE-BL', 'ECON', 'ODI-SR-B', 'ODI-RUNS-S', 'AGE', 'SR-BL']

predictors = predictors.drop(columns_to_be_removed, axis = 1)

get_vif_factors(predictors)
# Step 8. Split the dataset into training and test sets

from sklearn.model_selection import train_test_split



train_X, test_X, train_y, test_y = train_test_split( predictors , target, train_size = 0.8, random_state = 42 )
# Step 9. Fit a multiple linear regression model (OLS)

import statsmodels.api as sm

ipl_model_2 = sm.OLS(train_y, train_X).fit()
# Step 10. From the model summary, find the statistically significant features influencing the response variable

ipl_model_2.summary2()
significant_vars = ['COUNTRY_IND', 'COUNTRY_ENG', 'SIXERS', 'CAPTAINCY EXP']

predictors = predictors[significant_vars]



ipl_model_3 = sm.OLS(target, predictors).fit()

ipl_model_3.summary2()