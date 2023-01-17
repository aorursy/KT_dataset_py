# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='white')

%matplotlib inline



import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor



from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import the dataset from sklearn library

from sklearn.datasets import load_boston

data = load_boston()

print(data.keys())
# Load features, and target variable, combine them into a single dataset

X = pd.DataFrame(data.data, columns=data.feature_names)

# Add constant 

X = sm.add_constant(X)

y = pd.Series(data.target, name='MEDV')

dataset = pd.concat([X, y], axis=1)



# Split training and test dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



# Print dataset description

print(data.DESCR)
dataset.head()
dataset.info()
dataset.describe()
plt.figure(figsize=(10, 8))

sns.distplot(dataset['MEDV'], rug=True)

plt.show()
mask = np.zeros_like(dataset.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)



plt.figure(figsize=(10,10))

sns.heatmap(dataset.corr(), annot=True, vmin=-1, vmax=1, square=True, mask=mask, cmap=cmap, linewidths=2)

plt.show()
# Fit a linear model to data

exog = X_train.drop('const', axis=1)

endog = y_train

model = sm.OLS(endog, exog).fit()
# Print model summary

model.summary()
# Checking VIF

variables = model.model.exog

vif = np.array([variance_inflation_factor(variables, i) for i in range(variables.shape[1])])

vif
exog = X_train.drop('const', axis=1)

endog = y_train



# Fit model

model = sm.OLS(endog, exog).fit()

# Calculate VIF

variables = model.model.exog

vif = np.array([variance_inflation_factor(variables, i) for i in range(variables.shape[1])])

vif[0] = 0



while sum(np.array(vif) > 5) > 0:

    # Find the id of the feature with largest VIF value

    max_vif_id = np.argmax(vif)



    # Delete that feature from exog dataset

    exog = exog.drop(exog.columns[max_vif_id], axis=1)

    

    # Fit model again

    model = sm.OLS(endog, exog).fit()

    # Calculate VIF

    variables = model.model.exog

    vif = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]

    vif[0] = 0



model.summary()
# Check if any P-value larger or equal to 0.05

while sum(model.pvalues >= 0.05) > 0:

    # Find the index of the feature of the largest p-value

    max_pvalue_id = np.argmax(model.pvalues)



    # Delete that feature from exog dataset

    exog = exog.drop(max_pvalue_id, axis=1)

    

    # Fit model again

    model = sm.OLS(endog, exog).fit()

    

model.summary()
# Fit a linear model to data

exog = X_train

endog = y_train

model = sm.OLS(endog, exog).fit()
# Print model summary

model.summary()
# Checking VIF

variables = model.model.exog

vif = np.array([variance_inflation_factor(variables, i) for i in range(variables.shape[1])])

vif
exog = X_train

endog = y_train



# Fit model

model = sm.OLS(endog, exog).fit()

# Calculate VIF

variables = model.model.exog

vif = np.array([variance_inflation_factor(variables, i) for i in range(variables.shape[1])])

vif[0] = 0



while sum(np.array(vif) > 5) > 0:

    # Find the id of the feature with largest VIF value

    max_vif_id = np.argmax(vif)



    # Delete that feature from exog dataset

    exog = exog.drop(exog.columns[max_vif_id], axis=1)

    

    # Fit model again

    model = sm.OLS(endog, exog).fit()

    # Calculate VIF

    variables = model.model.exog

    vif = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]

    vif[0] = 0



model.summary()
# Check if any P-value larger or equal to 0.05

while sum(model.pvalues >= 0.05) > 0:

    # Find the index of the feature of the largest p-value

    max_pvalue_id = np.argmax(model.pvalues)



    # Delete that feature from exog dataset

    exog = exog.drop(max_pvalue_id, axis=1)

    

    # Fit model again

    model = sm.OLS(endog, exog).fit()

    

model.summary()