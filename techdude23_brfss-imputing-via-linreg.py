# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import linear_model # regression model



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import the data

new_data = pd.read_csv('../input/2015.csv')
var_cols = ['MENTHLTH', 'SEX', '_AGEG5YR', 'EDUCA', 'EMPLOY1', 'INCOME2', '_RACE', 'NUMADULT', 'MARITAL', 'VETERAN3', 'PREGNANT', 

            'ADPLEASR', 'ADDOWN', 'ADSLEEP', 'ADENERGY', 'ADEAT1', 'ADFAIL', 'ADTHINK', 'ADMOVE']



sk_data = pd.DataFrame(new_data, columns = var_cols)



# MENTHLTH code: [77, 99]

sk_data['MENTHLTH'].replace([77, 99], np.nan, inplace=True)

sk_data['MENTHLTH'].replace(88, 0, inplace=True)



# _AGEG5YR code: 14 is 'missing/refuse to answer'

sk_data['_AGEG5YR'].replace(14, np.nan, inplace=True)



# EDUCA code: 9 is 'refuse to answer'

sk_data['EDUCA'].replace(9, np.nan, inplace=True)



# EMPLOY1 code: 9 is 'refuse to answer'

sk_data['EMPLOY1'].replace(9, np.nan, inplace=True)



# INCOME2 code: [77, 99] is 'refuse to answer/don't know'

sk_data['INCOME2'].replace([77, 99], np.nan, inplace=True)



# _RACE code: 9 is 'don't know/refuse to answer/not sure'

sk_data['_RACE'].replace(9, np.nan, inplace=True)



# MARITAL code: 9 is 'refuse to answer'

sk_data['MARITAL'].replace(9, np.nan, inplace=True)



# VETERAN3 code: [7, 9] is 'don't know/refused'

sk_data['VETERAN3'].replace([7, 9], np.nan, inplace=True)



# PREGNANT code: [7, 9] is 'don't know/refused'

sk_data['PREGNANT'].replace([7, 9], np.nan, inplace=True)



##########



for i in var_cols[11:]:

    sk_data[i].replace([77, 99], np.nan, inplace=True)

    sk_data[i].replace(88, 0, inplace=True)



sk_data.head(10)
# Normalize / Transform to prevent prediction of negative values

# Ensure outputs are 0 to 14

# Use logistic transform f(x) = 14 / (1 + e^(-x))

# Inverse: f^(-1) = -log(14 / y - 1)



def logistic_inverse(y):

    threshold = 3

    if y == 0:

        return -threshold

    elif y == 14:

        return threshold

    return -np.log(14 / y-1)



def logistic(x):

    return 13 / (1 + np.exp(-x))



# We want the threshold value to be larger than the next largest value, but not too much.

print([logistic_inverse(x) for x in np.arange(15)])
sk_data.shape
# drop nulls from ['MENTHLTH':'PREGNANT']

sk_data = sk_data.dropna(axis=0, subset=var_cols[:10]).reset_index(drop=True)

sk_data.head()

print(sk_data.shape)
def isPregnant(x):

    if x['SEX'] == 1 or (x['_AGEG5YR'] >= 6 and x['_AGEG5YR'] <= 13):

        return 2

    else:

        return x['PREGNANT']



sk_data['PREGNANT'] = sk_data.apply(isPregnant, axis=1)

print('Missing values from PREGNANT:', sk_data['PREGNANT'].isnull().sum())

sk_data['PREGNANT'].value_counts()
###########

# Evaluation via train/test set



src_data = sk_data.copy()



cols = var_cols[11:]

questions = pd.DataFrame(src_data, columns=cols)

src_data.drop(cols, axis = 1, inplace=True)

src_data.drop('MENTHLTH', axis=1, inplace=True)



# Convert to dummy

for i in src_data.columns[::-1]:

    dummy_data = pd.get_dummies(src_data[i], prefix=i) # Convert col to dummy

    src_data.drop(i, axis=1, inplace=True) # Drop the original col

    src_data = dummy_data.join(src_data) # append dummies to src

src_data = src_data.join(questions) # append back the 8 questions
print("Shape:", src_data.shape)

src_data.head()
model_data = src_data.dropna().reset_index(drop=True)

print("Shape:", model_data.shape)

model_data.head()
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(model_data, test_size=0.15)

train_data.head()
test_data.head()
input_vars = train_data.columns[:68]

output_vars = train_data.columns[68:]



train_input = train_data[input_vars]

train_output = train_data[output_vars]

train_input.head()
# Fit the data to the model

regr = linear_model.LinearRegression()

# regr.fit(train_input, train_output)



# In order to bound the output with 0 and 14, we need to perform a logistic transform

train_output_transformed = pd.DataFrame()

for col in output_vars:

    train_output_transformed[col] = train_output[col].apply(logistic_inverse)

train_output_transformed.head()

regr.fit(train_input, train_output_transformed)
test_data_copy = test_data.copy()

test_data_copy.head()
for i in test_data_copy.index:

    test_data_copy.loc[i, np.random.choice(output_vars, size=2, replace=False)] = np.nan

    # Randomly set 2 values to null

test_data_copy.head()
test_data.head()
pred_cols = var_cols[11:]

pred_cols = [i + '_prediction' for i in pred_cols]



pred_transformed = regr.predict(test_data_copy[input_vars])

pred_transformed = pd.DataFrame(pred_transformed, columns=output_vars)

# we need to transform the logistic values back to [0, 14]

pred = pd.DataFrame()

for col in output_vars:

    new_col = col + "_prediction"

    pred[new_col] = pred_transformed[col].apply(logistic)

pred = pred.round(0)

pred.head()
pred.describe()
test_data_impute = test_data.copy().reset_index(drop=True)

test_data_impute.head()
test_data_impute = test_data_impute.join(pred)

test_data_impute.head()
error_cols = var_cols[11:]

error_cols = [i + '_error' for i in error_cols]

error_cols
for i in range(8):

    test_data_impute[error_cols[i]] = (test_data_impute[var_cols[(11+i)]] - test_data_impute[pred_cols[i]])**2



test_data_impute.head()
len(test_data_copy.index) * 2
np.sum(np.sum(pd.isnull(test_data_copy.loc[:, var_cols[11:]])))
error_sum = np.sum(np.sum(test_data_impute[error_cols]))

error_n = np.sum(np.sum(pd.isnull(test_data_copy.loc[:, var_cols[11:]])))



print('error sum:', error_sum, '\nerror n:', error_n)

print("Mean square error:", error_sum / error_n)