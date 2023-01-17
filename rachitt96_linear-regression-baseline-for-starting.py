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
# Importing required libraries



from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_log_error

import matplotlib.pyplot as plt
#let's load datasets



train_df = pd.read_csv("/kaggle/input/ashrae-energy-prediction/train.csv")

bilding_df = pd.read_csv("/kaggle/input/ashrae-energy-prediction/building_metadata.csv")

test_df = pd.read_csv("/kaggle/input/ashrae-energy-prediction/test.csv")
# Joining required dataset in single dataframe



df_train_building_left = pd.merge(train_df, bilding_df, on = "building_id", how = "left")

df_test_building_left = pd.merge(test_df, bilding_df, on = "building_id", how = "left")
df_train_building_left.head()
for column in df_train_building_left:

    print(column + "\t" + str(df_train_building_left[column].isnull().any()))
# converting data into training and testing part



X_train, X_test, y_train, y_test = train_test_split(df_train_building_left[["building_id", "meter", "site_id", "primary_use", "square_feet"]], df_train_building_left["meter_reading"], test_size = 0.25)
# Applying label encoding technique on "primary_use" column as it contains categorical text data



label_encoder = preprocessing.LabelEncoder()



X_train["primary_use"] = label_encoder.fit_transform(X_train["primary_use"])

X_test["primary_use"] = label_encoder.transform(X_test["primary_use"])



# Normalizing the dataset because "square_feet" is in different scale than other columns 



standard_scaler = preprocessing.StandardScaler().fit(X_train)



X_train = standard_scaler.transform(X_train)

X_test = standard_scaler.transform(X_test)
# Building a simple linear regression model on preprocessed data



lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train.values)

y_pred_lr = lin_reg.predict(X_test)
y_pred_lr[y_pred_lr < 0] = 0
# Calculating accuracy



print(np.sqrt(mean_squared_log_error( y_test, y_pred_lr )))
# Plot importance of all features based on linear regression coefficients



%matplotlib inline

plt.bar(["building_id", "meter", "site_id", "primary_use", "square_feet"], lin_reg.coef_)
label_encoder = preprocessing.LabelEncoder()



df_train_building_left["primary_use"] = label_encoder.fit_transform(df_train_building_left["primary_use"])

df_test_building_left["primary_use"] = label_encoder.transform(df_test_building_left["primary_use"])



final_X_train = df_train_building_left[["building_id", "meter", "site_id", "primary_use", "square_feet"]]

final_X_test = df_test_building_left[["building_id", "meter", "site_id", "primary_use", "square_feet"]]



final_y_train = df_train_building_left["meter_reading"]



standard_scaler = preprocessing.StandardScaler().fit(final_X_train)

final_X_train = standard_scaler.transform(final_X_train)

final_X_test = standard_scaler.transform(final_X_test)



lin_reg = LinearRegression()

lin_reg.fit(final_X_train, final_y_train.values)

y_pred_lr = lin_reg.predict(final_X_test)



submission = pd.DataFrame({'row_id':df_test_building_left['row_id'], 'meter_reading':y_pred_lr})



submission.to_csv("ashrae_prediction.csv",index=False)