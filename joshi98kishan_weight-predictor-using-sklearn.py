import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_absolute_error
df = pd.read_csv('../input/weight-height.csv')



# One hot encoding a nominal feature -> 'Gender'

ohe = OneHotEncoder(sparse = False)



arr = ohe.fit_transform(df.Gender.values.reshape(-1, 1)) # Returns array of encoded feature



new_features_name = ohe.categories_[0].tolist() # i.e. 'Female' and 'Male'



df_encoded = pd.DataFrame(data = arr, columns = new_features_name)



# Adding 'Height' and 'Weight' columns to this new dataframe

df_encoded[['Height', 'Weight']] = df.iloc[:, 1:3]

df_encoded.head()
# Analysing relation between every pair of feature

pd.plotting.scatter_matrix(df_encoded, alpha = 0.3, figsize=(10, 10))

plt.show()
#splitting dataset into 80% training and 20% testing dataset

df_encoded_train, df_encoded_test = train_test_split(df_encoded, test_size = 0.2)
lr = LinearRegression()



features_train = df_encoded_train.drop(columns = 'Weight')

target_train = df_encoded_train.Weight



features_test = df_encoded_test.drop(columns = 'Weight')

target_test = df_encoded_test.Weight



lr.fit(features_train, target_train)

coefficients = lr.coef_



print(f'Coefficient of Female is {coefficients[0]}')

print(f'Coefficient of Male is {coefficients[1]}')

print(f'Coefficient of Height is {coefficients[2]}')
y_pred = lr.predict(features_test)



# R2 Score

score = r2_score(target_test, y_pred)

print('R2 score of this model is ', score)
# Mean Absolute Error

mae = mean_absolute_error(target_test, y_pred)

print('Mean Absolte Error of this model is ', mae)