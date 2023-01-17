import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



print(train_data.shape)

print(test_data.shape)
df_train = pd.DataFrame(train_data)

df_test = pd.DataFrame(test_data)
df_train.head()
cols_missed_train = df_train.isnull().sum()

cols_missed_test = df_test.isnull().sum()



print('Missed values in train dataset:')

print(cols_missed_train[cols_missed_train > 0])

print('Missed values in test dataset:')

print(cols_missed_test[cols_missed_test > 0])
print("Total missing values in train and test:", (cols_missed_train.sum().sum()+cols_missed_test.sum().sum()))

print("Train dataset missing value percentage:", df_train.isnull().sum()*100/len(df_train))

print("Test dataset missing value percentage:", df_test.isnull().sum()*100/len(df_test))
df_train_copy = df_train.copy()

df_test_copy = df_test.copy()
# Replacing missing values in the train dataset

df_train_copy['Age'] = df_train_copy['Age'].fillna(df_train_copy['Age']).mode()[0]

df_train_copy['Cabin'] = df_train_copy['Cabin'].fillna('None')

df_train_copy['Embarked'] = df_train_copy['Embarked'].fillna(df_train_copy['Embarked']).mode()[0]



print('Missing values in the train dataset: ', df_train_copy.isnull().sum().sum())



# Replacing missing values in the test dataset

df_test_copy['Age'] = df_test_copy['Age'].fillna(df_test_copy['Age']).mode()[0]

df_test_copy['Fare'] = df_test_copy['Fare'].fillna(df_test_copy['Fare']).mode()[0]

df_test_copy['Cabin'] = df_test_copy['Cabin'].fillna('None')



print('missing values in the test dataset: ', df_test_copy.isnull().sum().sum())
df_train_copy.dtypes.value_counts()
df_train_full = df_train_copy.copy()

df_test_full = df_test_copy.copy()
train_num = df_train_full.select_dtypes(exclude='object')
#Histogram for Survived

sns.distplot(train_num['Survived'], color='green')
# Histogram of SalePrice depending on MSZoning (normalized)

df_train_full.groupby('Pclass')['Survived'].plot.hist(density=1, alpha=0.6)

plt.title('Distribution by Pclass')

plt.legend()
# Histogram of SalePrice depending on MSZoning (normalized)

df_train_full.groupby('Sex')['Survived'].plot.hist(density=1, alpha=0.6)

plt.title('Distribution by Sex')

plt.legend()
test_id = df_test['PassengerId']
df_train_full.drop(columns='PassengerId', axis=1, inplace=True)

df_test_full.drop(columns='PassengerId', axis=1, inplace=True)
n_df = df_train_full.copy()

corr_matrix = n_df.corr()



mask = np.zeros_like(corr_matrix, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

colormap = sns.diverging_palette(220, 10, as_cmap=True)



f, ax = plt.subplots(figsize=(8,6))

plt.title('Correlation matrix', fontsize=16)

sns.heatmap(corr_matrix, mask=mask, square=True, cmap=colormap, vmax=1, center=0, annot=True, fmt='.1f')
survived = df_train_full['Survived']

new_cols_train = df_train_full.drop(columns='Survived', axis=1)

new_cols_train.head()
y = survived

X = new_cols_train

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)



print(train_X.shape)

print(test_X.shape)
s = (train_X.dtypes == 'object')

categors = list(s[s].index)



label_train_X = train_X.copy()

label_test_X = test_X.copy()



OHE = OneHotEncoder(handle_unknown='ignore', sparse=False)

OHE_train_X = pd.DataFrame(OHE.fit_transform(label_train_X[categors]))

OHE_test_X = pd.DataFrame(OHE.transform(label_test_X[categors]))



OHE_train_X.index = label_train_X.index

OHE_test_X.index = label_test_X.index



num_train_X = label_train_X.drop(categors, axis=1)

num_test_X = label_test_X.drop(categors, axis=1)



OHE_train = pd.concat([num_train_X, OHE_train_X], axis=1)

OHE_test = pd.concat([num_test_X, OHE_test_X], axis=1)
xgb = XGBRegressor(n_estimator=1000, learning_rate=0.1)

xgb.fit(OHE_train, train_y,

       early_stopping_rounds=5,

       eval_set=[(OHE_test, test_y)],

       verbose=False)



val_pred = xgb.predict(OHE_test)



rmse = mean_squared_error(test_y, val_pred)

rmse
xgb.score(OHE_train, train_y)
u = (df_test_full.dtypes == 'object')

categor_test = list(u[u].index)



OHE_test_valid = pd.DataFrame(OHE.transform(df_test_full[categor_test]))

OHE_test_valid.index = df_test_full.index

num_test = df_test_full.drop(categor_test, axis=1)



OHE_test_final = pd.concat([num_test, OHE_test_valid], axis=1)
final_pred = xgb.predict(OHE_test_final)
df_pred = pd.DataFrame({"PassengerId":test_id, "Survived":final_pred})

df_pred.Survived = df_pred.Survived.round().astype("int")

df_pred.to_csv('submission.csv', sep=',', encoding='utf-8', index=False)