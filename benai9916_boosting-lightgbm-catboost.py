import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
# Import Data

census_df = pd.read_csv('../input/adult-census-income/adult.csv')
# size of data
census_df.shape
# Show first 5 rows

census_df.head(5)
# change columns name
cols = census_df.columns
census_df.columns = cols.str.replace('.','_')
# print some information about data

census_df.info()
# check for missing values
census_df.isna().sum()
# some columns have '?' 

ques = census_df == '?'
ques.sum()
# check  workclass and occupation both column contain question mark 
compare = census_df[census_df[['workclass', 'occupation']] == '?'].any(axis=1)
census_df[compare]
# check with groupby
census_df.groupby(by='workclass')['hours_per_week'].mean()
# check the values in workclass 
census_df.workclass.value_counts()
# will eliminte '?' rows
df = census_df[census_df.occupation !='?']

ques = df == '?'
ques.sum()
# check the most fequent value in native_country columns

df.native_country.max()
# replace country with the most frequent value
df.native_country = df.loc[:, ['native_country']].replace('?', 'Yugoslavia')
df.native_country.unique()
# # print some stastical about data
census_df.describe()
# pair plot
sns.pairplot(census_df)
# check the count of target variable
sns.countplot(df.income)
# count plot against income
cols = ['workclass', 'relationship', 'marital_status', 'race', 'sex']

for c in cols:
    plt.figure(figsize=(12,4))
    sns.countplot(x=c, hue='income', data=df)
    plt.show()
# check for outliers
fig, (a,b)= plt.subplots(1,2,figsize=(20,6))
sns.boxplot(y='hours_per_week',x='income',data=df,ax=a)
sns.boxplot(y='age',x='income',data=df,ax=b)
plt.show()
#  Correlation Matrix with Spearman method

plt.figure(figsize=(15,15))

sns.heatmap(df.corr(), annot=True, cmap='BrBG', vmin=-1, vmax=1, center= 0,
            square=True, linewidths=2, cbar_kws={"shrink": .5}).set(ylim=(15, 0))
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# transform categorical value to numeric
for i in df.columns:
    df[i] = le.fit_transform(df[i])
df.head()
x = df.iloc[:, :-1]
y = df['income']
# train and test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)
%time
# create an object
lgb = LGBMClassifier(learning_rate= 0.1,
 boosting_type= 'gbdt',
 objective= 'binary',
 metric= 'binary_logloss',
 sub_feature= 0.5,
 num_leaves= 8,
 min_data= 50,
 max_depth= 15)


# fit the data

d_train = lgb.fit(x_train, y_train)
# predict the target on the train dataset

y_pred_lgb = lgb.predict(x_test)


#Accuracy
accuracy = accuracy_score(y_pred_lgb,y_test)
print('Accuracy Score : ', round(accuracy, 2) * 100)

#Confusion matrix
cm = confusion_matrix(y_test, y_pred_lgb)
print('\nConfusion matric :\n ', cm)
# find out the indices of categorical variables
categorical_var = np.where(x_train.dtypes != np.float)[0]
print('\nCategorical Variables indices : ',categorical_var)
%time
# create an object
cb = CatBoostClassifier(iterations=90, learning_rate=0.7, logging_level='Silent')

# fit the model
cb.fit(x_train, y_train, cat_features=categorical_var, eval_set=(x_test, y_test),verbose=False)
# predict
y_predict_cb = cb.predict(x_test)
print('\nTarget on train data',y_predict_cb) 

# Accuray Score on train dataset
accuracy_train_cb = accuracy_score(y_predict_cb, y_test)
print('\naccuracy_score on train dataset : ', round(accuracy_train_cb,2)* 100)

#Confusion matrix
cm = confusion_matrix(y_test, y_predict_cb)
print('\nConfusion matric :\n ', cm)
