import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

df=pd.read_csv("C:/Users/sid/Desktop/breast_cancer.csv")

df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
df['diagnosis'].head(10)
df=pd.get_dummies(df,'diagnosis',drop_first=True)

df.head()
sns.scatterplot(x='texture_worst',y='smoothness_mean',data=df)

plt.show()
sns.regplot(x='radius_mean',y='perimeter_mean',data=df,color='g')

plt.show()
sns.boxplot(x='compactness_mean',y='area_mean',data=df)

plt.show()
plt.figure(figsize=(25,20))

sns.pairplot(df)
#lets chk the value counts of 2 variables

df['compactness_mean'].value_counts().head().plot(kind='bar',color='g')
df.corr()
df.std()
plt.figure(figsize =(20,6))

sns.barplot(x='radius_mean',y='texture_mean',data =df,palette='viridis')

plt.xlabel('Mean Radius of the lump')

plt.ylabel('Texture of the lump')

plt.figure(figsize =(20,6))

sns.barplot(x='perimeter_worst',y='area_worst',data =df, hue= 'diagnosis_M')
plt.figure(figsize=(20,15),dpi=100)

sns.heatmap(df.corr(),annot=True)
import sklearn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
#splitting the train and test data

df_train,df_test=train_test_split(df,train_size=0.7,random_state=100)

print(df_train.shape)

print(df_test.shape)
#intiate the object

scaler=MinMaxScaler()

#create the list of numeric variables

num_vars=['radius_mean','diagnosis_M','fractal_dimension_worst','symmetry_worst','concave points_worst',

          'area_se']

df_train[num_vars]=scaler.fit_transform(df_train[num_vars])

df_train.head()
#visulizing the data with train data

plt.figure(figsize=(15,12))

sns.heatmap(df_train.corr(),annot=True)
sns.boxplot(x='diagnosis_M',y='fractal_dimension_worst',data=df_train)

plt.show()
df_train.head()
#define x,y

y_train=df_train.pop('diagnosis_M')

X_train=df_train

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
#Running RFE with the output number of the variable equal to 15
lm=LinearRegression()

lm.fit=X_train,y_train

rfe = RFE(lm, 15)             # running RFE

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]

col
X_train.columns[~rfe.support_]
# Creating X_train dataframe with RFE selected variables

X_train_rfe = X_train[col]
# Adding a constant variable





X_train_rfe = sm.add_constant(X_train_rfe)



# Running the linear model

lm = sm.OLS(y_train,X_train_rfe).fit()
# Summary of our linear model

print(lm.summary())
from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_rfe

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_pred = lm.predict(X_train_rfe)
# Plotting the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_pred), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)                         # X-label
plt.scatter(y_train,(y_train - y_train_pred))

plt.show()
#applying scaling on test set

df_test[num_vars] = scaler.transform(df_test[num_vars])
df_test.describe()
#diving X_test and y_test

y_test=df_test.pop('diagnosis_M')

X_test=df_test

# Creating X_test_new dataframe by dropping variables from X_test

#X_test_new = X_test[X_train_rfe.columns]

X_test_new = X_test[col]



# Adding a constant variable 

X_test_new = sm.add_constant(X_test_new)
# Making predictions

y_test_pred = lm.predict(X_test_new)
# Plotting y_test and y_test_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_test_pred)

fig.suptitle('y_test vs y_test_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                               # X-label

plt.ylabel('y_test_pred', fontsize=16)                          # Y-label