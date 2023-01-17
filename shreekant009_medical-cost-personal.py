import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/insurance/insurance.csv')
df.head()
df.describe()
df.isnull().sum()
df_male = df[df["sex"] == "male"]
df_female = df[df["sex"] == "female"]
#Male who are smoking with their Insurance Charge

g = sns.FacetGrid(df_male,  row="smoker")

g = g.map(plt.hist, "charges")
#Female who are smoking with their Insurance Charge

g = sns.FacetGrid(df_female,  row="smoker")

g = g.map(plt.hist, "charges")
#Checking Male and Female Smaoking at what age

sns.boxplot(x='age',y='sex',hue='smoker',data=df)
#Checking bmi with charges(smoker with bmi between 30 to 50 have high charges)

sns.lmplot(x='charges', y='bmi',hue='smoker',data=df,palette='coolwarm')
#Smoker and NO Smoker with childrens

g = sns.FacetGrid(df, col="children",  row="smoker")

g = g.map(plt.hist, "age")
df['region'].unique()
#Region + Age wise people getting insurance

g = sns.FacetGrid(df, col="region",  row="smoker",hue='sex')

# Notice hwo the arguments come after plt.scatter call

g = g.map(plt.scatter, "charges","age").add_legend()
#As Region doesn't play role in Insurances redemtion so we will drop it.

df = df.drop(labels=['region'],axis=1)
df.head()
#Using Label Encoder on labels=sex,smoker,children

from sklearn import preprocessing 

label_encoder = preprocessing.LabelEncoder()

df['sex'] = label_encoder.fit_transform(df['sex'])

df['smoker'] = label_encoder.fit_transform(df['smoker'])

df['children'] = label_encoder.fit_transform(df['children'])
X = df[['age','sex','bmi','children','smoker']]

y = df['charges']
#Train Test

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=101)
#Creating training Model

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)
#Predictions

predictions = lm.predict(X_test)
sns.distplot((y_test-predictions),bins=50);
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#calculating r squared

SS_Residual = sum((y_test-predictions)**2)

SS_Total = sum((y_test-np.mean(y_test))**2)

r_squared = 1 - (float(SS_Residual))/SS_Total

print('R Squared:', r_squared)
#regression plot of the real test values versus the predicted values



plt.figure(figsize=(16,8))

sns.regplot(y_test,predictions)

plt.xlabel('Predictions')

plt.ylabel('Actual')

plt.title("Linear Model Predictions")

plt.grid(False)

plt.show()