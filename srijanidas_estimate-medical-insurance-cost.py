import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
med_insurance_df = pd.read_csv("../input/insurance/insurance.csv")
med_insurance_df.head(2)
med_insurance_df.info()
med_insurance_df['region'].unique()
med_insurance_df.groupby('region').max()['charges']
med_insurance_df.groupby('region').mean()['charges']
pd.get_dummies(med_insurance_df['region'])
med_insurance_df['northeast_region'] = pd.get_dummies(med_insurance_df['region'])['northeast']

med_insurance_df['southeast_region'] = pd.get_dummies(med_insurance_df['region'])['southeast']

med_insurance_df['male'] = pd.get_dummies(med_insurance_df['sex'])['male']

med_insurance_df['smoker'] = pd.get_dummies(med_insurance_df['smoker'])['yes']

med_insurance_df.head(2)
# It is obvious that females have lower BMI than male, the output supports the fact



print("Maximun ",med_insurance_df.groupby('male').max()['bmi'])

print("Mean ",med_insurance_df.groupby('male').mean()['bmi'])
print("Maximun ",med_insurance_df.groupby('smoker').max()['bmi'])

print("Mean ",med_insurance_df.groupby('smoker').mean()['bmi'])
print("Maximun ",med_insurance_df.groupby('male').max()['charges'])

print("Mean ",med_insurance_df.groupby('male').mean()['charges'])
# Smokers have more medical charges than non-smokers



print("Maximun ",med_insurance_df.groupby('smoker').max()['charges'])

print("Mean ",med_insurance_df.groupby('smoker').mean()['charges'])
plt.figure(figsize=(10,6))

sns.heatmap(med_insurance_df.corr(), annot=True, cmap='viridis')
sns.scatterplot(x='age', y='charges', data=med_insurance_df, hue='smoker')
sns.scatterplot(x='age', y='charges', data=med_insurance_df, hue='male')
sns.countplot(x='smoker', data=med_insurance_df, hue='male')
sns.countplot(x='smoker', data=med_insurance_df, hue='region')
sns.countplot(x='male', data=med_insurance_df)
med_insurance_df.columns
for col in ['sex', 'children', 'region', 'male']:

  if col in med_insurance_df.columns:

    med_insurance_df.drop(col, axis=1, inplace=True)
X = med_insurance_df.drop('charges', axis=1)

y = med_insurance_df['charges']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)



pred = lm.predict(X_test)
print(lm.intercept_)
df = pd.DataFrame(lm.coef_, index=X.columns, columns=['Coefficient'])

df
plt.scatter(y_test, pred)

plt.xlabel("y_test")

plt.ylabel("pred")

plt.title("True value vs. Predicted")
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, pred))

print('MSE:', metrics.mean_squared_error(y_test, pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))