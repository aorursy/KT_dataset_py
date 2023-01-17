# import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Import datase

health_data = pd.read_csv('../input/insurance/insurance.csv')

health_data.head()
health_data.describe()
health_data.dtypes
health_data.isnull().sum()
health_data.shape
plt.figure(figsize = (12,8))

sns.set(style = 'whitegrid')

sns.distplot(health_data['charges'], kde=True)

plt.title('Total Charges Distribution')
plt.figure(figsize = (12,8))

sns.set(style = 'whitegrid')

sns.distplot(np.log10(health_data['charges']), kde=True, color = 'g')

plt.title('Total Charges Distribution - After Applying Log')
## Check distribution of data for Smoker and Non-Smokers



f = plt.figure(figsize=(12,8))



ax = f.add_subplot(121)

sns.distplot(health_data[(health_data.smoker == 'yes')]["charges"],color='r',ax=ax)

ax.set_title("Distribution Charges for Smokers")



ax = f.add_subplot(122)

sns.distplot(health_data[(health_data.smoker == 'no')]["charges"],color='g',ax=ax)

ax.set_title("Distribution Charges for Non-Smokers")

f, ax = plt.subplots(1, 1, figsize=(10, 5))

ax = sns.countplot(x='smoker', hue='sex', data=health_data, palette='cool')
plt.figure(figsize= (10,5))

plt.title("Age Distribution")

ax = sns.distplot(health_data["age"], color = 'r')
sns.catplot(x="smoker", kind="count",hue = 'sex', palette="rainbow", data=health_data[(health_data.age >= 18) & (health_data.age <= 20)])

plt.title("The number of smokers and non-smokers (18-20 years old)")
plt.figure(figsize = (10,5))

plt.title('Charges for 18-20 Age Patients who smoke')

sns.boxplot(x='charges', y='smoker', data=health_data[(health_data.age >= 18) & (health_data.age <= 20)])
plt.figure(figsize= (10,5))

plt.title("BMI distribution")

ax = sns.distplot(health_data["bmi"], color = 'g')
plt.figure(figsize=(12,5))

plt.title("Distribution of charges for patients with BMI greater than 25")

ax = sns.distplot(health_data[(health_data.bmi >= 25)]['charges'], color = 'm')
plt.figure(figsize=(12,5))

plt.title("Distribution of charges for patients with BMI greater than 25")

ax = sns.distplot(health_data[(health_data.bmi <= 25)]['charges'], color = 'm')
ax =sns.lmplot(x = 'age', y='charges', data = health_data, hue = 'smoker')
ax =sns.lmplot(x = 'bmi', y='charges', data = health_data, hue = 'smoker')
ax =sns.lmplot(x = 'children', y='charges', data = health_data, hue = 'smoker')
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



#Sex

le.fit(health_data['sex'].drop_duplicates())

health_data['sex'] = le.transform(health_data['sex'])



#smoker

le.fit(health_data['smoker'].drop_duplicates())

health_data['smoker'] = le.transform(health_data['smoker'])



#region

le.fit(health_data['region'].drop_duplicates())

health_data['region'] = le.transform(health_data['region'])

health_data.dtypes
plt.figure(figsize=(10,8))

sns.heatmap(health_data.corr(), annot = True, cmap = 'YlGnBu')
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
X = health_data.drop(['charges'], axis =1)

y = health_data['charges'] 



X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=42)
lr = LinearRegression()

lr_model = lr.fit(X_train, y_train)



#Prediction on train data

y_train_pred = lr_model.predict(X_train)



#Prediction on test data

y_test_pred = lr_model.predict(X_test)
# Accuracy Score

print(lr_model.score(X_test, y_test))
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import r2_score,mean_squared_error
polynomial = PolynomialFeatures(degree=2)

polynomial_model = polynomial.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(polynomial_model,y, train_size=0.7, random_state=42)



# Build second LR model using polynomial features

lr_model_2 = LinearRegression().fit(X_train,y_train)





#PRedict the values

y_train_pred = lr_model_2.predict(X_train)



#Predict test values

y_test_pred = lr_model_2.predict(X_test)
print(lr_model_2.score(X_test, y_test))
polynomial_LR_model = pd.DataFrame({'Actual Values': y_test, 'Predicted Values':y_test_pred})
from sklearn.ensemble import RandomForestRegressor
rfe = RandomForestRegressor(n_estimators =100,

                           criterion = 'mse',

                           random_state =42,

                           n_jobs=-1)
rfe.fit(X_train, y_train)



# predict train data

y_pred_train = rfe.predict(X_train)



# predict test data 

y_test_pred = rfe.predict(X_test)

print('R2 train data: %.3f, R2 test data: %.3f' % (

r2_score(y_train,y_pred_train),

r2_score(y_test,y_test_pred)))
polynomial_LR_model.head(10)