# import the analysis and visualization libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')
# load the dataset

insurance = pd.read_csv('../input/insurance/insurance.csv')
# check the head of the dataset

insurance.head(5)
# check the info of the dataset

insurance.info()
# check a few statistics of the dataset

insurance.describe()
# visualise the number of people to their relative age groups as a histogram

plt.figure(figsize=(9,5))

plt.hist(insurance['age'],bins=20)

plt.xlabel('Age')

plt.ylabel('Num of people')

# Shows the biggest age group category to be around 20
# People aged 18 and 19 were represented more than any other age by over a factor of 2.

insurance['age'].value_counts().head(5)
# Visualizing the bmi of each sex and seperating them into group of smokers and non smokers

sns.boxplot(data=insurance, x='sex',y='bmi',hue='smoker')
# Regression line indicating a small correlation between bmi and their medical costs

sns.lmplot(data=insurance,x='bmi',y='charges',aspect=2)
# Visualizing the relationship of all the features

sns.pairplot(insurance)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

model = LinearRegression()
# Our aim is to predict the charges feature

# To create a better fit, we need to change the string variable into machine readable intergers.

insurance['sex'] = insurance['sex'].apply(lambda x : 1 if x=='male' else 0)
insurance.rename(columns={'sex':'male'},inplace=True)
insurance['smoker'] = insurance['smoker'].apply(lambda x : 1 if x=='yes' else 0)
regions = pd.get_dummies(insurance['region'],drop_first=True)
insurance_df = pd.concat([insurance,regions],axis=1)
insurance_df.drop('region',axis=1,inplace=True)
# We can now start splitting our data into train/test sets

X_train, X_test, y_train, y_test = train_test_split(insurance_df.drop('charges',axis=1),insurance['charges'],test_size=0.33)
# Fit our training data to the our model

model.fit(X_train,y_train)
# Make the predictions using the X_test

predictions = model.predict(X_test)
# Let's check how well the model performed

from sklearn.metrics import mean_squared_error
print('RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))
plt.figure(figsize=(10,7))

plt.hist(y_test-predictions,bins=20)
coef_df = pd.DataFrame(model.coef_,columns=['Coef'],index = ['age','male','bmi','children','smoker','northwest','southeast','southwest'])
coef_df