

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

import statsmodels.api as sm

import sklearn





df=pd.read_csv('../input/housesalesprediction/kc_house_data.csv')

df['date'] = pd.to_datetime(df['date'])

df.set_index('id', inplace=True)
#df.describe()
#Variable 'price', 'sqft_living' values are really large and will affect the absolute numbers of the regression model. 

#I will normalise the data using log



df['price'] = np.log(df['price'])

df['sqft_living'] = np.log(df['sqft_living'])







df.describe()

#check for missing or null values in the data set. Looks like everything is in place.
null= df.isnull()
sns.heatmap(null,cbar=False,cmap='viridis',yticklabels=False)

#there is no missing values
#I created a coorelation matrix to check which variables are strongerly correlated with the target variable 'price'
plt.subplots(figsize=(10,8))

sns.heatmap(df.corr(method='pearson'),annot=True,linecolor="black",cmap='coolwarm',fmt="1.1f", linewidths=0.25, vmax=1.0, square=True)

plt.title("Data Correlation",fontsize=50)

plt.show()

#Data Visulations 
#Let's have a look at the different variables and their relation with the target variable 'price'
filter = ['sqft_living','sqft_above','sqft_living15', 'sqft_lot15']

sns.pairplot(data=df, x_vars=['sqft_living','sqft_above','sqft_living15', 'sqft_lot15'],y_vars='price',kind='scatter')

plt.show()
#sqft_living has a stronger correlation with the price as compare to other variables. Lets focus on sqft_living
plt.figure(figsize = (8, 5))

sns.jointplot(x='sqft_living', y='price',data=df, 

              alpha = 0.5,)

plt.xlabel('Sqft Living')

plt.ylabel('Sale Price')

plt.show()
#Let's check the 'zipcode' variable and important zip code is on sqft_living and price relation
df["zipcode"].nunique()
df['zipcode'].value_counts()
#zip code 98103 has most of houses sold 
df.groupby('zipcode')['price'].mean().reset_index().sort_values('price',ascending=False)
#Most expensive zipcode is 98039
#let's plot sqft_living  for both the zipcodes
zip98103 = df['zipcode'] == 98103 

zip98039 = df['zipcode'] == 98039
plt.figure(figsize = (8, 5))

sns.jointplot(x='sqft_living', y='price',data=df[zip98103], 

              alpha = 0.5,)

plt.xlabel('Sqft Living')

plt.ylabel('Sale Price')

plt.show()
#Plot of sqft_living vs pirce for zipcode 98103 is almost similar with the original plot for all zipcode. 
plt.figure(figsize = (8, 5))

sns.jointplot(x='sqft_living', y='price',data=df[zip98039], 

              alpha = 0.5,)

plt.xlabel('Sqft Living')

plt.ylabel('Sale Price')

plt.show()
#Zipcode 98039 has an interesting plot. Sqft_living vs price for this zipcode has a strong positive correlation. 

# we will only include zipcode 98039 in our prediction model to get a better result
f, axes = plt.subplots(1, 2, figsize=(25,5))

sns.countplot(x='bedrooms' , data=df, ax=axes[1])

sns.boxplot(x='bedrooms', y='price', data=df, ax=axes[0])
f, axes = plt.subplots(1, 2, figsize=(25,5))

sns.countplot(x='bathrooms' , data=df, ax=axes[1])

sns.boxplot(x='bathrooms', y='price', data=df, ax=axes[0])
f, axes = plt.subplots(1, 2, figsize=(25,5))

sns.countplot(x='grade' , data=df, ax=axes[1])

sns.boxplot(x='grade', y='price', data=df, ax=axes[0])
f, axes = plt.subplots(1, 1, figsize=(5,5))

sns.boxplot(x='waterfront',y='price' , data=df)

plt.show()

sns.boxplot(x='view', y='price', data=df)
#Let's check the 'zipcode' variable and how it is correlated with the 'price'
plt.figure(figsize = (8, 5))

sns.jointplot(y='long', x='price',data=df, 

              alpha = 0.5,)

plt.xlabel('Sqft Living')

plt.ylabel('Sale Price')

plt.show()
# HOUSE PREDICTIONS
#Now we will use some model to Predict house prices.
#First we try to run Linear Regression model to predict the prices



features1 = ['sqft_living','grade', 'bathrooms','sqft_above','sqft_living15','lat','sqft_lot15']

features2= ['sqft_living','grade', 'bathrooms','sqft_above','sqft_living15','lat','view','bedrooms','condition']

features3 =['sqft_living','grade', 'bathrooms','sqft_above','sqft_living15','lat','view','bedrooms','condition','yr_built','sqft_lot15','floors','waterfront','zipcode']



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression













x= df[features1]

y=df['price']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=100)
lm=LinearRegression()
lm.fit(X_train, y_train)
Score1=lm.score(X_test, y_test)
round(Score1,2)
# The predicition score for feature 1 is 71% which is weak so we will try feature 2.
x= df[features2]

y=df['price']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=100)
lm=LinearRegression()
lm.fit(X_train, y_train)
Score2=lm.score(X_test, y_test)
round(Score2,2)
# The predicition score for feature 2 is 74% . Let see what will be the score with feature3?
x= df[features3]

y=df['price']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=100)
lm=LinearRegression()
lm.fit(X_train, y_train)
Score3=lm.score(X_test, y_test)
round(Score3,2)
# The predicition score for feature 3 is 77% which is stronger than previous two features so we will use this for predicition.
print(lm.intercept_)
lm.coef_
pd.DataFrame(lm.coef_,x.columns, columns=['coef'])
Prediction1=lm.predict(X_test)
Prediction1
y_test
from sklearn import metrics

round(metrics.mean_absolute_error(y_test, Prediction1),2)
round(metrics.mean_squared_error(y_test, Prediction1),2)
round(np.sqrt(metrics.mean_squared_error(y_test, Prediction1)),2)
fig, ax = plt.subplots()

ax.scatter(y_test, Prediction1)

ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)

ax.set_xlabel('Price')

ax.set_ylabel('Predicted')

plt.show()

plt.rcParams['figure.figsize'] = (25,5)
sns.distplot((y_test-Prediction1))
compare = pd.DataFrame({'Prediction': Prediction1, 'Test Data' : y_test})

compare.head(10)