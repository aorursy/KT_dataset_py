#importing data visualization and manipulation libraries



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns



#importing machine learning libraries



from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.model_selection import train_test_split
#importing dataset



df = pd.read_csv("/kaggle/input/co2-emission-by-vehicles/CO2 Emissions_Canada.csv")
#checking for null values, didn't expect any



df.isnull().sum()
#I chose to rename this column to something easier to type as it is used very frequently 



df.rename(columns={'CO2 Emissions(g/km)' : 'CO2_emission'}, inplace=True)
#updated dataset



df
#getting to know the dataset a little more in the next few steps



df['Fuel Type'].value_counts()
df['Transmission'].value_counts()
#discovering correlation



df.corr()['CO2_emission'].sort_values()
#heatmap for a better understanding of correlated values



plt.figure(figsize = (8,6))

corr = df.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap = 'Blues', square = True)
#I have a habit of using pairplot function of seaborn to see how each individual graph looks like



sns.pairplot(df)
#Some visualizations to show our understanding of the dataset



mkI = df['Make'].value_counts().index

mkV = df['Make'].value_counts().values

plt.figure(figsize = (10,8))

sns.barplot(mkI,mkV)

plt.xticks(rotation='vertical')
mkI = df['Vehicle Class'].value_counts().index

mkV = df['Vehicle Class'].value_counts().values

plt.figure(figsize = (10,8))

sns.barplot(mkV,mkI, orient = 'h', palette='Spectral')

plt.xticks(rotation='vertical')
#this boxplot shows us that Vans typically emit more CO2 when compared to other vehicle classes



plt.figure(figsize = (10,8))

sns.boxplot(x="Vehicle Class", y="CO2_emission", data=df)

plt.xticks(rotation = 'vertical')
sns.boxplot(df['Fuel Consumption City (L/100 km)'], color = "red")

plt.show()

sns.boxplot(df['Fuel Consumption Hwy (L/100 km)'])

plt.show()

sns.boxplot(df['Fuel Consumption Comb (L/100 km)'], color = 'green')

plt.show()
plt.figure(figsize = (10,8))

sns.boxplot(x = 'Fuel Type' , y = 'CO2_emission', data = df)

plt.xticks([0,1,2,3,4],['Premium Gasoline','Diesel','Regular Gasoline','Ethanol','Natural Gas'])

plt.show()
plt.figure(figsize = (10,8))

sns.catplot(x = 'Cylinders' , y = 'CO2_emission', data = df)

plt.show()
#Ethanol typically is the most efficient fuel type 



plt.figure(figsize = (10,8))

sns.boxplot(y = 'Fuel Consumption Comb (mpg)', x = 'Fuel Type', data = df, palette = 'muted')

plt.xticks([0,1,2,3,4],['Premium Gasoline','Diesel','Regular Gasoline','Ethanol','Natural Gas'])
plt.figure(figsize = (10,8))

sns.distplot(df['Fuel Consumption Comb (mpg)'], bins = 10, color = 'purple')
#assigning dependent and independent variables

#can be used with any column across the dataset provided hyperparameters are adjusted accordingly



x = df.iloc[:, -3].values

y = df.iloc[:, -1].values
#splitting and reshaping data into testing and training sets



xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.2, random_state = 0)



xTrain= xTrain.reshape(-1, 1)

yTrain= yTrain.reshape(-1, 1)

xTest = xTest.reshape(-1, 1)

yTest = yTest.reshape(-1, 1)
#linear regression model achieving 85% accuracy

#at the end of the kernel I attempted to create and use my own linear regression model to find out coefficient and intercept without using scikit learn



reg = LinearRegression()

reg.fit(xTrain, yTrain)

regYpred = reg.predict(xTest)

print(reg.score(xTest,yTest))
#I printed the coefficient and the intercept here to compare my model built from scratch against the imported scikit learn model



print('regression coefficient', reg.coef_, 'intercept', reg.intercept_)
f, ax = plt.subplots(1, figsize=(10, 8), sharex=True)



sns.stripplot(y = yTest.flatten(), color = 'darkmagenta', alpha = 0.7, label = 'Test Data')

sns.stripplot(y = regYpred.flatten(), color = 'lawngreen', alpha = 0.7, label = 'Train Data')

plt.legend()

plt.show()
#I used these histograms to show Predicted values vs. Actual values in all three models



sns.distplot(regYpred, bins = 20, color = 'red')

plt.title = 'Predicted values'

plt.show()

sns.distplot(df['CO2_emission'], bins = 20)

plt.title = 'Actual values'

plt.show()
#Regression line showing best fit



sns.regplot(x = 'Fuel Consumption Comb (L/100 km)', y = 'CO2_emission', data  = df, color = 'blue')
#Decision Tree model got us a higher accuracy at 88%



from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state = 0)

dtr.fit(xTrain, yTrain)

dtrYpred = dtr.predict(xTest)

dtrScore = r2_score(yTest,dtrYpred)

print('R2 test data: %.3f' % dtrScore)
f, ax = plt.subplots(1, figsize=(10, 8), sharex=True)



sns.stripplot(y = yTest.flatten(), color = 'darkmagenta', alpha = 0.7, label = 'Test Data')

sns.stripplot(y = dtrYpred.flatten(), color = 'lawngreen', alpha = 0.7, label = 'Train Data')

plt.legend()

plt.show()
sns.distplot(dtrYpred, bins = 20, color = 'red')

plt.show()

sns.distplot(df['CO2_emission'], bins = 20)

plt.show()
#Random Forest Regressor had the highest accuracy standing at 89%

#I used a for loop for the n estimators to see which yielded the highest accuracy, it landed at 20



from sklearn.ensemble import RandomForestRegressor



rfr = RandomForestRegressor(n_estimators = 20, random_state = 0)

rfr.fit(xTrain, yTrain)

rfrYpred = rfr.predict(xTest)

rfrScore = r2_score(yTest,rfrYpred)

print('R2 test data: %.3f' % rfrScore)
f, ax = plt.subplots(1, figsize=(10, 8), sharex=True)



sns.stripplot(y = yTest.flatten(), color = 'darkmagenta', alpha = 0.7, label = 'Test Data')

sns.stripplot(y = regYpred.flatten(), color = 'lawngreen', alpha = 0.7, label = 'Train Data')

plt.legend()

plt.show()
sns.distplot(rfrYpred, bins = 20, color = 'red')

plt.show()

sns.distplot(df['CO2_emission'], bins = 20)

plt.show()
#calculating mean of x and y values



X,Y = xTrain,yTrain

xMean = np.mean(X)

yMean = np.mean(Y)
#calculating variance and covariance



covar = 0

var = 0

for i in range (len(X)):

    covar += (X[i] - xMean) * (Y[i] - yMean)

    var += (X[i]-xMean) ** 2      
#computing coefficient and intercepts based on previous calculations



coeff = covar/var

intercept = yMean - (coeff * xMean)



print('intercept is',intercept, 'coefficient is', coeff)