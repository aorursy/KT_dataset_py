%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

# generate related variables

from numpy import mean

from numpy import std

from numpy.random import randn

from numpy.random import seed

from numpy import cov





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Question 1

data = pd.read_csv("../input/dwdm-dirt-lab-1/DWDM Dirt Lab 1.csv",encoding='ISO-8859-1')

data.head(10)
# Question 2

data['Region'].value_counts().sort_values(ascending=False)

data['Region'].value_counts()[2:3].sort_values(ascending=False)
# Question 3

region_df = data['Region'].value_counts().rename_axis('regions').reset_index(name='counts') 

region_df[region_df.counts > 8].plot(kind='bar', x= "regions", y="counts", title= "Region Occurance over 8", figsize=(12,9))





# Question 4

# convert 'F' to female

gender_df = data['Gender'].value_counts().rename_axis('gender').reset_index(name='counts')

gender_df



data['Gender'] = np.where(data['Gender']!='Male', 'Female', data['Gender'])

gender_df = data['Gender'].value_counts().rename_axis('gender').reset_index(name='counts')

gender_df


# Create a list of colors

colors = ["#346beb", "#D69A80"]



# Create a pie chart

data['Gender'].value_counts().plot.pie(colors=colors,autopct='%1.1f%%', fontsize=9, figsize=(6, 6))

# Question 5

data['Footlength_cm'] =  np.where(data['Footlength_cm'].str.isnumeric()==True,data['Footlength_cm'],'NaN')

data['Armspan_cm'] =  np.where(data['Armspan_cm'].str.isnumeric()==True,data['Armspan_cm'],'NaN')

data['Height_cm'] =  np.where(data['Height_cm'].str.isnumeric()==True,data['Height_cm'],'NaN')



df1.dtypes #detemine the datatype of the column
# Question 6

data.isnull().sum(axis=0)

# Question 7

data = data.drop([0], axis=0 )

data
# Question 8

data = data.dropna(axis=0)

data.isnull().sum(axis=0)
# Question 9

data[[ 'Age-years', 'Height_cm', 'Footlength_cm','Armspan_cm','Languages_spoken', 'Travel_time_to_School','Reaction_time', 'Score_in_memory_game' ]].astype(float).corr()
# Question 10

extract = data[['Armspan_cm','Footlength_cm', 'Reaction_time','Height_cm']] 

print(extract)
extract
#Question 11

extract["Armspan_cm"] = extract.Armspan_cm.astype(float)

extract["Footlength_cm"] = extract.Footlength_cm.astype(float)

extract["Height_cm"] = extract.Height_cm.astype(float)

extract["Reaction_time"] = extract.Reaction_time.astype(float)
extract.plot(kind='scatter', x='Armspan_cm', y="Reaction_time", title="Relationship between Armspan and Reaction time", figsize=(12,9))
# Question  12



# ensure there is not missing data

extract = extract.dropna(axis=0)



extract.isnull().sum(axis=0)





X_data =extract[['Height_cm','Footlength_cm','Armspan_cm']]

Y_data  = extract['Reaction_time']
#splits data into 70/30 for training and testing repectively

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30) 



#import linear model package

from sklearn import linear_model
# Create an instance of linear regression

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
reg.coef_
X_train.columns
print("Regression Coefficients")

pd.DataFrame(reg.coef_,index=X_train.columns,columns=["Coefficient"])
#intercept

reg.intercept_
#validate regression model

test_predicted = reg.predict(X_test)

test_predicted
#Question 14

from sklearn.metrics import mean_squared_error, r2_score
# mean squared error

print("Mean squared error: %.2f" % mean_squared_error(y_test, test_predicted))
# Explained variance score: 1 is perfect prediction

# R squared

print('Variance score: %.2f' % r2_score(y_test, test_predicted))
reg.score(X_test,y_test)
# Intercept

reg.intercept_
# Make predictions using the testing set

test_predicted = reg.predict(X_test)

test_predicted
help(reg.score)
# Make predictions using the testing set

test_predicted = reg.predict(X_test)



# determine residuals

residiuals = y_test - test_predicted



import seaborn as sns



sns.distplot(residiuals)
sns.residplot(y_test,test_predicted)
#R-Squared = Explained variance of the model / Total variance of the target variable

# R squared

print('Variance score: %.2f' % r2_score(y_test, test_predicted))
reg.score(X_test,y_test)
pca = PCA(n_components=1)

pca.fit(extract[X_train.columns])
pca.components_
pca.n_features_

pca.n_components_

X_test
X_reduced = pca.transform(X_test)

X_reduced
plt.scatter(X_reduced, y_test,  color='black')



plt.scatter(X_reduced, y_test,  color='black')

plt.scatter(X_reduced, test_predicted, color='blue')

plt.plot(X_reduced, test_predicted, color='red',linewidth=1)



plt.xticks(())

plt.yticks(())



plt.show()
import seaborn as sns



sns.distplot((y_test-test_predicted))
reg.predict([[190,26,130]])
# Question 18

import sklearn.metrics as metrics

mae = metrics.mean_absolute_error(y_test, test_predicted)

mse = metrics.mean_squared_error(y_test, test_predicted)

rmse = np.sqrt(mse) # or mse**(0.5)  

r2 = metrics.r2_score(y_test, test_predicted)



print("Results of sklearn.metrics:")

print("MAE:",mae)

print("MSE:", mse)

print("RMSE:", rmse)


