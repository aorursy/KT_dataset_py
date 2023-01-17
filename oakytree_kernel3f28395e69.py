DATA_DIR = "/kaggle/input/ucfai-core-sp20-regression"

# import some important stuff

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import scipy.stats as st

from sklearn import datasets, linear_model
# Get some data 

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 

y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])



# Let's plot the data to see what it looks like

plt.scatter(x, y, color = "black") 

plt.show()
# calculating the coefficients



# number of observations/points 

n = np.size(x) 



# mean of x and y vector 

m_x, m_y = np.mean(x), np.mean(y) 



# calculating cross-deviation and deviation about x 

SS_xy = np.sum(y*x - n*m_y*m_x) 

SS_xx = np.sum(x*x - n*m_x*m_x) 



# calculating regression coefficients 

b_1 = SS_xy / SS_xx 

b_0 = m_y - b_1*m_x



#var to hold the coefficients

b = (b_0, b_1)



#print out the estimated coefficients

print("Estimated coefficients:\nb_0 = {} \nb_1 = {}".format(b[0], b[1])) 
# Sklearn learn require this shape

x = x.reshape(-1,1)

y = y.reshape(-1,1)



# making the model

regress = linear_model.LinearRegression()

regress.fit(x, y)
# plotting the actual points as scatter plot 

plt.scatter(x, y, color = "black", 

           marker = "o", s = 30) 



# predicted response vector 

y_pred = b[0] + b[1]*x 



# plotting the regression line 

plt.plot(x, y_pred, color = "blue") 



# putting labels 

plt.xlabel('x') 

plt.ylabel('y') 



# function to show plot 

plt.show()
# here we can try out any data point

print(regress.predict([[6]]))
housing_data =  pd.read_csv(f"{DATA_DIR}/train.csv") 



# Mean Sales price 

mean_price = np.mean(housing_data["SalePrice"])

print("Mean Price : " + str(mean_price))



# Variance of the Sales Price 

var_price = np.var(housing_data["SalePrice"], ddof=1)

print("Variance of Sales Price : " + str(var_price))



# Median of Sales Price 

median_price = np.median(housing_data["SalePrice"])

print("Median Sales Price : " + str(median_price))



# Skew of Sales Price 

skew_price = st.skew(housing_data["SalePrice"])

print("Skew of Sales Price : " + str(skew_price))



housing_data["SalePrice"].describe()
plt.boxplot(housing_data["SalePrice"])

plt.ylabel("Sales Price")

plt.show()
plt.scatter(housing_data["GrLivArea"], housing_data["SalePrice"])

plt.ylabel("Sales Price")

plt.show()
# we need to reshape the array to make the sklearn gods happy

area_reshape = housing_data["GrLivArea"].values.reshape(-1,1)

price_reshape = housing_data["SalePrice"].values.reshape(-1,1)



# Generate the Model

model = linear_model.LinearRegression(fit_intercept=True)

model.fit(area_reshape, price_reshape)

price_prediction = model.predict(area_reshape)



# plotting the actual points as scatter plot 

plt.scatter(area_reshape, price_reshape) 



# plotting the regression line 

plt.plot(area_reshape, price_prediction, color = "red") 



# putting labels 

plt.xlabel('Above Ground Living Area') 

plt.ylabel('Sales Price') 



# function to show plot 

plt.show()
# we're going to need a different model, so let's import it

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
# read_csv allow us to easily import a whole dataset

data = pd.read_csv(f"{DATA_DIR}/adult.data", names =["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","income"])



# this tells us whats in it 

print(data.info())
# data.head() gives us some the the first 5 sets of the data

data.head()
# this is the function that give us some quick info about continous data in the dataset

data.describe()
# put the name of the parameter you want to test

test = "capital-gain"

#raise NotImplementedError()
# but before we make our model, we need to modify our data a bit



# little baby helper function

def incomeFixer(x):

    if x == " <=50K":

        return 0

    else:

        return 1



# change the income data into 0's and 1's

data["income"] = data.apply(lambda row: incomeFixer(row['income']), axis=1)



# get the data we are going to make the model with 

x = np.array(data[test])

y = np.array(data["income"])



# again, lets make the scikitlearn gods happy

x = x.reshape(-1,1)



# Making the test-train split

x_train, x_test, y_train, y_test = train_test_split(x ,y ,test_size=0.25, random_state=42)
# now make data model!

logreg = LogisticRegression(solver='liblinear')

logreg.fit(x_train,y_train)
# now need to test the model's performance

print(logreg.score(x_test,y_test))
# Run test and submit to kaggle competition!

test_data = pd.read_csv(f"{DATA_DIR}/adult.test", names =["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","income"])

test_data = test_data.drop("income", axis=1)

test_data = test_data.drop(test_data.index[0], axis=0)



#get the data we are going to make the model with 

x_test = np.array(test_data[test])



#again, lets make the scikitlearn gods happy

x_test = x_test.reshape(-1,1)



predictions = logreg.predict(x_test)

predictions = pd.DataFrame({'Category': predictions})



predictions.to_csv('submission.csv', header=['Category'], index_label='Id')