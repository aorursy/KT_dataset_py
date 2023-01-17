#Numpy is used so that we can deal with array's, which are necessary for any linear algebra

# that takes place "under-the-hood" for any of these algorithms.



import numpy as np





#Pandas is used so that we can create dataframes, which is particularly useful when

# reading or writing from a CSV.



import pandas as pd





#Matplotlib is used to generate graphs in just a few lines of code.



import matplotlib.pyplot as plt





#Import the classes we need to test linear, ridge, and lasso to compare



from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV



#Need these for selecting the best model

from sklearn.model_selection import KFold, GridSearchCV





#These will be our main evaluation metrics 

from sklearn.metrics import r2_score, mean_squared_error







# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split



# Will use this to "normalize" our data.

from sklearn.preprocessing import normalize



#read the data from csv

dataset = pd.read_csv('../input/50-startups/50_Startups.csv')



#take a look at our dataset.  head() gives the first 5 lines. 

dataset.head()
#drop the column

dataset = dataset.drop(columns = ['State'])



#take a look again 

dataset.head()
#set independent variable by using all rows, but just column 1.

X = dataset.iloc[:, :-1].values



#set the dependent variable using all rows but only the last column. 

y = dataset.iloc[:, -1].values



#lets take a look at X right now.

X[0:10]
X = normalize(X, 'l2')



X[0:10]
#split the dataset.  Take 40% to be our test set. 



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)

#this sets the object regressor to the class of LinearRegression from the Sklearn library.

regressor = LinearRegression()



#this fits the model to our training data.

regressor.fit(X_train, y_train)
#Predict on our test set.

y_pred = regressor.predict(X_test)
#calculate the R^2 score

score = r2_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)



#print out our score properly formatted as a percent.

print("R^2 score:", "{:.4f}%".format(score))

print("MSE", round(mse,2))
alphas = [-5, -1, 1e-4, 1e-3, 1e-2, 1, 5]



def test_alpha(a):

    model_lasso = Lasso(alpha=a)

    model_lasso.fit(X_train, y_train) 

    pred_test_lasso = model_lasso.predict(X_test)

    new_score = r2_score(y_test, pred_test_lasso)

    new_mse = mean_squared_error(y_test, pred_test_lasso)

    print('ALPHA: {:.3f} R2 SCORE: {:.4f}% new_score, {:.1f}'.format(a, new_score, new_mse))

    

    

for alpha in alphas:

    test_alpha(alpha)

alphas = [-5, -1, 1e-4, 1e-3, 1e-2, 1, 5]



def test_alpha_ridge(a):

    model_lasso = Ridge(alpha=a)

    model_lasso.fit(X_train, y_train) 

    pred_test_lasso = model_lasso.predict(X_test)

    new_score = r2_score(y_test, pred_test_lasso)

    new_mse = mean_squared_error(y_test, pred_test_lasso)

    print('ALPHA: {:.3f} R2 SCORE: {:.4f}% new_score, {:.1f}'.format(a, new_score, new_mse))

    

    

for alpha in alphas:

    test_alpha_ridge(alpha)
new_alphas = [1e-15,1e-10,1e-8,1e-4, 1e-3, 1e-2, 1]



for alpha in new_alphas:

    test_alpha_ridge(alpha)