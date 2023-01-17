import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 
from sklearn.datasets import load_boston 

boston = load_boston() 
boston.data.shape
data = pd.DataFrame(boston.data) 

data.columns = boston.feature_names 

  

data.head(10)
data['Price'] = boston.target 

data.head()
data.describe() 
data.info()
x = boston.data

y = boston.target

from sklearn.model_selection import train_test_split ,cross_val_score 

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size =0.2, 

                                                    random_state = 0) 

print("xtrain shape : ", xtrain.shape) 

print("xtest shape  : ", xtest.shape) 



print("ytrain shape : ", ytrain.shape) 

print("ytest shape  : ", ytest.shape)
from sklearn.linear_model import LinearRegression 

regressor = LinearRegression() 

regressor.fit(xtrain, ytrain) 
y_pred = regressor.predict(xtest)


from sklearn import metrics

from sklearn.metrics import r2_score

print('Mean Absolute Error : ', metrics.mean_absolute_error(ytest, y_pred))

print('Mean Square Error : ', metrics.mean_squared_error(ytest, y_pred))

print('RMSE', np.sqrt(metrics.mean_squared_error(ytest, y_pred)))

print('R squared error', r2_score(ytest, y_pred))
plt.scatter(ytest, y_pred, c = 'green') 

plt.xlabel("Price: in $1000's") 

plt.ylabel("Predicted value") 

plt.title("True value vs predicted value : Linear Regression") 

plt.show() 
from sklearn.linear_model import Ridge

cross_val_scores_ridge = []
alpha = [] 

from statistics import mean

# Loop to compute the different values of cross-validation scores 

for i in range(1, 9): 

    ridgeModel = Ridge(alpha = i * 0.25) 

    ridgeModel.fit(xtrain, ytrain) 

    scores = cross_val_score(ridgeModel, x, y, cv = 10) 

    avg_cross_val_score = mean(scores)*100

    cross_val_scores_ridge.append(avg_cross_val_score) 

    alpha.append(i * 0.25) 

    # Loop to print the different values of cross-validation scores 

for i in range(0, len(alpha)): 

    print(str(alpha[i])+' : '+str(cross_val_scores_ridge[i])) 
# Building and fitting the Ridge Regression model 

ridgeModelChosen = Ridge(alpha = 2) 

ridgeModelChosen.fit(xtrain, ytrain) 

  

# Evaluating the Ridge Regression model 

print(ridgeModelChosen.score(xtest, ytest)) 
y_pred2 = ridgeModel.predict(xtest) 
print('Mean Absolute Error : ', metrics.mean_absolute_error(ytest, y_pred))

print('Mean Square Error : ', metrics.mean_squared_error(ytest, y_pred))

print('RMSE', np.sqrt(metrics.mean_squared_error(ytest, y_pred)))

print('R squared error', r2_score(ytest, y_pred))
plt.scatter(ytest, y_pred2, c = 'blue') 

plt.xlabel("Price: in $1000's") 

plt.ylabel("Predicted value") 

plt.title("True value vs predicted value : Ridge Regression") 

plt.show()
from sklearn.linear_model import Lasso
# List to maintain the cross-validation scores 

cross_val_scores_lasso = [] 

  

# List to maintain the different values of Lambda 

Lambda = [] 

  

# Loop to compute the cross-validation scores 

for i in range(1, 9): 

    lassoModel = Lasso(alpha = i * 0.25, tol = 0.0925) 

    lassoModel.fit(xtrain, ytrain) 

    scores = cross_val_score(lassoModel, x, y, cv = 10) 

    avg_cross_val_score = mean(scores)*100

    cross_val_scores_lasso.append(avg_cross_val_score) 

    Lambda.append(i * 0.25) 

  

# Loop to print the different values of cross-validation scores 



for i in range(0, len(alpha)): 

    print(str(alpha[i])+' : '+str(cross_val_scores_lasso[i]))
# Building and fitting the Lasso Regression Model 

lassoModelChosen = Lasso(alpha = 2, tol = 0.0925) 

lassoModelChosen.fit(xtrain, ytrain) 

  

# Evaluating the Lasso Regression model 

print(lassoModelChosen.score(xtest, ytest))
y_pred2 = lassoModelChosen.predict(xtest) 
print('Mean Absolute Error : ', metrics.mean_absolute_error(ytest, y_pred2))

print('Mean Square Error : ', metrics.mean_squared_error(ytest, y_pred2))

print('RMSE', np.sqrt(metrics.mean_squared_error(ytest, y_pred2)))

print('R squared error', r2_score(ytest, y_pred2))
plt.scatter(ytest, y_pred2, c = 'black') 

plt.xlabel("Price: in $1000's") 

plt.ylabel("Predicted value") 

plt.title("True value vs predicted value : Lasso Regulariation") 

plt.show()
# Building the two lists for visualization 

models = ['Linear Regression', 'Ridge Regression','Lasso Regression'] 

scores = [regressor.score(xtest, ytest), 

         ridgeModelChosen.score(xtest, ytest), 

        lassoModelChosen.score(xtest, ytest)] 

  

# Building the dictionary to compare the scores 

mapping = {} 

mapping['Linear Regreesion'] = regressor.score(xtest, ytest) 

mapping['Ridge Regreesion'] = ridgeModelChosen.score(xtest, ytest) 

mapping['Lasso Regression'] = lassoModelChosen.score(xtest, ytest) 



  

# Printing the scores for different models 

for key, val in mapping.items(): 

    print(str(key)+' : '+str(val)) 
# Plotting the scores 

plt.bar(models, scores) 

plt.xlabel('Regression Models') 

plt.ylabel('Score') 

plt.show() 