import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import sys
import matplotlib.pyplot as plt
# reading input data
input_data = pd.read_csv("../input/Life Expectancy Data.csv")
input_data.head()
# we will ignore countries and years, as they are a big giveaway for multiple years of the same country
x_input = input_data.drop(['Life expectancy ', 'Country', 'Year'], axis=1)  
y_input = input_data['Life expectancy ']
x_input = x_input.drop(['Status'], axis=1)
my_imputer = SimpleImputer(strategy="most_frequent") #strategy accounts for both strings and integer values
x_input = pd.DataFrame( my_imputer.fit_transform(x_input) )
y_input = pd.DataFrame( my_imputer.fit_transform(pd.DataFrame(y_input)) )
train_x, temp_x, train_y, temp_y = train_test_split(x_input,y_input, test_size=0.2)
val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size=0.5)
def add_bias(data):
    temp = data.copy()
    temp.insert(0,'bias',1)
    return temp
b_train_x = add_bias(train_x)
b_val_x = add_bias(val_x)
b_test_x = add_bias(test_x)
plain_lin_reg_weights = np.linalg.inv(np.transpose(b_train_x) @ b_train_x) @ np.transpose(b_train_x) @ train_y

plain_lin_reg_predictions = b_test_x @ plain_lin_reg_weights
plain_lin_reg_error = mean_squared_error(plain_lin_reg_predictions, test_y)
def normalize (data):
    return (data - data.mean() ) / data.std()
norm_train_x = normalize(train_x)
norm_val_x = normalize(val_x)
norm_test_x = normalize(test_x)
bnorm_train_x = add_bias(norm_train_x)
bnorm_val_x = add_bias(norm_val_x)
bnorm_test_x = add_bias(norm_test_x)
# finds the optimal lambda value between the optional parameters for min(mn), max(mx), and jump size
def cross_val(train_x, train_y, val_x, val_y, mn = 0.001, mx = 2, jump = 0.0005):
    best_lambd = mn
    lowest_error = sys.maxsize
    
    for lambd in np.arange(mn, mx, jump):
        cur_weights = np.linalg.inv((np.transpose(train_x) @ train_x) + (lambd*np.eye(len(train_x.axes[1]))) ) @ np.transpose(train_x) @ train_y
        cur_predictions = val_x @ cur_weights
        cur_error = mean_squared_error(cur_predictions, val_y)
        if (lowest_error > cur_error):
            lowest_error = cur_error
            best_lambd = lambd
            
    return best_lambd
ridge_lambda = cross_val(bnorm_train_x, train_y, bnorm_val_x, val_y)
ridge_lin_reg_weights = np.linalg.inv((np.transpose(bnorm_train_x) @ bnorm_train_x) + (ridge_lambda * np.eye(len(bnorm_train_x.axes[1]))) ) @ np.transpose(bnorm_train_x) @ train_y

ridge_lin_reg_predictions = bnorm_test_x @ ridge_lin_reg_weights
ridge_lin_reg_error = mean_squared_error(ridge_lin_reg_predictions, test_y)
def lasso_cross_val2( train_x, train_y, val_x, val_y, mn = 0.001, mx = 2, jump = 0.0005):
    best_lambd = mn
    lowest_error = sys.maxsize
    for lambd in np.arange(mn, mx, jump):
        temp_model = linear_model.Lasso(alpha=lambd, max_iter=10000)
        temp_model.fit(train_x, train_y)
        cur_error = mean_squared_error(temp_model.predict(val_x), val_y)
        if (lowest_error > cur_error):
            lowest_error = cur_error
            best_lambd = lambd
            
    return best_lambd
lasso_lambd = lasso_cross_val2(norm_train_x, train_y, norm_val_x, val_y)
lasso_model = linear_model.Lasso(alpha=lasso_lambd, max_iter=10000)
lasso_model.fit(norm_train_x, train_y)

lasso_lin_reg_predictions = lasso_model.predict(norm_test_x)
lasso_lin_reg_error = mean_squared_error(lasso_lin_reg_predictions, test_y.values)
alphas, _, coefs = linear_model.lars_path(train_x.values, train_y.values.ravel(), method='lasso')

xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.figure(figsize=[20,7])
plt.plot(xx, coefs.T)

plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')

plt.ylim(0,3)
plt.show()
print('The residual sum of square error is:\nPlain Least Squares Regression: ', plain_lin_reg_error, \
                                          '\nRidge Regression: ', ridge_lin_reg_error, \
                                           '\nLasso Regression: ', lasso_lin_reg_error)