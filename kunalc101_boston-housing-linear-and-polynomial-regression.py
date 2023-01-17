import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

%matplotlib inline
boston = load_boston()
# boston is a dictionary, let's check what it contains
print(boston.keys())
print(boston.DESCR)
bos = pd.DataFrame(boston.data, columns=boston.feature_names)

bos['MEDV'] = boston.target
print("Dataframe type : {}".format(type(bos)))
print("Dataframe shape: {}".format(bos.shape))
print("Dataframe features: {}".format(list(bos.columns.values)))
bos.head()
bos.tail()
bos.describe()
# check for missing values in all the columns
print("[INFO] df isnull():\n {}".format(bos.isnull().sum()))
# set the size of the figure
sns.set(rc={'figure.figsize':(12, 8)})

g = sns.PairGrid(bos, vars=['LSTAT', 'RM', 'CRIM', 'NOX', 'MEDV'], height=1.5, aspect=1.5)
g = g.map_diag(plt.hist)
g = g.map_lower(sns.regplot, lowess=True, scatter_kws={'s': 15, 'alpha':0.3}, 
                line_kws={'color':'red', 'linewidth': 2})
g = g.map_upper(sns.kdeplot, n_levels=15, cmap='coolwarm')
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
plt.show()
fig, axes = plt.subplots(2, 2)
fig.suptitle("Scatterplot and Boxplot for LSTAT and RM")

sns.regplot(x=bos['LSTAT'], y=bos['MEDV'], lowess=True, scatter_kws={'s': 25, 'alpha':0.3},
            line_kws={'color':'purple', 'linewidth': 2}, ax=axes[0, 0])

sns.boxplot(x=bos['LSTAT'], ax=axes[0, 1])

sns.regplot(x=bos['RM'], y=bos['MEDV'], lowess=True, scatter_kws={'s': 25, 'alpha':0.3},
            line_kws={'color':'purple', 'linewidth': 2}, ax=axes[1, 0])

sns.boxplot(x=bos['RM'], ax=axes[1, 1]).set(xlim=(3, 9))

plt.show()
fig, axes = plt.subplots(2, 2)
fig.suptitle("Scatterplot and Boxplot for CRIM and NOX")

sns.regplot(x=bos['CRIM'], y=bos['MEDV'], lowess=True, scatter_kws={'s': 25, 'alpha':0.3},
            line_kws={'color':'purple', 'linewidth': 2}, ax=axes[0, 0])

sns.boxplot(x=bos['CRIM'], ax=axes[0, 1])

sns.regplot(x=bos['NOX'], y=bos['MEDV'], lowess=True, scatter_kws={'s': 25, 'alpha':0.3},
            line_kws={'color':'purple', 'linewidth': 2}, ax=axes[1, 0]).set(xlim=(0.35, 0.9))
            
sns.boxplot(x=bos['NOX'], ax=axes[1, 1]).set(xlim=(0.35, 0.9))

plt.show()
fig, axes = plt.subplots(2, 2)
fig.suptitle("Histogram of Key Features")

sns.distplot(bos['LSTAT'], bins=30, ax=axes[0, 0])
sns.distplot(bos['RM'], bins=30, ax=axes[0, 1])
sns.distplot(bos['CRIM'], bins=30, ax=axes[1, 0])
sns.distplot(bos['NOX'], bins=30, ax=axes[1, 1])

plt.show()
# plot a histogram showing the distribution of the target variable
sns.distplot(bos['MEDV'], bins=30)

plt.show()
# compute the pair wise correlation for all columns  
correlation_matrix = bos.corr(method='pearson').round(2)
 
# use the heatmap function from seaborn to plot the correlation matrix
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)

plt.title('Pearson pair-wise Correlation Matrix')
plt.show()
X = pd.DataFrame(np.c_[bos['LSTAT'], bos['RM'], bos['CRIM'], bos['NOX']], columns=['LSTAT', 'RM', 'CRIM', 'NOX'])
Y = bos['MEDV']

print(X.shape)
print(Y.shape)
# splits the training and test data set in 75% : 25%
# assign random_state to any value.This ensures consistency.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
lm = LinearRegression()
lm.fit(X_train, Y_train)

print('Linear Regression coefficients: {}'.format(lm.coef_))
print('Linear Regression intercept: {}'.format(lm.intercept_))

# model evaluation for training set
y_train_predict = lm.predict(X_train)

# plt.plot(np.unique(Y_train), np.poly1d(np.polyfit(Y_train, y_train_predict, 1))(np.unique(Y_train)), 
#         linewidth=2, color='r')

# calculating the intercept and slope for the regression line
b, m = np.polynomial.polynomial.polyfit(Y_train, y_train_predict, 1)
sns.scatterplot(Y_train, y_train_predict, alpha=0.4)
sns.regplot(Y_train, y_train_predict, truncate=True, scatter_kws={'s': 20, 'alpha':0.3}, line_kws={'color':'green', 'linewidth': 2})
sns.lineplot(np.unique(Y_train), np.unique(np.poly1d(b + m * np.unique(Y_train))), linewidth=0.5, color='r')

plt.xlabel("Actual Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Actual Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$ [Training Set]")
 
plt.show()
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)
 
print("The linear model performance for training set")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
# model evaluation for testing set
y_test_predict = lm.predict(X_test)
sns.scatterplot(Y_test, y_test_predict, alpha=0.4)
sns.regplot(Y_test, y_test_predict, truncate=True, scatter_kws={'s': 20, 'alpha':0.3}, line_kws={'color':'green', 'linewidth': 2})
 
plt.xlabel("Actual Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Actual Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$ [Test Set]")
 
plt.show()
# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
 
# r-squared score of the model
r2 = r2_score(Y_test, y_test_predict)

print("\nThe linear model performance for testing set")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
y_train_residual = y_train_predict - Y_train
y_test_residual = y_test_predict - Y_test

plt.subplot(1, 2, 1)
sns.distplot(y_train_residual, bins=15)
plt.title('Residual Histogram for Training Set')

plt.subplot(1, 2, 2)
sns.distplot(y_test_residual, bins=15)
plt.title('Residual Histogram for Test Set')

plt.show()
fig, axes = plt.subplots()
fig.suptitle('Residual plot of Training and Test set')

# Plot the residuals after fitting a linear model
sns.residplot(y_train_predict, y_train_residual, lowess=True, color="b", ax=axes, label='Training Set', 
              scatter_kws={'s': 25, 'alpha':0.3})

sns.residplot(y_test_predict, y_test_residual, lowess=True, color="g", ax=axes, label='Test Set',
              scatter_kws={'s': 25})

legend = axes.legend(loc='upper left', shadow=True, fontsize='large')
legend.get_frame().set_facecolor('#f9e79f')

plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.show()
"Creates a polynomial regression model for the given degree"
poly_features = PolynomialFeatures(degree=2)
   
# transform the features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)
   
# fit the transformed features to Linear Regression
poly_model = LinearRegression()

poly_model.fit(X_train_poly, Y_train)
     
# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)
   
# predicting on test data-set
y_test_predicted = poly_model.predict(poly_features.fit_transform(X_test))
y_train_residual = y_train_predicted - Y_train
y_test_residual = y_test_predicted - Y_test

plt.subplot(1, 2, 1)
sns.distplot(y_train_residual, bins=15)
plt.title('Residual Histogram for Training Set [Polynomial Model]')

plt.subplot(1, 2, 2)
sns.distplot(y_test_residual, bins=15)
plt.title('Residual Histogram for Test Set [Polynomial Model]')

plt.show()
sns.scatterplot(Y_train, y_train_predicted, alpha=0.4)
sns.regplot(Y_train, y_train_predicted, scatter_kws={'s': 20, 'alpha':0.3}, line_kws={'color':'green', 'linewidth': 2}, order=2)
 
plt.xlabel("Actual Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Actual Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$ [Training Set]")
 
plt.show()
sns.scatterplot(Y_test, y_test_predicted, alpha=0.4)
sns.regplot(Y_test, y_test_predicted, scatter_kws={'s': 20, 'alpha':0.3}, line_kws={'color':'green', 'linewidth': 2}, order=2)
 
plt.xlabel("Actual Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Actual Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$ [Test Set]")
 
plt.show()
# evaluating the model on training data-set
rmse_train = np.sqrt(mean_squared_error(Y_train, y_train_predicted))
r2_train = r2_score(Y_train, y_train_predicted)
     
print("The polynomial model performance for the training set")
print("RMSE of training set is {}".format(rmse_train))
print("R2 score of training set is {}".format(r2_train))
# evaluating the model on test data-set
rmse_test = np.sqrt(mean_squared_error(Y_test, y_test_predicted))
r2_test = r2_score(Y_test, y_test_predicted)

print("The polynomial model performance for the test set")
print("RMSE of test set is {}".format(rmse_test))
print("R2 score of test set is {}".format(r2_test))