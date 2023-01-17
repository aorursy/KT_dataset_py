# Linear Regression for Boston house prices
import numpy
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression

filename = '../input/boston-house-prices/housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv(filename, delim_whitespace=True, names=names)
# SHAPE OF THE DATASET
print (dataset.shape)
# Describe the dataset
print (dataset.describe())
array = dataset.values
X = array[:,0:13]
Y = array[:,13]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
random_state=seed)
model = LinearRegression()
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
finalResult = result*100
print("The Accuracy Score  - {}". format(finalResult))
kfold = KFold(n_splits=5, random_state=7)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f) " % (results.mean(), results.std()))
print("The Actual result  - {}". format(results))
# synthetic dataset for classification (binary) 
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs, make_regression
plt.style.use('ggplot')
plt.style.use('seaborn-colorblind')
plt.figure(figsize=(7,7))
plt.title('Sample regression problem with one input variable')
X_R1, y_R1 = make_regression(n_samples = 100, n_features=1,
                            n_informative=1, bias = 150.0,
                            noise = 30, random_state=0)
plt.scatter(X_R1, y_R1, marker= 'o', s=50)
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1,random_state = 0)
linreg = LinearRegression().fit(X_train, y_train)

print('linear model coeff (w): {}'.format(linreg.coef_))
print('linear model intercept (b): {:.3f}'.format(linreg.intercept_))
print('R-squared score (training): {:.3f}'.format(linreg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(linreg.score(X_test, y_test)))
plt.figure(figsize=(7,7))
plt.scatter(X_R1, y_R1, marker= 'o', s=50, alpha=0.8)
plt.plot(X_R1, linreg.coef_ * X_R1 + linreg.intercept_, 'r-')
plt.title('Least-squares linear regression')
plt.xlabel('Feature value (x)')
plt.ylabel('Target value (y)')
plt.show()
df = pd.read_csv('../input/additionalcrimedataset/crimedata.csv', sep=',', na_values='?', header=0)
requiredColumns = [5, 6] + list(range(11,26)) + list(range(32, 103)) + [145]  
df = df.iloc[:,requiredColumns].dropna()

X = df.iloc[:,range(0,88)]
y = df['ViolentCrimesPerPop']
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
linearReg = LinearRegression().fit(X_train,y_train)

#print ("Linear Model Intercept - {}".format(linearReg.intercept_))
#print ("Linear Model Coefficient - \n {}".format(linearReg.coef_))
print ("Training Score - {:.3f}".format(linearReg.score(X_train,y_train)))
print ("Testing Score - {:.3f}".format(linearReg.score(X_test,y_test)))
from sklearn.linear_model import Ridge
import numpy as np
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
ridgeReg = Ridge(alpha=20.0).fit(X_train,y_train)

print ("Training Score - {:.3f}".format(ridgeReg.score(X_train,y_train)))
print ("Testing Score - {:.3f}".format(ridgeReg.score(X_test,y_test)))
print('Number of non-zero features: {}'.format(np.sum(ridgeReg.coef_ != 0)))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

ridgeReg = Ridge(alpha=20.0).fit(X_train_scale,y_train)

print ("Training Score - {:.3f}".format(ridgeReg.score(X_train_scale,y_train)))
print ("Testing Score - {:.3f}".format(ridgeReg.score(X_test_scale,y_test)))
print('Number of non-zero features: {}'.format(np.sum(ridgeReg.coef_ != 0)))

for newAlpha in [0, 1, 10, 20, 50, 100, 1000]:
    linridge = Ridge(alpha = newAlpha).fit(X_train_scale, y_train)
    r2_train = linridge.score(X_train_scale, y_train)
    r2_test = linridge.score(X_test_scale, y_test)
    num_coeff_bigger = np.sum(abs(linridge.coef_) > 1.0)
    print('Alpha = {:.2f}\nnum abs(coeff) > 1.0: {}, \
r-squared training: {:.2f}, r-squared test: {:.2f}\n'
         .format(newAlpha, num_coeff_bigger, r2_train, r2_test))
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
scaler = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

LassoReg = Lasso(alpha=2.0, max_iter = 10000).fit(X_train_scale,y_train)

print ("Training Score - {:.3f}".format(LassoReg.score(X_train_scale,y_train)))
print ("Testing Score - {:.3f}".format(LassoReg.score(X_test_scale,y_test)))
print('Number of non-zero features: {}'.format(np.sum(LassoReg.coef_ != 0)))
print('Lasso regression: effect of alpha regularization\n\
parameter on number of features kept in final model\n')

for alpha in [0.5, 1, 2, 3, 4, 5, 10, 20, 50]:
    linlasso = Lasso(alpha, max_iter = 10000).fit(X_train_scale, y_train)
    r2_train = linlasso.score(X_train_scale, y_train)
    r2_test = linlasso.score(X_test_scale, y_test)
    
    print('Alpha = {:.2f}\nFeatures kept: {}, r-squared training: {:.2f}, \
r-squared test: {:.2f}\n'.format(alpha, np.sum(linlasso.coef_ != 0), r2_train, r2_test))
