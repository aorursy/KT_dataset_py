import os

print(os.listdir("../input"))
# ignore warnings

import warnings

warnings.filterwarnings("ignore")



# Wrangling

import pandas as pd



# Exploring

import scipy.stats as stats



# Visualizing

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('classic')



# Modeling

# THIS LINE YIELDED AN ERROR. I RESEARCHED THE ERROR, BUT DID NOT WANT TO DO THE PIP INSTALL 

# REQUIRED TO FIX IT.

# import statsmodels.api as sm 



from scipy.stats import pearsonr



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
train = pd.read_csv("../input/train.csv")

print("Train columns:  %s" % list(train.columns))

print("Train dimensions (rows, columns):", train.shape)

test = pd.read_csv("../input/test.csv")

print("Test columns:  %s" % list(test.columns))

print("Test dimensions (rows, columns):", test.shape)
train.isnull().sum()
train.loc[train['y'].isnull()]
train.loc[train['x'] > 99]
train.drop(labels=213, inplace=True)
train.loc[train['x'] > 99]
train.describe()
test.isnull().sum()
train = pd.DataFrame(train)

test = pd.DataFrame(test)
train.head()
print("Train columns:  %s" % list(train.columns))

print("Train dimensions (rows, columns):", train.shape)

print("Test columns:  %s" % list(test.columns))

print("Test dimensions (rows, columns):", test.shape)
X_train = train.drop(columns='y')

print("X_train:")

print(type(X_train))

print(X_train.head())

print()

y_train = train.drop(columns='x')

print("y_train")

print(type(y_train))

print(y_train.head())

print()

X_test = test.drop(columns='y')

print("X_test:")

print(type(X_test))

print(X_test.head())

print()

y_test = test.drop(columns='x')

print("y_test")

print(type(y_test))

print(y_test.head())

print()
print(X_train.isnull().sum())

print(y_train.isnull().sum())

print(X_test.isnull().sum())

print(y_test.isnull().sum())
if X_train.shape[0] == y_train.shape[0]:

    print("X & y train rows ARE equal")

else:

    print("X & y train rows ARE NOT equal")





if X_test.shape[0] == y_test.shape[0]:

    print("X & y test rows ARE equal")

else:

    print("X & y test rows ARE NOT equal")



if train.shape[1] == test.shape[1]:

    print("Number of columns in train & test ARE equal")

else:

    print("Number of columns in train & test ARE NOT equal")



train_split = train.shape[0] / (train.shape[0] + test.shape[0])

test_split = test.shape[0] / (train.shape[0] + test.shape[0])



print("Train Split: %.2f" % train_split)

print("Test Split: %.2f" % test_split)
with sns.axes_style('white'):

    j = sns.jointplot("x", "y", data=train, kind='reg', height=5);

    j.annotate(stats.pearsonr)

plt.show()
# This is roughly equivalent to sns.jointplot, but we see here that we have the

# flexibility to customize the type of the plots in each position.



g = sns.PairGrid(train)

g.map_diag(plt.hist)

g.map_offdiag(plt.scatter);
plt.figure(figsize=(8,4))

sns.heatmap(train.corr(), cmap='Blues', annot=True)
# pearsonr(X_train, y_train)
# ols_model = sm.OLS(y_train, X_train)

# fit = ols_model.fit()

# fit.summary()
# Create linear regression objects

lm1 = LinearRegression()

print(lm1)
lm1.fit(X_train[['x']], y_train)

print(lm1)



lm1_y_intercept = lm1.intercept_

print(lm1_y_intercept)



lm1_coefficients = lm1.coef_

print(lm1_coefficients)
print('Univariate - y = b + m * exam1')

print('    y-intercept (b): %.2f' % lm1_y_intercept)

print('    coefficient (m): %.2f' % lm1_coefficients[0])

print()
y_pred_lm1 = lm1.predict(X_train[['x']])
mse_lm1 = mean_squared_error(y_train, y_pred_lm1)

print("lm1\n  mse: {:.3}".format(mse_lm1)) 
r2_lm1 = r2_score(y_train, y_pred_lm1)



print('  {:.2%} of the variance in the y can be explained by x.'.format(r2_lm1))
plt.scatter(y_pred_lm1, y_pred_lm1 - y_train, c='g', s=40)

plt.hlines(y=0, xmin=50, xmax=100)

plt.title("Residual plot")

plt.ylabel('Residuals')
# Make predictions using the testing set

y_pred_test = lm1.predict(X_test[['x']])
mse = mean_squared_error(y_test, y_pred_test)



print("Mean squared error: %.2f" % mse)
r2 = r2_score(y_test, y_pred_test)



print('{:.2%} of the variance in y can be explained by x.'

      .format(r2))
plt.scatter(y_pred_test, y_pred_test - y_test, c='g', s=40)

plt.hlines(y=0, xmin=50, xmax=100)

plt.title("Residual plot")

plt.ylabel('Residuals')