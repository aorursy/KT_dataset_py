import pandas as pd  # To read data

import numpy as np # To calculate data

df = pd.read_csv("/kaggle/input/house-prices-dataset/train.csv")

df_test = pd.read_csv("/kaggle/input/house-prices-dataset/test.csv")

df.head()
df.describe()
df.dtypes
import pandas as pd

from matplotlib import pyplot as plt

from pylab import rcParams

rcParams['figure.figsize'] = 7,7 

import seaborn as sns

import numpy as np

sns.set(color_codes=True, font_scale=1.2)



%matplotlib inline

%config InlineBackend.figure_format = 'retina'





import heatmapk

from heatmapk import heatmap, corrplot

plt.figure(figsize=(10, 10))

corrplot(df.corr(), size_scale=300);
corr_matrix=df.corr()

corr_matrix['SalePrice'].sort_values(ascending=False)
df1 = df[['OverallQual', 'GrLivArea', 'GarageCars', 

          'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'SalePrice']].copy()

df1.head()
df1.isnull().sum()
%matplotlib inline

import matplotlib.pyplot as plt

df1.hist(bins=50, figsize=(20,15))

plt.savefig("attribute_histogram_plots")

plt.show()


sns.regplot(x='1stFlrSF', y='SalePrice', data=df1)
df1[['1stFlrSF', 'SalePrice']].corr()
sns.regplot(x='GarageArea', y='SalePrice', data=df1)

df1[['GarageArea','SalePrice']].corr()
sns.regplot(x='GarageCars', y='SalePrice', data=df1)

df1[['GarageCars', 'SalePrice']].corr()
sns.regplot(x='GrLivArea', y='SalePrice', data=df1)
df1[['GrLivArea', 'SalePrice']].corr()
sns.regplot(x='OverallQual', y='SalePrice', data=df1)
df1[['OverallQual', 'SalePrice']].corr()
sns.regplot(x='TotalBsmtSF', y='SalePrice', data=df1)
df1[['TotalBsmtSF', 'SalePrice']].corr()
from scipy import stats
pearson_coef, p_value=stats.pearsonr(df1['OverallQual'], df1['SalePrice'])

print('The Pearson Correlation Coefficient is ', pearson_coef, 'with a P-value of P =', p_value)
pearson_coef, p_value=stats.pearsonr(df1['GrLivArea'], df1['SalePrice'])

print('The Pearson Correlation Coefficient is ', pearson_coef, 'with a P-value of P=', p_value)
pearson_coef, p_value=stats.pearsonr(df1['GarageCars'], df1['SalePrice'])

print('The Pearson Correlation Coefficient is ', pearson_coef, 'with a P-value of P=', p_value)
pearson_coef, p_value=stats.pearsonr(df1['GarageArea'], df1['SalePrice'])

print('The Pearson Correlation Coefficient is ', pearson_coef, 'with a P-value of P=', p_value)
pearson_coef, p_value=stats.pearsonr(df1['TotalBsmtSF'], df1['SalePrice'])

print('The Pearson Correlation Coefficient is', pearson_coef, ' with a P-value of P=', p_value)
pearson_coef, p_value=stats.pearsonr(df1['1stFlrSF'], df1['SalePrice'])

print('The Pearson Correlation Coefficient is', pearson_coef, 'with a P-value of P=', p_value)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures
msk = np.random.rand(len(df)) < 0.8

train =df1[msk]

test =df1[~msk]
plt.scatter(train.OverallQual, train.SalePrice, color='green')

plt.xlabel('Overall Quality')

plt.ylabel('Sales Price')

plt.show()
from sklearn import linear_model



regr = linear_model.LinearRegression()

train_x = np.asanyarray(train[['OverallQual']])

train_y = np.asanyarray(train[['SalePrice']])

regr.fit (train_x, train_y)



# The coefficients

print('Coefficients: ', regr.coef_)

print('Intercept: ', regr.intercept_)
plt.scatter(train.OverallQual,  train.SalePrice, color ='green')

plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], '-r')

plt.xlabel('Overall Quality')

plt.ylabel('Sales Price')
from sklearn.metrics import r2_score



test_x = np.asanyarray(test[['OverallQual']])

test_y = np.asanyarray(test[['SalePrice']])

test_y_hat = regr.predict(test_x)



print('Mean Absolute Error: %.2f' % np.mean(np.absolute(test_y_hat-test_y)))

print('Residual sum of squares (MSE): %.2f' % np.mean((test_y_hat - test_y) **2))

print('R2-score: of %.2f' % r2_score(test_y_hat, test_y))     
X=df1[['OverallQual']]

Y=df1['SalePrice']

lm=LinearRegression()

lm

lm.fit(X, Y)

Yhat=lm.predict(X)

Yhat[0:5]
lm.intercept_
lm.coef_
Yhat_df1 = pd.DataFrame(Yhat)

Yhat_df1.columns = ['SalePrice']

Yhat_df1.head(10)
Id = df_test[['Id']]

Yhat = pd.concat([Id,Yhat_df1],axis=1)

Yhat.head(10)
Z = df1[['OverallQual', 'GrLivArea', 'GarageCars', 

          'GarageArea', 'TotalBsmtSF', '1stFlrSF']]
lm.fit(Z, df1['SalePrice'])
lm.intercept_
lm.coef_
y_output = lm.predict(Z)
y_output_df1 = pd.DataFrame(y_output)

y_output_df1.columns = ['SalePrice']

y_output_df1.head(10)
Id = df_test[['Id']]

Y_output = pd.concat([Id,y_output_df1],axis=1)

Y_output.head(10)
x2 = Z

y2 = df1['SalePrice']

lm.fit(x2, y2)

lm.score(x2,y2)
Input=[('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)),

      ('model', LinearRegression())]
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

import numpy as np





test = pd.read_csv("/kaggle/input/house-prices-dataset/test.csv")

df.head()

#Opening our file with the training data in

train = pd.read_csv("/kaggle/input/house-prices-dataset/train.csv")



#We are trying to predict the sale price column

target = train.SalePrice



#Get rid of the answer and anything thats not an object

train = train.drop(['SalePrice'],axis=1).select_dtypes(exclude=['object'])



#Split the data into test and validation

train_X, test_X, train_y, test_y = train_test_split(train,target,test_size=0.25)



#Impute all the NaNs

my_imputer = SimpleImputer()

train_X = my_imputer.fit_transform(train_X)

test_X = my_imputer.fit_transform(test_X)



#Simplist XGBRegressor

#my_model = XGBRegressor()

#my_model.fit(train_X, train_y)



my_model = XGBRegressor(n_estimators=300, learning_rate=0.08)

my_model.fit(train_X, train_y, early_stopping_rounds=4, 

             eval_set=[(test_X, test_y)], verbose=False)





#Make predictions

predictions = my_model.predict(test_X)



print("Mean absolute error = " + str(mean_absolute_error(predictions,test_y)))
# make predictions



predictions = my_model.predict(test_X)



from sklearn.metrics import mean_absolute_error

print('Mean Absolute Error : ' + str(mean_absolute_error(predictions, test_y)))