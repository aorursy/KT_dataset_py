# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import norm

from scipy.stats import skew

sns.set(style='white', context='notebook', palette='deep')



# Any results you write to the current directory are saved as output.
dataset  = pd.read_csv("../input/DC_Properties.csv")

dataset.head()
#dataset.isnull().sum()

dummy_dataset = dataset

dummy_dataset['Price_Flag'] = np.where(dummy_dataset.PRICE > 0 , 1,0)

unknown_dataset = dummy_dataset[dummy_dataset.Price_Flag != 1]

unknown_dataset.shape
from sklearn.model_selection import train_test_split

dataset = dummy_dataset[dummy_dataset.Price_Flag != 0]

dataset = dataset.drop(['Price_Flag','X','Y','CMPLX_NUM','FULLADDRESS','LONGITUDE','CITY','STATE','NATIONALGRID','CENSUS_BLOCK','SALEDATE','QUADRANT'],axis=1)

dataset.GBA = dataset.GBA.fillna(dataset.GBA.mean())

dataset.AYB = dataset.AYB.fillna(dataset.AYB.median())

dataset.STORIES = dataset.STORIES.fillna(dataset.STORIES.median())

dataset.KITCHENS = dataset.KITCHENS.fillna(dataset.KITCHENS.median())

dataset.NUM_UNITS = dataset.NUM_UNITS.fillna(dataset.NUM_UNITS.median())

dataset.YR_RMDL = dataset.YR_RMDL.fillna(dataset.YR_RMDL.median())

dataset.LIVING_GBA = dataset.LIVING_GBA.fillna(dataset.LIVING_GBA.mean())

dataset.STYLE = dataset.STYLE.fillna(dataset.STYLE.mode()[0])

dataset.STRUCT = dataset.STRUCT.fillna(dataset.STRUCT.mode()[0])

dataset.GRADE = dataset.GRADE.fillna(dataset.GRADE.mode()[0])

dataset.CNDTN = dataset.CNDTN.fillna(dataset.CNDTN.mode()[0])

dataset.EXTWALL = dataset.EXTWALL.fillna(dataset.EXTWALL.mode()[0])

dataset.ROOF = dataset.ROOF.fillna(dataset.ROOF.mode()[0])

dataset.INTWALL = dataset.INTWALL.fillna(dataset.INTWALL.mode()[0])

dataset.ASSESSMENT_SUBNBHD  = dataset.ASSESSMENT_SUBNBHD.fillna(dataset.ASSESSMENT_SUBNBHD.mode()[0])

dataset.isnull().sum()


train_dataset = dataset

train_dataset = train_dataset.drop('Unnamed: 0',axis =1)

cat = len(train_dataset.select_dtypes(include=['object']).columns)
num = len(train_dataset.select_dtypes(include=['int64','float64']).columns)
print('Total Features: ', cat, 'categorical', '+',
      num, 'numerical', '=', cat+num, 'features')
corrmat = train_dataset.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
k = 11 #number of variables for heatmap
cols = corrmat.nlargest(k, 'PRICE')['PRICE'].index
cm = np.corrcoef(train_dataset[cols].values.T)
sns.set(font_scale=1.00)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
most_corr
# Gross Building Area vs  Price

sns.jointplot(x=train_dataset['GBA'], y=train_dataset['PRICE'], kind='reg')
# AYB vs  Price

var = 'AYB'
data = pd.concat([train_dataset['PRICE'], train_dataset[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="PRICE", data=data)
fig.axis(ymin=0, ymax=2000000);
plt.xticks(rotation=90);
# Kitchens vs  Price

var = 'KITCHENS'
data = pd.concat([train_dataset['PRICE'], train_dataset[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="PRICE", data=data)
fig.axis(ymin=0, ymax=2000000);
plt.xticks(rotation=90);
# Stories vs  Price

var = 'STORIES'
data = pd.concat([train_dataset['PRICE'], train_dataset[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="PRICE", data=data)
fig.axis(ymin=0, ymax=5000000);
plt.xticks(rotation=90);
from sklearn.preprocessing import LabelEncoder
cols = train_dataset.select_dtypes(include=['object']).columns
# Process columns and apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train_dataset[c].values)) 
    train_dataset[c] = lbl.transform(list(train_dataset[c].values))

train_dataset.head()
# We use the numpy fuction log which  applies log to all elements of the column
train_dataset["PRICE"] = np.log(train_dataset["PRICE"])

#Check the new distribution 
sns.distplot(train_dataset['PRICE'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train_dataset['PRICE'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(train_dataset['PRICE'], plot=plt)
plt.show()

y_train = train_dataset.PRICE.values

print("Skewness: %f" % train_dataset['PRICE'].skew())
print("Kurtosis: %f" % train_dataset['PRICE'].kurt())
train_dataset = (train_dataset - train_dataset.mean()) / (train_dataset.max() - train_dataset.min())
train_dataset = pd.get_dummies(train_dataset)
print(train_dataset.shape)
train_dataset, test_dataset = train_test_split(train_dataset, test_size=0.2)

train_dataset.shape
import statsmodels.api as sm

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

train_dataset_Y = train_dataset.PRICE.values

train_dataset_X = train_dataset.drop('PRICE',axis =1)

train_dataset_X.shape

train_dataset_X = sm.add_constant(train_dataset_X)

Pricing_model = sm.OLS(train_dataset_Y,train_dataset_X)

result = Pricing_model.fit()

print(result.summary())

print("RMSE: ",np.sqrt(mean_squared_error(result.fittedvalues,train_dataset_Y)))
test_dataset_Y = test_dataset.PRICE.values

test_dataset = test_dataset.drop('PRICE',axis=1)

predictions = result.predict(sm.add_constant(test_dataset))

RMSE = np.sqrt(mean_squared_error(predictions,test_dataset_Y))

print('The RMSE of the predicted values is ',RMSE)


plt.scatter(test_dataset_Y, predictions)

plt.legend()
plt.title('OLS predicted values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()
from sklearn.linear_model import Ridge

from sklearn.linear_model import RidgeCV

from sklearn.metrics import r2_score

## training the model

train_dataset_X = train_dataset_X.drop('const',axis=1)

regr_cv = RidgeCV(alphas=[0.1,1,2,3,4,5,6,7,0.5,0.8])

model_cv = regr_cv.fit(train_dataset_X,train_dataset_Y)

model_cv.alpha_

ridgeReg = Ridge(alpha=5, normalize=True)

ridgeReg.fit(train_dataset_X,train_dataset_Y)

pred = ridgeReg.predict(test_dataset)

# calculating mse

mse = np.sqrt(mean_squared_error(pred , test_dataset_Y))

print("The Root mean square error of Ridge Regression is ", mse)

print("The R2 value of Ridge Regression is ",r2_score(test_dataset_Y,pred))
from sklearn.linear_model import Lasso

lassoReg = Lasso(alpha=20, normalize=True)

lassoReg.fit(train_dataset_X,train_dataset_Y)

pred = lassoReg.predict(test_dataset)

# calculating mse

rmse = np.sqrt(mean_squared_error(pred, test_dataset_Y))

print("The Root mean square error of Lasso Regression is ", rmse)

print("The R2 value of Lasso Regression is ",r2_score(test_dataset_Y,pred))

#lassoReg.score(test_dataset,test_dataset_Y)


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(train_dataset_X,train_dataset_Y)
 
# Predicting a new result
y_pred = regressor.predict(test_dataset)

rmse = np.sqrt(mean_squared_error(y_pred, test_dataset_Y))

print("The Root mean square error of Decision Tree Regression is ", rmse)

print("The R2 value of Decision Tree Regression is ",r2_score(test_dataset_Y,y_pred))



from sklearn import neighbors


knn = neighbors.KNeighborsRegressor(5)

pred_test = knn.fit(train_dataset_X,train_dataset_Y).predict(test_dataset)

RMSE = np.sqrt(mean_squared_error(test_dataset_Y, pred_test))

print("The Root Mean Squared Error of KNN Regression is ",RMSE)

print("The R2 value of KNN Regression is ",r2_score(test_dataset_Y,pred_test))
# from sklearn.svm import SVR

# svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

# svr_lin = SVR(kernel='linear', C=1e3)

# svr_poly = SVR(kernel='poly', C=1e3, degree=3)

# y_rbf = svr_rbf.fit(train_dataset_X,train_dataset_Y).predict(test_dataset)

# y_lin = svr_lin.fit(train_dataset_X,train_dataset_Y).predict(test_dataset)

# y_poly = svr_poly.fit(train_dataset_X,train_dataset_Y).predict(test_dataset)

# RMSE_1 = np.sqrt(mean_squared_error(test_dataset_Y, y_rbf))

# print("The Root Mean Squared Error of SVM (Radial Basis Function) Regression is ",RMSE_1)

# print("The R2 value of SVM (Radial Basis Function) Regression is ",r2_score(test_dataset_Y,y_rbf))

# RMSE_2 = np.sqrt(mean_squared_error(test_dataset_Y, y_lin))

# print("The Root Mean Squared Error of SVM(Linear) Regression is ",RMSE_2)

# print("The R2 value of SVM(Linear) Regression is ",r2_score(test_dataset_Y,y_lin))

# RMSE_3 = np.sqrt(mean_squared_error(test_dataset_Y, y_poly))

# print("The Root Mean Squared Error of SVM(Polynomial) Regression is ",RMSE_3)

# print("The R2 value of SVM(Polynomial) Regression is ",r2_score(test_dataset_Y,y_poly))
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 25)
# Train the model on training data
pred = rf.fit(train_dataset_X,train_dataset_Y).predict(test_dataset)

RMSE_1 = np.sqrt(mean_squared_error(test_dataset_Y, pred))

print("The Root Mean Squared Error of Random Forest Regression is ",RMSE_1)

print("The R2 value of Random Forest Regression is ",r2_score(test_dataset_Y,pred))
from sklearn.ensemble import GradientBoostingRegressor

pred = GradientBoostingRegressor(n_estimators=100, learning_rate=0.3,max_depth=1, random_state=0, loss='ls').fit(train_dataset_X,train_dataset_Y).predict(test_dataset)

RMSE_1 = np.sqrt(mean_squared_error(test_dataset_Y, pred))

print("The Root Mean Squared Error of Gradient Boosting Regression is ",RMSE_1)

print("The R2 value of Gradient Boosting Regression is ",r2_score(test_dataset_Y,pred))