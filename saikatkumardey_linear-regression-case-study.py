import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')





from sklearn import metrics



import numpy as np



# allow plots to appear directly in the notebook

%matplotlib inline
data = pd.read_csv('../input/advertising.csv/Advertising.csv', index_col=0)

data.head(100)
data.shape
data.info()
data.describe()
f, axes = plt.subplots(2, 2, figsize=(15, 7), sharex=False)            # Set up the matplotlib figure

sns.despine(left=True)



sns.distplot(data.sales, color="b", ax=axes[0, 0])



sns.distplot(data.TV, color="r", ax=axes[0, 1])



sns.distplot(data.radio, color="g", ax=axes[1, 0])



sns.distplot(data.newspaper, color="m", ax=axes[1, 1])
JG1 = sns.jointplot("newspaper", "sales", data=data, kind='reg')
JG2 = sns.jointplot("radio", "sales", data=data, kind='reg')
JG3 = sns.jointplot("TV", "sales", data=data, kind='reg')
sns.pairplot(data, height = 2, aspect = 1.5)
sns.pairplot(data, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', size=5, aspect=1, kind='reg')
data.corr()
plt.figure(figsize=(7,5))

sns.heatmap(round(data.corr(),2),annot=True)

plt.show()
features = ['TV', 'radio', 'newspaper']                # create a Python list of feature names

target = ['sales']                                     # Define the target variable
data.head()
data[features]
data[target]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.05, random_state=5000)
print('Train cases as below')

print('X_train shape: ',X_train.shape)

print('y_train shape: ',y_train.shape)

print('\nTest cases as below')

print('X_test shape: ',X_test.shape)

print('y_test shape: ',y_test.shape)
X_train.head()
y_train.head()
X_test.head()
y_test.head()
#Instantiating the model

from sklearn.linear_model import LinearRegression

lr_model = LinearRegression(fit_intercept=True)
lr_model.fit(X_train, y_train)
print('Intercept:',lr_model.intercept_)          # print the intercept 

print('Coefficients:',lr_model.coef_)  
X_train.columns
(lr_model.coef_).T
pd.DataFrame((lr_model.coef_).T,index=X_train.columns,\

             columns=['Co-efficients']).sort_values('Co-efficients',ascending=False)
y_pred_train = lr_model.predict(X_train)  
y_pred_train                                                         # make predictions on the training set
y_pred_test = lr_model.predict(X_test)                                  # make predictions on the testing set
y_pred_test
MAE_train = metrics.mean_absolute_error(y_train, y_pred_train)

MAE_test = metrics.mean_absolute_error(y_test, y_pred_test)
print('MAE for training set is {}'.format(MAE_train))

print('MAE for test set is {}'.format(MAE_test))
MSE_train = metrics.mean_squared_error(y_train, y_pred_train)

MSE_test = metrics.mean_squared_error(y_test, y_pred_test)
print('MSE for training set is {}'.format(MSE_train))

print('MSE for test set is {}'.format(MSE_test))
RMSE_train = np.sqrt( metrics.mean_squared_error(y_train, y_pred_train))

RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))
print('RMSE for training set is {}'.format(RMSE_train))

print('RMSE for test set is {}'.format(RMSE_test))
data['sales'].mean()
RMSE_test/data['sales'].mean()
np.random.seed(123456)                                                # set a seed for reproducibility

nums = np.random.rand(len(data))

mask_suburban = (nums > 0.33) & (nums < 0.66)                         # assign roughly one third of observations to each group

mask_urban = nums > 0.66

data['Area'] = 'rural'

data.loc[mask_suburban, 'Area'] = 'suburban'

data.loc[mask_urban, 'Area'] = 'urban'



data.head(50)
data.groupby(['Area'])['sales'].mean().sort_values(ascending=False).plot(kind = 'bar')
a = sns.scatterplot(x="TV", y="sales", data=data, hue='Area')
a = sns.scatterplot(x="Area", y="sales", data=data)
#data.to_csv("data_with_area.csv")
features = ['TV', 'radio', 'newspaper', 'Area']

cat_cols = ['Area']                                           # Define the categorical variables
data_with_dummies = pd.get_dummies(data, columns=cat_cols, drop_first=True)

data_with_dummies.head()
feature_cols = ['TV', 'radio', 'newspaper', 'Area_suburban', 'Area_urban']             # create a Python list of feature names

X = data_with_dummies[feature_cols]  

y = data_with_dummies.sales

lr_model_cat = LinearRegression()
lr_model_cat.fit(X,y)
y_pred_cat = lr_model_cat.predict(X)  
pd.DataFrame((lr_model_cat.coef_).T,index=X.columns,\

             columns=['Co-efficients']).sort_values('Co-efficients',ascending=False)
print('Intercept:',lr_model_cat.intercept_)
data_with_dummies['predictions'] = y_pred_cat
data_with_dummies
data_with_dummies['error'] = data_with_dummies['sales'] - data_with_dummies['predictions']
data_with_dummies['error'].describe()
data_with_dummies.plot.scatter(x='sales', y='predictions',\

                      figsize=(8,5), grid=True, title='Actual vs Predicted')
sns.distplot(data_with_dummies['error'])
data_with_dummies[data_with_dummies['error']<-4]
data_with_dummies.plot.scatter(x='sales', y='error',\

                      figsize=(8,5), grid=True, title='Actual vs Predicted')
data_with_dummies.to_csv('data_with_predictions.csv')