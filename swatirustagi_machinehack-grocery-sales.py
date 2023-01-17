#importing libraries

import pandas as pd

import numpy as np

from scipy import stats

from scipy.stats import kurtosistest



import plotly.express as px

import matplotlib.pyplot as plt

import seaborn as sns

from pandas.plotting import scatter_matrix

import warnings



%matplotlib inline

warnings.simplefilter("ignore")
#reading the data

train = pd.read_csv("/kaggle/input/grocery-sales-forecast-weekend-hackathon/Grocery_Sales_ParticipantsData/Train.csv")

test = pd.read_csv("/kaggle/input/grocery-sales-forecast-weekend-hackathon/Grocery_Sales_ParticipantsData/Test.csv")
#checking on the basic

print(train.head(2))

print("*"*50)

print(train.isnull().sum())

print("*"*50)

print("The shape of Train Dataset is:",train.shape)
train.describe()
#checking normality of data

k2, p = stats.normaltest(train['GrocerySales'])

alpha = 1e-3

print("p = {:g}".format(p))

p = 3.27207e-11

if p < alpha:  # null hypothesis: x comes from a normal distribution

    print("Data Not Normalize")

else:

    print("Data Noramlize")
#correlation comparison 

corr = train.corr()

corr.style.background_gradient(cmap='coolwarm')
#scatter matrix

scatter_matrix(train)

plt.show()
#checking on outliers

plt.figure(figsize=(5,5))

sns.boxplot(x= 'variable', y = 'value', data = pd.melt(train[['GrocerySales']]))
#treemap

fig = px.treemap(train, path=['GrocerySales'],color='GrocerySales', hover_data=['Day', 'GrocerySales'])

fig.show()
#scatter plot

fig = px.scatter(train, x="Day", y="GrocerySales", trendline="ols")

fig.show()
#Grocery Sales density plot 

sns.set_style("darkgrid")

sns.kdeplot(data=train['GrocerySales'],label="GrocerySales" ,shade=True)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error
X = train['Day']

y = train['GrocerySales']

X_pred = test['Day']
X = np.array(X)

X = X.reshape(-1, 1)
X_pred = np.array(X_pred)

X_pred = X_pred.reshape(-1, 1)
#scaling the data

rs = RobustScaler()
X_scaled = rs.fit_transform(X)

X_pred_scaled = rs.fit_transform(X_pred)
train_X, test_X, train_y, test_y = train_test_split(X, y,test_size=0.3,random_state = 42)
lr = LinearRegression()

lr.fit(train_X,train_y)
predict = lr.predict(test_X)

print('Mean Squared Error: %.2f'% mean_squared_error(test_y,predict))

print('Mean Absolute Error: %.2f'% mean_absolute_error(test_y,predict))
y_pred_lr = lr.predict(X_pred)
y_pred_lr
knn = KNeighborsRegressor()

knn.fit(train_X,train_y)
predict_knn = knn.predict(test_X)

print('Mean Squared Error: %.2f'% mean_squared_error(test_y,predict_knn))

print('Mean Absolute Error: %.2f'% mean_absolute_error(test_y,predict_knn))
y_pred_knn = knn.predict(X_pred)
y_pred_knn
scaled_train_X, scaled_test_X, train_y, test_y = train_test_split(X_scaled, y, test_size=0.3,random_state = 42)
lr = LinearRegression()

lr.fit(scaled_train_X,train_y)
scaled_predict = lr.predict(scaled_test_X)

print('Mean Squared Error: %.2f'% mean_squared_error(test_y,scaled_predict))

print('Mean Absolute Error: %.2f'% mean_absolute_error(test_y,scaled_predict))
y_pred_scaled_lr = lr.predict(X_pred_scaled)
y_pred_scaled_lr
knn = KNeighborsRegressor()

knn.fit(scaled_train_X,train_y)
scaled_predict_knn = knn.predict(scaled_test_X)

print('Mean Squared Error: %.2f'% mean_squared_error(test_y,scaled_predict_knn))

print('Mean Absolute Error: %.2f'% mean_absolute_error(test_y,scaled_predict_knn))
y_pred_scaled_knn = knn.predict(X_pred_scaled)
y_pred_scaled_knn
sns.distplot(test_y-predict)

plt.title("Histogram of Residuals")

plt.xlabel("Residuals")

plt.ylabel("Frequency")

plt.show()
sns.distplot(test_y-predict_knn)

plt.title("Histogram of Residuals")

plt.xlabel("Residuals")

plt.ylabel("Frequency")

plt.show()
sns.distplot(test_y-scaled_predict)

plt.title("Histogram of Residuals")

plt.xlabel("Residuals")

plt.ylabel("Frequency")

plt.show()
sns.distplot(test_y-scaled_predict_knn)

plt.title("Histogram of Residuals")

plt.xlabel("Residuals")

plt.ylabel("Frequency")

plt.show()
df1 = pd.DataFrame(y_pred_lr,columns=['GrocerySales_lr'])

df2 = pd.DataFrame(y_pred_knn,columns=['GrocerySales_knn'])

df3 = pd.DataFrame(y_pred_scaled_lr,columns=['GrocerySales_lrs'])

df4 = pd.DataFrame(y_pred_scaled_knn,columns=['GrocerySales_knns'])
temp = [test, df1, df2, df3, df4]
test_result = pd.concat(temp, axis=1, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, copy=True)
test_result.head(2)
fig = px.scatter(x=test_result.Day, y=test_result.GrocerySales_knn)

fig.show()
fig = px.scatter(x=test_result.Day, y=test_result.GrocerySales_knns)

fig.show()
fig = px.scatter(x=test_result.Day, y=test_result.GrocerySales_lr)

fig.show()
fig = px.scatter(x=test_result.Day, y=test_result.GrocerySales_lrs)

fig.show()