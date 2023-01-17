import pandas as pd
import numpy as np

data = pd.read_csv("../input/brasilian-houses-to-rent/houses_to_rent_v2.csv")
data.head()
data.info()
# Convert datatype of floor into numeric value.
data['floor'] = pd.to_numeric(data['floor'], errors = 'coerce')
data.info()
data.isnull().sum()
# in floor I am replacing the null value with 0 as a ground floor.
data = data.fillna(0)
print(len(data))
print(data.describe())
# check the outliers in rooms, bathroom,floor
import seaborn as sns
import matplotlib.pyplot as plot

f,axes = plot.subplots(1,3)

sns.boxplot(y = 'floor', data = data, ax=axes[0])
sns.boxplot(y = 'rooms', data = data, ax = axes[1])
sns.boxplot(y = 'bathroom', data = data, ax = axes[2])

# outlier treatment:

q1 = data.floor.quantile(0.25)
q3 = data.floor.quantile(0.75)
IQR = q3-q1
data_1 = data[(data.floor >= q1-1.5*IQR) & (data.floor <= q3 + 1.5 * IQR)]

q1 = data_1.rooms.quantile(0.25)
q3 = data_1.rooms.quantile(0.75)
IQR = q3-q1
data_1 = data_1[(data_1.rooms >= q1-1.5*IQR) & (data_1.rooms <= q3 + 1.5 * IQR)]

q1 = data_1.bathroom.quantile(0.25)
q3 = data_1.bathroom.quantile(0.75)
IQR = q3-q1
data_1 = data_1[(data_1.bathroom >= q1-1.5*IQR) & (data_1.bathroom <= q3 + 1.5 * IQR)]

print(len(data_1))

f,axes = plot.subplots(1,3)

sns.boxplot(y = 'floor', data = data_1, ax=axes[0])
sns.boxplot(y = 'rooms', data = data_1, ax = axes[1])
sns.boxplot(y = 'bathroom', data = data_1, ax = axes[2])
plot.tight_layout()
sns.distplot(data_1['rent amount (R$)'],bins = 30)
# Data has categorical variable. so I am converting categorical variable into numeric.
data_1 = pd.get_dummies(data = data_1 , columns = ['furniture','animal'])
print(data_1.head())
print(len(data_1))
data_1.groupby('city').size()

# drop the city column.

data_1 = data_1.drop(['city'],axis = 1)

# Need to scale the dataset.

def normalize(x):
    return ((x- np.mean(x))/(max(x)-min(x)))

data_1 = data_1.apply(normalize)
data_1.head()

import seaborn as sns
fig,ax = plot.subplots(figsize=(10,10))
cal_corr = data_1.corr().round(2)
sns.heatmap(cal_corr,annot = True, linewidths = 1, ax=ax)
xData = pd.DataFrame(data_1[['area','rooms','bathroom','parking spaces','fire insurance (R$)','furniture_furnished','furniture_not furnished','hoa (R$)','total (R$)']], columns = ['area','rooms','bathroom','parking spaces','fire insurance (R$)','furniture_furnished','furniture_not furnished','hoa (R$)','total (R$)'])
print(xData.head())
yData = pd.DataFrame(data_1['rent amount (R$)'], columns = ['rent amount (R$)'])
print(yData.head())
# import the library for spliting the data.
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(xData,yData, train_size = 0.7, test_size = 0.3,random_state = 5)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
import statsmodels.api as sm

x_train = sm.add_constant(x_train)

lm_model1 = sm.OLS(y_train,x_train).fit()
print(lm_model1.summary())
x_train = x_train.drop(['rooms'],1)
lm_model2 = sm.OLS(y_train,x_train).fit()
print(lm_model2.summary())
x_train = x_train.drop(['fire insurance (R$)'],1)
lm_model3 = sm.OLS(y_train,x_train).fit()
print(lm_model3.summary())
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
ytrain_predic = lm_model3.predict(x_train)
rmse = (np.sqrt(mean_squared_error(y_train,ytrain_predic))).round(3)
r2 = r2_score(y_train,ytrain_predic).round(3)
print('RMSE for training data is : {}'.format(rmse))
print('r2 for training data is : {}'. format(r2))

# for test dataset we need to drop the columns which we drop during building the model.
x_test_model3 = sm.add_constant(x_test)
x_test_model3 = x_test_model3.drop(['rooms','fire insurance (R$)'], axis = 1)
ytest_predic = lm_model3.predict(x_test_model3)
rmse = (np.sqrt(mean_squared_error(y_test,ytest_predic))).round(3)
r2 = r2_score(y_test,ytest_predic).round(3)
print('RMSE for test data is : {}'.format(rmse))
print('r2 for test data is : {}'. format(r2))
