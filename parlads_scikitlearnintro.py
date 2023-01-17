# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from sklearn import (datasets, model_selection)
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})


import matplotlib.pyplot as plt

from statsmodels.stats.outliers_influence import variance_inflation_factor
iris= datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df['tgt'] = iris.target
iris_df.head()
part1 , part2 = model_selection.train_test_split(iris_df)
len(iris_df), len(part1), len(part2)
from sklearn import neighbors
my_model = neighbors.KNeighborsClassifier()
my_model.fit(part1.drop(columns = 'tgt'), part1.tgt)
my_model.predict(part2.drop(columns = 'tgt'))
from sklearn import metrics

preds = my_model.predict(part2.drop(columns='tgt'))
metrics.accuracy_score(preds, part2.tgt)
cm = metrics.confusion_matrix(part2.tgt, preds)
cm
import seaborn as sns

%matplotlib inline
sns.heatmap(cm)
#load the csv
data  = pd.read_csv('/kaggle/input/satandgpa-lr/SATandGPA_LinearRegression.csv')
data.head()
x = data['SAT'] # input
y = data['GPA'] # output
x_matrix = x.values.reshape(-1,1)
print(x.shape)
y.shape
reg = LinearRegression()
reg.fit(x_matrix, y)
#r-square
reg.score(x_matrix,y)
#cofficient
reg.coef_
#intercept
reg.intercept_
new_data = pd.DataFrame(data=[1740, 1760], columns=['SAT'])
new_data
reg.predict(new_data)
new_data['Predicted GPA'] = reg.predict(new_data)
new_data
#plot it in scatter

plt.scatter(x,y)
yhat = reg.coef_*x_matrix + reg.intercept_
fig = plt.plot(x, yhat, lw = 4 , c= 'orange' , label = 'reg line')
plt.xlabel('SAT Score', fontsize = 20)
plt.ylabel('Student GPA', fontsize = 20)
# plt.xlim(0)
# plt.ylim(0)
plt.show()
data = pd.read_csv('/kaggle/input/multiplelineregressionsampledata/1.02. Multiple linear regression.csv')
data
data.describe()
#independent and dependent

x = data[['SAT', 'Rand 1,2,3']]
y = data['GPA']
reg = LinearRegression()
reg.fit(x,y)
reg.coef_
reg.score(x, y)
reg.intercept_
#calc r-squar
reg.score(x, y)
# R-square Formula

x.shape
r2 = reg.score(x,y)
n = x.shape[0]
p = x.shape[1]

adjusted_r2 = 1-(1 -r2) * (n-1)/(n-p-1)
adjusted_r2
f_regression(x,y) # it will produce two array , one is stastices and anothere is p value. P value is demed as more important
p_value = f_regression(x,y)[1]
p_value.round(3)
reg_summary  = pd.DataFrame(data= x.columns.values , columns=['Features'])
reg_summary
reg_summary ['cofficient'] = reg.coef_
reg_summary ['p-value'] = p_value.round(3)
reg_summary
#using standerScaller
scaler = StandardScaler()
scaler.fit(x)
x_scale = scaler.transform(x)
x_scale
#regression with scale

reg = LinearRegression()
reg.fit(x_scale,y)
reg.coef_
reg.intercept_
reg_summary = pd.DataFrame([['Intercept'] , ['SAT'] , ['Rand 1,2,3']] , columns=['Features'])
reg_summary['Weights'] = reg.intercept_, reg.coef_[0], reg.coef_[1]
reg_summary # output : intercept is call bias, it's nothing but bias
new_data = pd.DataFrame(data = [[1700,2],[1800,1]], columns = ['SAT', 'Rand 1,2,3'])
new_data
reg.predict(new_data)
#Train Test split
a = np.arange(1,101)
a
b = np.arange(501, 601)
b
a_train, a_test, b_train, b_test = train_test_split(a,b, test_size = 0.2, random_state = 42)
a_train.shape, a_test.shape
a_train
a_test
raw_data = pd.read_csv('/kaggle/input/input-file/1.04. Real-life example.csv')
raw_data.head()
raw_data.describe(include='all')
#data cleaning 
data = raw_data.drop(['Model'], axis=1) # droping from the column
data.describe(include='all')
data.isnull().sum()
# droping from the row
data_no_mv= data.dropna(axis=0)
data_no_mv.describe()
# we are trying to dig the outliers because from the above discriptors price have bigger jump between data like min and max p[rice]
sns.distplot(data_no_mv['Price']) 
#solution ==> remove top 1 percent of observations
q = data_no_mv['Price'].quantile(0.99)
data_1  = data_no_mv[data_no_mv['Price'] < q]
data_1.describe()
sns.distplot(data_no_mv['Price']) 
sns.distplot(data_no_mv['Mileage']) 
# milage has the same problem as price, so we are doing same things
q = data_1['Mileage'].quantile(0.99)
data_2  = data_1[data_1['Mileage'] < q]
sns.distplot(data_2['Mileage']) 
# EngineV has the same problem as price, so we are doing same things

data_3  = data_2[data_2['EngineV'] < 6.5]
sns.distplot(data_3['EngineV']) 
# for year, we are only keeping value greater then 1 percentile

q = data_3['Year'].quantile(0.01)
data_4  = data_3[data_3['Year'] > q]
sns.distplot(data_4['Year']) 
data_cleaned = data_4.reset_index(drop=True)
data_cleaned.describe(include='all')
# checking the linerity

f, (ax1, ax2,ax3) = plt.subplots(1,3,sharey = True, figsize = (15,3))
ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
ax1.set_title('price and year')

ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
ax2.set_title('price and Engine')

ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
ax3.set_title('price and Mileage')
# since plot are not quite linear , we are going to use log transformations
log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price
# checking the linerity

f, (ax1, ax2,ax3) = plt.subplots(1,3,sharey = True, figsize = (15,3))
ax1.scatter(data_cleaned['Year'], data_cleaned['log_price'])
ax1.set_title('price and year')

ax2.scatter(data_cleaned['EngineV'], data_cleaned['log_price'])
ax2.set_title('price and Engine')

ax3.scatter(data_cleaned['Mileage'], data_cleaned['log_price'])
ax3.set_title('price and Mileage')

plt.show()
# data_cleaned = data_cleaned.drop(['Price'], axis=1)
data_cleaned.head(10)
data_cleaned.columns.values
variables = data_cleaned[['Mileage', 'Year', 'EngineV']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])]
vif['features'] = variables.columns
vif 
# vif : vif between 1 < VIF 5 : perfectly okay
# vif : vif = 1 : no Multicollinearity
# vif : vif :  VIF > 5 : unacceptable
#now year has way more effect on model, so we drop it, otherwise it will be overfitting
data_no_Multicollinearity = data_cleaned.drop(['Year'], axis=1)
data_with_dummies = pd.get_dummies(data_no_Multicollinearity, drop_first=True)
data_with_dummies.head()
data_with_dummies.columns.values
cols = ['log_price', 'Mileage', 'EngineV','Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']
data_preprocessed = data_with_dummies[cols]
data_preprocessed.head()
targets= data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'], axis=1)
scaler = StandardScaler()
scaler.fit(inputs)
input_scaled = scaler.transform(inputs)
x_train, x_test , y_train, y_test  = train_test_split(input_scaled, targets, test_size  = 0.2, random_state = 42)
reg = LinearRegression()
reg.fit(x_train, y_train)
y_hat  = reg.predict(x_train)
plt.scatter(y_train, y_hat)

plt.xlabel('target (y_train)', size =20)
plt.ylabel('Predictions (y_hat)', size =20)

plt.xlim(6, 13)
plt.ylim(6, 13)

plt.show()
sns.distplot(y_train - y_hat)
plt.title('Residual PDF', size =20)

plt.show()
reg.score(x_train, y_train)
reg_summary  = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['weights'] = reg.coef_
reg_summary
data_cleaned['Brand'].unique()
y_hat_test = reg.predict(x_test)
plt.scatter(y_test, y_hat_test)

plt.xlabel('target (y_test)', size =20)
plt.ylabel('Predictions (y_hat_test)', size =20)

plt.xlim(6, 13)
plt.ylim(6, 13)

plt.show()
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Predictions'])
df_pf.head()
df_pf['Target'] = np.exp(y_test)
df_pf.head()
y_test.head(20)
y_test = y_test.reset_index(drop=True)
y_test.head(20)
df_pf['Target'] = np.exp(y_test)
df_pf
df_pf['Residual'] = df_pf['Target'] - df_pf['Predictions']
df_pf['Differences%'] = np.absolute(df_pf['Residual']/df_pf['Target'] *100)
df_pf
df_pf.describe()
pd.options.display.max_rows = 999
pd.set_option('display.float_format', lambda x: '%.2f' % x)
df_pf.sort_values(by=['Differences%'])