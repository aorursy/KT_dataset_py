import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import linear_model
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import os
for dirname, _, filenames in os.walk('/kaggle/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
fp = '/kaggle/input/house_data - Copy.csv'
data = pd.read_csv(fp,parse_dates=True, index_col=0)
data.head()
data['date'] = pd.to_datetime(data.date).dt.year
data['yr_renovated'] = data['yr_renovated'].replace(to_replace = 0 , value= data['yr_built'].min())
data.head(10)
data1 = data.copy()
data1
data1.columns
dummy = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade']
data2 = pd.get_dummies(data1, prefix_sep='-', drop_first=True, columns=dummy).copy()
data2
X = data2.drop('price', axis =1)
y = data2['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.80, random_state = 1)
X_train.size/(X_train.size + X_test.size)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df
model.score(X_test, y_test)
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])  
coeff_df
data3 = data2.copy()
data3
corre = data3.corr()
plt.figure(figsize= (10,10))
sns.heatmap(corre)
#Better Approach
#A = data3.drop('price', axis=1)
#B = data3['price']
# Create correlation matrix
corr_matrix = data3.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
upper
to_drop
data4 = data3.drop(data3[to_drop], axis = 1).copy()
data4
#Better Approach
x1 = data4.drop('price', axis=1)
y1 = data4['price']
X_train1, X_test1, y_train1, y_test1 = train_test_split(x1, y1, train_size = 0.85, random_state = 2)
X_train1.size/(X_train1.size + X_test1.size)
model1 = LinearRegression(n_jobs=-1).fit(X_train1, y_train1)
predi = model1.predict(X_test1)
dfa = pd.DataFrame({'Actual': y_test1, 'Predicted': predi})
dfa
df2 = dfa.head(25)
df2.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test1, predi))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test1, predi))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test1, predi)))
model1.score(X_test1, y_test1)
x = data2.drop(['price'],axis=1)
y = data2[['price']]
x_train, x_test, y_train, y_test =train_test_split(x, y, test_size = 0.25, random_state = 0, shuffle = False)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
lasso_reg = linear_model.Lasso(alpha= 0.00010, tol=0.001, random_state=1, normalize=True, positive= True)
lasso_reg.fit(x_train, y_train)
y_pred = lasso_reg.predict(x_test)
lasso_reg.score(x_test, y_test)
rd = linear_model.Ridge(random_state=1).fit(x_train, y_train)
rd.predict(x_test)
rd.score(x_test, y_test)
params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}
reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(x_train, y_train)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, rd.predict(x_test)))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, rd.predict(x_test)))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, rd.predict(x_test))))
a= reg.score(x_test, y_test).round(4)*100
print('Score is :{}'.format(a)+'%')
