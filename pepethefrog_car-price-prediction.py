import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

%matplotlib inline
'''
Import the data and check the head
'''
data = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/vw.csv')
data.head()
'''
Show main numerical data and plot the most ambiguous column
'''
sns.boxplot(data['mpg'])
data.describe()
'''
Due to boxplot we can see some anomalies like 188 and 0.3 miles per galon
(equivalent to 1.25 and 784 liters per 100km)
So i desided to replace this values with mean value in this column

Also it is impossible to buy a car without taxes on it, so all examples with
tax value smaller than 25£ replace with mean value

One of the most economy cars (Smart) has engine size equal 0.9 liters, so all engine volumes
smaller than 0.9 replace with mean value

Now dataset is more clear
'''
perc99 = data.mpg.quantile(0.99)
perc01 = data.mpg.quantile(0.01)
data['mpg'][(data.mpg > perc99) | (data.mpg < perc01)] = data.mpg.mean()

data['engineSize'][data.engineSize < 0.9] = data.engineSize.mean()

data['tax'][data.tax < 25] = data.tax.mean()
sns.boxplot(data['mpg'])
data.describe()
colors = ['#000099', '#ffff00']
sns.heatmap(pd.isnull(data), cmap=sns.color_palette(colors))
'''
Now plot pairwise relationships in a dataset.
I can se here strong relationship between price and mpg,mileage,year columns
Also tax in relation with mpg
Of course there are more relations in the data but it is not very obvious 
'''
sns.pairplot(data)
data.head()
columns = ['model', 'year', 'transmission', 'fuelType']



onehot = OneHotEncoder(sparse=False)
enc = pd.DataFrame(onehot.fit_transform(data[columns]))
X = pd.concat([enc, data[['price', 'mileage', 'tax', 'mpg', 'engineSize']]], axis=1)
Y = X['price']
del(X['price'])
X.head()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=54)
model_ridge = Ridge(alpha=0.001)
model_ridge.fit(X_train, Y_train)
print('coefficient of determination:', model_ridge.score(X_test, Y_test))
temp = []
for i in np.arange(0.01, 1, 0.01):
    model_ridge = Ridge(alpha=i)
    model_ridge.fit(X_train, Y_train)
    temp.append(model_ridge.score(X_test, Y_test))
plt.plot(temp)
model_lasso = Lasso(alpha=0.2, max_iter=3000)
model_lasso.fit(X_train, Y_train)
print('coefficient of determination:', model_lasso.score(X_test, Y_test))
temp = []
for i in np.arange(0.1, 5, 0.1):
    model_lasso = Lasso(alpha=i, max_iter=1000)
    model_lasso.fit(X_train, Y_train)
    temp.append(model_lasso.score(X_test, Y_test))
plt.plot(temp)
model_en = ElasticNet(alpha=0.00001, l1_ratio=0.1)
model_en.fit(X_train, Y_train)
print('coefficient of determination:', model_en.score(X_test, Y_test))
temp = []
for i in np.arange(0.1, 1, 0.1):
    model_en = ElasticNet(alpha=0.00001, l1_ratio=i)
    model_en.fit(X_train, Y_train)
    temp.append(model_en.score(X_test, Y_test))
plt.plot(temp)
model_ridge.coef_
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline

model_sgd = make_pipeline(StandardScaler(),
                    SGDRegressor())

model_sgd.fit(X_train,Y_train)
model_sgd.score(X_test, Y_test)
label = LabelEncoder()
data['model'] = label.fit_transform(data['model'])
data['transmission'] = label.fit_transform(data['transmission'])
data['year'] = label.fit_transform(data['year'])
data['fuelType'] = label.fit_transform(data['fuelType'])
