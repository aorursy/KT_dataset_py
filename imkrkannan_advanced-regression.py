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
df=pd.read_csv("/kaggle/input/house_prices_1.csv")
df.head()
df.info()
print ("Train data shape:", df.shape)
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
df.averageprice.describe()
df.averagepricesemidetached.describe()
df.averagepricedetached.describe()
df.averagepriceterraced.describe()
df.averagepriceflatormaisonette.describe()
print ("Skew is:", df.averageprice.skew())
plt.hist(df.averageprice, color='black')
plt.show()
target = np.log(df.averageprice)
print ("Skew is:", target.skew())
plt.hist(target, color='red')
plt.show()
print ("Skew is:", df.averagepricesemidetached.skew())
plt.hist(df.averagepricesemidetached, color='orange')
plt.show()
target = np.log(df.averagepricesemidetached)
print ("Skew is:", target.skew())
plt.hist(target, color='black')
plt.show()
print ("Skew is:", df.averagepricedetached.skew())
plt.hist(df.averagepricedetached, color='orange')
plt.show()
target = np.log(df.averagepricedetached)
print ("Skew is:", target.skew())
plt.hist(target, color='black')
plt.show()
print ("Skew is:", df.averagepriceterraced.skew())
plt.hist(df.averagepriceterraced, color='orange')
plt.show()
target = np.log(df.averagepriceterraced)
print ("Skew is:", target.skew())
plt.hist(target, color='black')
plt.show()
print ("Skew is:", df.averagepriceflatormaisonette.skew())
plt.hist(df.averagepriceflatormaisonette, color='orange')
plt.show()
target = np.log(df.averagepriceflatormaisonette)
print ("Skew is:", target.skew())
plt.hist(target, color='black')
plt.show()
numeric_features = df.select_dtypes(include=[np.number])
numeric_features.dtypes
corr = numeric_features.corr()
print (corr['averageprice'].sort_values(ascending=False)[:5], '\n')
print (corr['averageprice'].sort_values(ascending=False)[-5:])
df.averageprice.unique()
df.averagepricedetached.unique()
df.averagepricesemidetached.unique()
df.averagepriceterraced.unique()
df.averagepriceflatormaisonette.unique()
plt.scatter(x=df['averageprice'], y=target)
plt.ylabel('averagepriceterraced')
plt.xlabel('Average price')
plt.show()
plt.scatter(x=df['averagepricesemidetached'], y=target)
plt.ylabel('averageprice')
plt.xlabel('Average price Semi detached')
plt.show()
plt.scatter(x=df['averagepricedetached'], y=target)
plt.ylabel('averageprice')
plt.xlabel('Average price Detached')
plt.show()
plt.scatter(x=df['averagepriceflatormaisonette'], y=target)
plt.ylabel('averageprice ')
plt.xlabel('Average price Flatormaisonette')
plt.show()
plt.scatter(x=df['averagepriceflatormaisonette'], y=np.log(df.averageprice))
plt.xlim(0,200000) # This forces the same scale as before
plt.ylabel('Average price')
plt.xlabel('Average price flatormaisonette')
plt.show()
nulls = pd.DataFrame(df.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls
print ("Unique values are:", df.geocode.unique())
categoricals = df.select_dtypes(exclude=[np.number])
categoricals.describe()
print ("Original: \n")
print (df.geoname.value_counts(), "\n")
condition_pivot = df.pivot_table(index='geoname', values='averageprice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='red')
plt.xlabel('Name of  the City')
plt.ylabel('average Sale Price')
plt.xticks(rotation=0)
plt.show()

def encode(x):
 return 1 if x == 'Partial' else 0
df['enc_condition'] = df.geoname.apply(encode)

condition_pivot = df.pivot_table(index='enc_condition', values='averageprice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Average price')
plt.xticks(rotation=0)
plt.show()
data = df.select_dtypes(include=[np.number]).interpolate().dropna()
sum(data.isnull().sum() != 0)
y = np.log(df.averageprice)
X = data.drop(['averageprice'], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))
actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.7,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()
for i in range (-2, 3):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, y_train)
    preds_ridge = ridge_model.predict(X_test)

    plt.scatter(preds_ridge, actual_values, alpha=.75, color='green')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(ridge_model.score(X_test, y_test),mean_squared_error(y_test, preds_ridge))
    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
    plt.show()