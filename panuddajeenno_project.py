# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
apps = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
apps.head()
#เช็คค่า null 
apps.isna().sum()
#ลบค่า NaN ออก
apps.dropna(how='any',inplace=True)
apps.Rating.unique()
apps.dtypes
apps = apps.astype({'Reviews':'int'})
apps.dtypes
apps.Installs = apps.Installs.apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
apps.Installs = apps.Installs.apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
apps.Installs = apps.Installs.apply(lambda x: int(x))
apps.Size = apps.Size.apply(lambda x: x.replace('M', '000') if 'M' in x else x)
apps.Size = apps.Size.apply(lambda x: x.replace('k','') if 'k' in str(x) else x)
apps.Size = apps.Size.apply(lambda x: x.replace('Varies with device','0') if 'Varies with device' in str(x) else x)
apps.Size = apps.Size.apply(lambda x: float(x))
apps.Price = apps.Price.apply(lambda x: x.replace('$','') if '$' in str(x) else x)
apps.Price = apps.Price.astype(float)
apps.Reviews = apps.Reviews.apply(lambda x: int(x))
apps.info()
apps['Rating'].describe()
apps = apps.drop(['App','Last Updated','Current Ver','Android Ver'],axis='columns')
apps.head()
# Check correlations
sns.heatmap(apps.corr(), annot=True)
plt.figure(figsize = (10,10))
sns.regplot(x="Price", y="Rating", color = 'darkorange',data=apps[apps['Reviews']<1000000]);
plt.title('Scatter plot Rating VS Price',size = 20)
g = sns.catplot(x="Content Rating",y="Rating",data=apps, kind="box", height = 10 ,palette = "Paired")
g.despine(left=True)
g.set_xticklabels(rotation=90)
g = g.set_ylabels("Rating")
plt.title('Box plot Rating VS Content Rating',size = 20)
plt.figure(figsize = (15,7))
sns.regplot(x="Rating", y="Reviews", color = 'blue',data=apps[apps['Reviews']<100000]);
plt.title('Reviews Vs Rating',size = 15)
plt.figure(figsize = (12,7))
sns.boxplot(x='Content Rating', y='Rating', hue='Type', data=apps, palette='PRGn')
plt.show()
categories = ['Category', 'Type', 'Content Rating', 'Genres']
apps1 = pd.get_dummies(apps.copy(), columns=categories,drop_first=True)
# กำหนด feature ของ X และ y
X = apps1.drop(columns=['Rating'],axis=1)
y = apps1['Rating']

# แบ่ง X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2, random_state=20)
dataTrain = LinearRegression()  
dataTrain.fit(X_train, y_train) #training the algorithm
y_pred = dataTrain.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.ensemble import RandomForestRegressor
datardf = RandomForestRegressor().fit(X_train,y_train) # Fitting the model.
predictions = datardf.predict(X_test) # Test set is predicted.
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions)) 
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions))) 