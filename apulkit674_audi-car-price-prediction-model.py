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
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn import metrics
path = r'/kaggle/input/used-car-dataset-ford-and-mercedes/audi.csv'

data = pd.read_csv(path)
print(data.head())
print(data.tail())
print(data.isnull().sum())
print(data.columns)
data.describe()
# Check how many categories are present in model column
data['model'].value_counts()
data['transmission'].value_counts()
data['fuelType'].value_counts()
from sklearn import preprocessing
converter = preprocessing.LabelEncoder()
data['fuelTypeNumeric'] = converter.fit_transform(data['fuelType'])
print(data.head())
data['transmission_numeric'] = converter.fit_transform(data['transmission'])
print(data.head())
data = pd.get_dummies(data,columns=['model'])
print(data.head())
sns.relplot(x="price",y="year",data=data)
sns.scatterplot(x='transmission',y='price',data=data)
X = data.drop(['price','transmission','fuelType'],axis=1)
y = data['price']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=25)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
regressor.score(X,y)
results = X_test.copy()
results["predicted"] = regressor.predict(X_test)
results["actual"]= y_test.copy()
results = results[['predicted', 'actual']]
results['predicted'] = results['predicted'].round(3)
results
regressor.coef_
regressor.intercept_
y_test = results['actual']
y_pred = results['predicted']

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))