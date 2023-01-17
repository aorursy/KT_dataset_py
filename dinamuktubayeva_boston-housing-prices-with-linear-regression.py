import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # data visualization
import sklearn # machine learning
from sklearn import metrics # error evaluation
from sklearn.datasets import load_boston # dataset
from sklearn.model_selection import train_test_split # splitting data
from sklearn.linear_model import LinearRegression # linear regression model


#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))
boston = load_boston()
data = pd.DataFrame(boston.data)
print(boston.DESCR)
data.head()
data.columns = boston.feature_names
data.head()
data.isnull().sum()
x_train, x_test, y_train, y_test = train_test_split(data, boston.target, test_size=0.8, random_state=10)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
for feature in x_train.columns:
    plt.scatter(x_train[feature], y_train)
    plt.xlabel(feature)
    plt.ylabel('price')
    plt.show()
features = ['CRIM','NOX','RM','LSTAT']
x = x_train[features]
y = y_train
print(x.shape)
print(y.shape)
model = LinearRegression()
model.fit(x, y)
predictions = model.predict(x_test[features])

plt.figure(figsize=(10,10))
plt.scatter(y_test, predictions)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('Actual price ($1000s)')
plt.ylabel('Predicted price ($1000s)')
plt.tight_layout()
res = pd.DataFrame({'Actual': y_test, 'Predicted': predictions, 'Difference': y_test-predictions})
res['Difference'].describe()
res['Difference'].plot(kind="bar", figsize=(20,10))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('R squared error:', metrics.r2_score(y_test, predictions))
sns.distplot((res['Difference']),bins=50);