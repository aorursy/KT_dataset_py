import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data = pd.read_csv('../input/bostonhoustingmlnd/housing.csv')
data.head()
data.info()
data.describe()
data.isnull().values.any()
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap = 'viridis')
sns.set_palette("GnBu_d")
sns.set_style("whitegrid")
sns.pairplot(data,height=2)
sns.heatmap(data.corr(),annot= True)
sns.lmplot(x= 'RM',y= 'MEDV',data= data)
data.columns
X = data[['RM', 'LSTAT', 'PTRATIO']]
y = data['MEDV']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print('Intercept: \n', lm.intercept_)
print('Coefficients: \n', lm.coef_)
predictions = lm.predict(X_test)
sns.scatterplot(y_test,predictions)
plt.title('Y Test Vs Predicted Y')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test,predictions))
print('MSE:', metrics.mean_squared_error(y_test,predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,predictions)))
metrics.explained_variance_score(y_test,predictions)
sns.distplot((y_test - predictions))
data_coeff = pd.DataFrame(lm.coef_, X.columns, columns= ['Coeffecient'])
data_coeff
