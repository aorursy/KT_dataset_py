import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
USAhousing = pd.read_csv('../input/usa-housing-dataset/USA_Housing_dataset.csv')
USAhousing.head()
USAhousing.info()
1.05e+6
USAhousing.describe()
1.23e6
USAhousing.columns
sns.pairplot(USAhousing)
sns.distplot(USAhousing['Price'])
sns.heatmap(USAhousing.corr(),annot=True)
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
prediction = model.fit(X_train,y_train)
# a.score(X_train,y_train)
# print the intercept
print(model.intercept_)
model.coef_
X.columns
coeff_df = pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])
coeff_df
predictions = model.predict(X_test)
predictions[0]
X_test.iloc[1718]
y_test[1718 ]
plt.scatter(y_test,predictions)

sns.distplot((y_test-predictions),bins=50);
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
model.score(X, y)