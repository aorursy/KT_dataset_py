import pandas as pd
import numpy as np

df_train=pd.read_csv("../input/usa-housingcsv/USA_Housing.csv")
df_train.head()
df_train.info()
df_train.describe()
import seaborn as sns
sns.pairplot(df_train)
sns.distplot(df_train['Price'])
sns.heatmap(df_train.corr())
USA_housing=df_train
X = USA_housing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']] # the input features 
y = USA_housing['Price'] # the target or output 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) #validation data is of szie 0.3*total data set
#random state is for getting same output 
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
import matplotlib.pyplot as plt
plt.scatter(y_test,predictions)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
