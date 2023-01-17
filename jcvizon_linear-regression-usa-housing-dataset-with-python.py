'''
Import all the needed libraries
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df=pd.read_csv('../input/USA_Housing.csv')
'''
A quick view of the dataset
'''
df.head(10)
df.info()
df.describe()
df.columns
sns.pairplot(df)
sns.distplot(df['Price'])
sns.heatmap(df.corr())
'''
X = Features
y = Target
'''
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101) #split tha features and target data into 60% 40%
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))
from sklearn.linear_model import LinearRegression
LRmodel=LinearRegression()
LRmodel.fit(X_train,y_train)
# print the intercept
print(LRmodel.intercept_)
LRmodel.coef_
coeff_df=pd.DataFrame(LRmodel.coef_,X.columns,columns=['Coefficient'])
coeff_df
predictions=LRmodel.predict(X_test)
import matplotlib.cm as cm
color=cm.rainbow(len(y))
plt.figure(figsize=(20,20))
plt.scatter(y_test,predictions,color=['green','red'])
sns.distplot((y_test-predictions),bins=50)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
