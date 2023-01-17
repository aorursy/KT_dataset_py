import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/housedata/data.csv')
df.head(5)
df.columns
df.describe()
df.info()
df.isnull()
df.isnull().sum()
sns.distplot(df['price'])
plt.scatter(df.price, df.sqft_living)
plt.title("price vs living area")
plt.scatter(df.price, df.sqft_lot)
plt.title("price vs sqft_lot")
plt.scatter(df.price, df.waterfront)
plt.title("price vs waterfront")
plt.xlabel("waterfront")
plt.ylabel("price")
plt.show()
plt.scatter(df.price, df.condition)
plt.title("price vs condition")
plt.xlabel("condition")
plt.ylabel("price")
plt.show()
sns.pairplot(df)
sns.heatmap(df.corr(), annot=True)
one_hot_encoded_df = pd.get_dummies(df)

X = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
       'floors', 'waterfront', 'view', 'condition', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101) 


lm = LinearRegression() 

lm.fit(X_train,y_train) 
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50); 
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions)) 
print('MSE:', metrics.mean_squared_error(y_test, predictions)) 
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions))) 