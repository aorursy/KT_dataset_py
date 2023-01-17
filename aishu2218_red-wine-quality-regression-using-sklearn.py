import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
wine= pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

wine.head()
wine.shape
wine.info()
wine.describe()
sns.pairplot(wine)
plt.figure(figsize=(15,10))

sns.heatmap(wine.corr(),annot=True)
X = wine.drop('quality',axis=1)

y = wine['quality']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=22)
from sklearn.linear_model import LinearRegression
# Fit the model

lm = LinearRegression()

lm.fit(X_train,y_train)
lm.coef_
coeff = pd.DataFrame(lm.coef_, X_train.columns, columns=['Coefficient'])

coeff
lm.intercept_
pred = lm.predict(X_test)
sns.distplot((y_test-pred),bins=30)

plt.title('Actual vs Predictions')
df= pd.DataFrame({'Actual':y_test,'Predictions':pred})

df['Predictions']= round(df['Predictions'],2)

df.head()
fig, ax = plt.subplots()

ax.scatter(y_test,pred)

ax.set_xlabel('Actual')

ax.set_ylabel('Predicted')

plt.show()
sns.regplot('Actual','Predictions',data=df)
from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, pred))

print('MSE:', metrics.mean_squared_error(y_test, pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
print('R squared: ',lm.score(X_train,y_train))