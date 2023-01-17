import pandas as pd
import numpy as np
import math
import xlrd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
 

data=pd.read_csv("../input/tesla-stock/TESLA_STOCK.csv")
print(data.head())
data.describe()
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(data['Open'])
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(data['High'])
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(data['Low'])
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(data['Close'])
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(data['Adj Close'])
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(data['Volume'])
Close = data["Close"] 
print(data.isnull().sum())
sns.lineplot(x='Open', y='Close', data=data)
plt.show()
sns.pairplot(data[["Open", "High", "Low", "Close", "Volume"]], diag_kind="kde")
plt.show()
print ("Find most important features relative to target")
corr = data.corr()
corr.sort_values(['Close'], ascending=False, inplace=True)
print(corr.Close) 
top_feature = corr.index[abs(corr['Close']>0.5)]
top_corr = data[top_feature].corr()
ax = sns.heatmap(top_corr, annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()
y=data["Close"]
print(y)
xdata=["Open", "High", "Low", "Volume"]
x=data[xdata]
print(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=3)
regressor=LinearRegression()
regressor.fit(x_train, y_train)
coeff_data = pd.DataFrame(regressor.coef_, x.columns, columns=['Coefficient'])
print(coeff_data) 
accuracy = regressor.score(x_test, y_test)
print('Accuracy: \n {}%'.format(int(accuracy * 100)))
print("Intercept: \n", regressor.intercept_)
print ("R^2 is: \n", regressor.score(x_test, y_test))
predictions = regressor.predict(x_test)
print ('RMSE is: \n', mean_squared_error(y_test, predictions))
y_pred = regressor.predict(x_test)
df=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df.head(10)) 
df1 = df.head(30)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
print("Price on day was 03.03.2020 is: \n Open 805 \n Close 745.51 \n High 806.98 \n Low 716.1106 \n Volume 25,784,000")
march_prediction=[[805, 806.98, 716.1106, 25784000]]
prediction=regressor.predict(march_prediction)
print("The predicted Close price on the 03.03.2020 2020 is:\n", prediction)