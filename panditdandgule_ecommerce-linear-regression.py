import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv('../input/ecommerce-customers/Ecommerce Customers.csv')
df.head()
df.describe()
df.info()
df["Time on Website"].corr(df['Yearly Amount Spent'])
sns.set_palette("GnBu_d")

sns.set_style('whitegrid')

sns.jointplot(df["Time on Website"],df['Yearly Amount Spent'])
sns.jointplot(df["Time on App"],df['Yearly Amount Spent'])
sns.jointplot(df['Time on App'],df['Length of Membership'],kind='hex')
sns.pairplot(df)
df["Length of Membership"].corr(df['Yearly Amount Spent'])
sns.lmplot("Length of Membership",'Yearly Amount Spent',data=df,fit_reg=True)
X=df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]

y=df['Yearly Amount Spent']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_test,y_test)
lm.coef_
pred=lm.predict(x_test)

pred
plt.scatter(y_test,pred)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
from sklearn import metrics

print('R2:',metrics.r2_score(y_test,pred))

print('MAE:',metrics.mean_absolute_error(y_test,pred))

print('MSE:',metrics.mean_squared_error(y_test,pred))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,pred)))
sns.distplot((y_test-pred),bins=50)
cdf=pd.DataFrame(lm.coef_,X.columns,columns=['Yearly Amount Spent'])

cdf
#### ** Do you think the company should focus more on their mobile app or on their website? **