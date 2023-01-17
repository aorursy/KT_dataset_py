import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))

%matplotlib inline

# Any results you write to the current directory are saved as output.
customer_df=pd.read_csv('../input/Ecommerce Customers.csv')

customer_df.head()
customer_df.describe()
customer_df.info()
sns.set_palette('GnBu_r')
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customer_df)

plt.show()
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customer_df)

plt.show()
sns.jointplot(x='Time on App',y='Length of Membership',data=customer_df,kind='hex')

plt.show()
sns.jointplot(x='Time on Website',y='Length of Membership',data=customer_df,kind='hex')

plt.show()
sns.pairplot(customer_df)

plt.show()
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customer_df)

plt.show()
y=customer_df['Yearly Amount Spent']

X=customer_df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
#Train-test split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42)

# I have set random_state=42 in order to get the same output every time i run this kernel
#creating and training the model

from sklearn.linear_model import LinearRegression

lm=LinearRegression()

lm.fit(X_train,y_train)

#coefficients of the model

print('Coefficients: \n',lm.coef_)
predictions =lm.predict(X_test)
plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()
from sklearn import metrics

print('MAE= ', metrics.mean_absolute_error(y_test,predictions))

print('MSE= ', metrics.mean_squared_error(y_test,predictions))

print('RMSE= ', np.sqrt(metrics.mean_squared_error(y_test,predictions)))
sns.distplot((y_test-predictions),bins=40);
cdf=pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'] )

cdf