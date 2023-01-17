import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/ecommerce-customers/Ecommerce Customers.csv')
df.head()
df.info()
df.describe()
# Pair plot for whole data frame
sns.pairplot(df)
sns.jointplot(data= df, x = 'Time on Website',y='Yearly Amount Spent')
sns.jointplot(data=df,x='Time on App',y = "Yearly Amount Spent")
sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=df)
sns.pairplot(df)
# check the most related column with yearly amount spent
sns.lmplot(data=df,x='Length of Membership', y='Yearly Amount Spent')
x = df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)
import sklearn.linear_model
from sklearn.linear_model import LinearRegression 
lm = LinearRegression()
lm.fit(x_train,y_train)
lm.coef_
pred = lm.predict(x_test)
sns.scatterplot(x=pred,y=y_test)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
from sklearn import metrics
print('MAE',metrics.mean_absolute_error(y_test,pred))
print('MSE',metrics.mean_squared_error(y_test,pred))
print('RMSE', np.sqrt(metrics.mean_squared_error(y_test,pred)))
# R SQU
metrics.explained_variance_score(y_test,pred)
sns.distplot((y_test-pred),bins=50)
cdf = pd.DataFrame(lm.coef_,x.columns,columns=['Coef'])
cdf