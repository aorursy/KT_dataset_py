import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
df=pd.read_csv('../input/insurance/insurance.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()
sns.countplot(df['sex'])
sns.countplot(df['region'])
sns.countplot(df['children'])
sns.countplot(df['smoker'],hue=df['children'])
sns.heatmap(df.corr(),annot=True)
sns.jointplot(df['charges'],df['bmi'],kind='kde')
sns.jointplot(df['charges'],df['age'],kind='kde')
sns.distplot(df['bmi'])
sns.distplot(df['charges'])
sns.jointplot(df['charges'],df['children'],kind='kde',color='r')
fig,ax=plt.subplots(figsize=(20,17))
sns.barplot(y=df['charges'],x=df['sex'],hue=df['children'],ax=ax)
sns.pairplot(df)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['region']=le.fit_transform(df['region'])
df['region']
df['sex']=pd.get_dummies(df['sex'],drop_first=True)
df['smoker']=pd.get_dummies(df['smoker'],drop_first=True)
df.head()

from sklearn.model_selection import train_test_split
x=df.drop('charges',axis=1,inplace=False)
y=df['charges']
x
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
lr=LinearRegression()
lr.fit(x_train,y_train)
first_predictions=lr.predict(x_test)

print('RMSE FOR FIRST PREDICTION:',np.sqrt(mean_squared_error(y_test,first_predictions)))
print('R SCORE FOR FIRST PREDICTION:',r2_score(y_test,first_predictions))
import statsmodels.regression.linear_model as lm
x= np.append(arr = np.ones((1338,1)).astype(int), values = x, axis = 1)
x
x_optimal=x[:,[0,1,2,3,4,5,6]]
regressor_ols=lm.OLS(endog=y,exog=x_optimal).fit()
regressor_ols.summary()
x_optimal=x[:,[0,1,3,4,5]]
regressor_ols=lm.OLS(endog=y,exog=x_optimal).fit()
regressor_ols.summary()
x_train,x_test,y_train,y_test=train_test_split(x_optimal,y,test_size=0.3)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
lr=LinearRegression()
lr.fit(x_train,y_train)
second_predictions=lr.predict(x_test)
print('RMSE FOR FIRST PREDICTION:',np.sqrt(mean_squared_error(y_test,second_predictions)))
print('R SCORE FOR FIRST PREDICTION:',r2_score(y_test,second_predictions))
