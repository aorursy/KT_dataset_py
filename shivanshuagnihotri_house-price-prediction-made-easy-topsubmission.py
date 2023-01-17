import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# missing values
train_data.head()
print(pd.isnull(train_data).sum())
# missing values
train_data.head()
print(pd.isnull(train_data).sum())
train_data.describe()
sns.barplot(x='OverallQual',y='SalePrice', data=train_data) # quality proportional to SalePrice 
plt.figure(figsize=(12,6))
sns.distplot(train_data['SalePrice']) # range of prices and their distribution 
train_data.info() 
corr = train_data.corr()
print(corr['SalePrice'].sort_values(ascending=False)[:5],'\n') # highly correlated attributes
plt.scatter(x='GrLivArea',y='SalePrice',data= train_data)
plt.ylabel('Sale Price')
plt.xlabel('Living area square feet')
plt.show()
# listing all the null columns properly 
null_data=pd.DataFrame(train_data.isnull().sum().sort_values(ascending=False)[:20])
null_data.columns=["Count"]
null_data.index.name='Feature'
null_data
# 19 coulumns have null values
# getting dummy data for street 
train_data['street_new']=pd.get_dummies(train_data.Street, drop_first=True)
test_data['street_new']=pd.get_dummies(test_data.Street, drop_first=True)

train_data.drop(['Street'],axis=1,inplace=True)
test_data.drop(['Street'],axis=1,inplace=True)
# encoding slae condition coulmn
def encode(x):
    return 1 if x == 'Partial' else 0
train_data['salecondition_encoded']=train.SaleCondition.apply(encode)
test_data['salecondition_encoded']=test.SaleCondition.apply(encode)

train_data.drop(['SaleCondition'],axis=1,inplace=True)
test_data.drop(['SaleCondition'],axis=1,inplace=True)
# selecting numeric values only and dropping null columns 
new_data = train_data.select_dtypes(include=[np.number]).interpolate().dropna()
sum(new_data.isnull().sum()!=0)
new_data.head() #all numerics
y=np.log(train_data.SalePrice)
X=new_data.drop(['SalePrice','Id'],axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=.30)
# Logistic Regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


from sklearn.metrics import mean_squared_error
print('RMSE is: ', mean_squared_error(y_test,y_pred))

plt.scatter(y_pred,y_test,alpha=.7)
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression')
plt.show()
'''
for submitting to kaggle 
# first predict on given test data

tid = test_data['Id']
test_new= test_data.select_dtypes(include=[np.number]).drop(['Id'],axis=1).interpolate()
test_new.head()

predictions=model.predict(test_new)
predictions = np.exp(predictions)


#sconvert to csv file named submission.csv

submit = pd.DataFrame({ 'Id' : tid, 'SalePrice': predictions })
submit.head()
submit.to_csv('house_price_prediction.csv', index=False)

'''

