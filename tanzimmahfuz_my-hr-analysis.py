import pandas as pd

import matplotlib.pyplot as plt
data=pd.read_csv('../input/hr-analytics/HR_comma_sep.csv')

data.head()
retention=data[data['left']==1]

retention.shape
retention.head()
holding=data[data['left']==0]

holding.shape
holding.head()
data.groupby('left').mean()

pd.crosstab(data.satisfaction_level,data.Department).plot(kind='bar',figsize=(22, 16))
max_satisfaction=data['satisfaction_level'].max()

max_satisfaction
a=data.loc[data['satisfaction_level']==1.0,'Department'].unique()

a
data.Department.unique()
data.Department.nunique()
min_satisfaction=data['satisfaction_level'].min()
min_satisfaction
a=data.loc[data['satisfaction_level']==0.09,'Department'].unique()

a
a=data.loc[data['satisfaction_level']==0.09,'Department'].nunique()

a
pd.crosstab(data.salary,data['satisfaction_level']==1.0).plot(kind='bar')
pd.crosstab(data.salary,data.left).plot(kind='bar')
pd.crosstab(data.left,data.salary).plot(kind='bar')
pd.crosstab(data.left,data.Department).plot(kind='bar', figsize=(22,16))
pd.crosstab(data.Department,data.left).plot(kind='bar', figsize=(22,16))
data.Department.value_counts()
new_data=data[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]

new_data.head()
#Now here i will apply the LOGISTICS REGRESSION analysis where the employees will leave or hold the job
dummy=pd.get_dummies(data.salary)

dummy.head()
new_data_dummy=pd.concat([new_data,dummy],axis='columns')

new_data_dummy.head()
new_data_dummy.drop('salary',axis='columns',inplace=True)

new_data_dummy.head()
new_data_dummy.head()
from sklearn.model_selection import train_test_split

x=new_data_dummy

y=data.left
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)
from sklearn.linear_model import LogisticRegression 

model=LogisticRegression()

model.fit(x_train,y_train)
model.predict(x_test)
model.score(x_test,y_test)