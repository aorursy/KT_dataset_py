import pandas as pd

import matplotlib.pyplot as ma
df=pd.read_csv('logis_code.csv')

df.head()
df.left.unique()
df['left'].value_counts()
df['Department'].value_counts()
ma.scatter(df.time_spend_company,df.left,marker="*",color='blue')
df['salary'].value_counts()
pd.crosstab(df.salary,df.left).plot(kind='bar')
du=pd.get_dummies(df.salary)
df=pd.concat([df,du],1)

df.head()
df=df.drop(['salary','last_evaluation','number_project','Work_accident','time_spend_company','high'],1)

df.head()
df.shape
df.isnull().sum()
df.groupby('left').mean()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df[['satisfaction_level','average_montly_hours','promotion_last_5years','low','medium']],df.left,train_size=0.78,random_state=121)
x_train.head()
from sklearn.linear_model import LogisticRegression
mod=LogisticRegression()

mod.fit(x_train,y_train)
mod.score(x_test,y_test)
mod.predict_proba(x_test)