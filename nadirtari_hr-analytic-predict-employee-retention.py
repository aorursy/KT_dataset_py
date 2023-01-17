import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('../input/HR_comma_sep.csv')
df.head()
df.info()
df.groupby('left').count()['satisfaction_level']
df.groupby('left').mean()
fig,axis=plt.subplots(nrows=2,ncols=1,figsize=(12,10))
sns.countplot(x='salary',hue='left',data=df,ax=axis[0])
sns.countplot(x='Department',hue='left',data=df,ax=axis[1])
df.columns
subdf=df[['satisfaction_level', 'average_montly_hours',  'Work_accident',
          'promotion_last_5years', 'salary']]
subdf.head()
dummies=pd.get_dummies(subdf['salary']) 
dummies.head()
dffinal=pd.concat([subdf,dummies],axis='columns')
dffinal.head(3)
X=dffinal.drop('salary',axis='columns')
y=df[['left']]
X.head(3)
from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)
model.coef_
model.predict(x_test)
model.predict_proba(x_test)
model.score(x_test,y_test)