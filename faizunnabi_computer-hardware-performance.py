import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data=pd.read_csv('../input/machine.data.txt',names=['Vendor','Model Name','MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP','ERP'])
data.head()
data.info()
data.describe()
sns.set_style('whitegrid')
sns.distplot(data['MYCT'],bins=30,kde=False)
sns.heatmap(data.isnull(),cbar=False,yticklabels=False,cmap='viridis')
df=data.drop('ERP',axis=1)
sns.jointplot(x='MYCT',y='PRP',data=df)
sns.jointplot(x='MMIN',y='PRP',data=df)
sns.pairplot(df)
df.columns
from sklearn.model_selection import train_test_split
X=df[['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN','CHMAX']]
y=df['PRP']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
model.coef_
df_temp=pd.DataFrame({'PRP':list(y_test),'ERP':predictions})
sns.lmplot(x='PRP',y='ERP',data=df_temp)
sns.distplot((y_test-predictions),bins=50)
#df_temp2=pd.DataFrame({'PRP':list(y_test),'ERP':predictions})
predict_all=model.predict(X)
df_temp2=pd.DataFrame({'ERP':list(data['ERP']),'Predicts':predict_all,'PRP':list(data['PRP'])})
sns.lmplot(x='ERP',y='Predicts',data=df_temp2)
