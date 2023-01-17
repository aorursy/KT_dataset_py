import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
data = pd.read_csv('../input/concrete-compressive-strength-data-set/compresive_strength_concrete.csv')
data
data.columns
df=data.copy()
df1=df.rename(columns={'Cement (component 1)(kg in a m^3 mixture)':'Cement',
       'Blast Furnace Slag (component 2)(kg in a m^3 mixture)':'BFS',
       'Fly Ash (component 3)(kg in a m^3 mixture)':'Fly_Ash',
       'Water  (component 4)(kg in a m^3 mixture)':'Water',
       'Superplasticizer (component 5)(kg in a m^3 mixture)':'Superplasticizer',
       'Coarse Aggregate  (component 6)(kg in a m^3 mixture)':'Coarser_agg',
       'Fine Aggregate (component 7)(kg in a m^3 mixture)':'Fine_agg',
       'Age (day)':'Days','Concrete compressive strength(MPa, megapascals) ':'strength'})
df1
df1.describe()
sns.distplot(df1['Days'])
df1['Days'].min()
df1['Days'].max()
df1['Days'].unique()
q=df1['Days'].quantile(0.80)
df2=df1[df1['Days']<q]
df2
sns.distplot(df2['Days'])

df2['Days'].min()
df2['Days'].max()
df2['Days'].unique()
x=df2.drop(['strength'],axis=1)
y=df2['strength']
x
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_sc=scaler.fit(x)
x_sc=scaler.transform(x)
x_sc
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_sc,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
reg.coef_
reg.intercept_
reg.score(x_train,y_train)
pred=reg.predict(x_test)

plt.scatter(y_test,pred)
plt.xlabel('Actual values')
plt.ylabel('Predicted Values')

data_output=pd.DataFrame({"Actual values":y_test,"Predicted values":pred})
data_output
coeffecients = pd.DataFrame(reg.coef_,x.columns)
coeffecients.columns = ['Coeffecient']
coeffecients

