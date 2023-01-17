# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')
state=pd.get_dummies(df['Province_State'],drop_first=True)
country=pd.get_dummies(df['Country_Region'],drop_first=True)

df_clean=pd.concat([df,state,country],axis=1)

df_clean.head()
df_clean.drop(['Id','Province_State','Country_Region'],axis=1,inplace=True)

df_clean['Date']=df_clean['Date'].apply(lambda x: x.replace("-",""))

df_clean.head()
df_clean['Date']=df_clean['Date'].astype(int)
X=df_clean.drop(['ConfirmedCases','Fatalities'],axis=1)

y_1=df_clean.iloc[:,1:2]

y_2=df_clean.iloc[:,2:3]
from sklearn.tree import DecisionTreeRegressor

tree_reg=DecisionTreeRegressor(random_state=42)

tree_reg.fit(X,y_1)
from sklearn.tree import DecisionTreeRegressor

tree_reg2=DecisionTreeRegressor(random_state=42)

tree_reg2.fit(X,y_2)
test=pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')

test.head()
test_state=pd.get_dummies(test['Province_State'],drop_first=True)

test_ctry=pd.get_dummies(test['Country_Region'],drop_first=True)

test=pd.concat([test,test_state,test_ctry],axis=1)

test.head()
test.drop(['Province_State','ForecastId','Country_Region'],axis=1,inplace=True)

test['Date']=test['Date'].apply(lambda x: x.replace("-",""))

test['Date']=test['Date'].astype(int)

test.head()
y_rand1=tree_reg.predict(test)

y_rand1=pd.DataFrame(y_rand1)

y_rand1.columns=['ConfirmedCases']

y_rand1['ConfirmedCases']=y_rand1['ConfirmedCases'].astype(int)

y_rand1.head()
y_rand2=tree_reg2.predict(test)

y_rand2=pd.DataFrame(y_rand2)

y_rand2.columns=['Fatalities']

y_rand2['Fatalities']=y_rand2['Fatalities'].astype(int)

y_rand2.head()
sub=pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")
sub.drop(['ConfirmedCases','Fatalities'],axis=1, inplace=True)

final=pd.concat([sub,y_rand1,y_rand2],axis=1)

final.head()
final.to_csv('submission.csv',index=False)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)  # Don't cheat - fit only on training data

X = scaler.transform(X)

 
scaler.fit(test)  # Don't cheat - fit only on training data

test = scaler.transform(test)
from sklearn.tree import DecisionTreeRegressor

tree_reg=DecisionTreeRegressor(random_state=42)

tree_reg.fit(X,y_1)
from sklearn.tree import DecisionTreeRegressor

tree_reg2=DecisionTreeRegressor(random_state=42)

tree_reg2.fit(X,y_2)
y_rand1=tree_reg.predict(test)

y_rand1=pd.DataFrame(y_rand1)

y_rand1.columns=['ConfirmedCases']

y_rand1['ConfirmedCases']=y_rand1['ConfirmedCases'].astype(int)

y_rand1.head()
y_rand2=tree_reg2.predict(test)

y_rand2=pd.DataFrame(y_rand2)

y_rand2.columns=['Fatalities']

y_rand2['Fatalities']=y_rand2['Fatalities'].astype(int)

y_rand2.head()


sub=pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")

sub.drop(['ConfirmedCases','Fatalities'],axis=1, inplace=True)

final=pd.concat([sub,y_rand1,y_rand2],axis=1)

final.head()
final.to_csv('submission.csv',index=False)