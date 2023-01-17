# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
%matplotlib inline
train=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")
test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")
test.head()
test.info()
train.info()
train.shape
test.shape
train.isnull().sum()
test.isnull().sum()
ID=train["Id"]
FID=test["ForecastId"]
train=train.drop(columns=['County','Province_State','Id'])
test=test.drop(columns=['County','Province_State','ForecastId'])
sns.pairplot(train)
sns.barplot(x="Target",y="TargetValue",data=train)
sns.barplot(x='Target',y='Population',data=train)
fig=plt.figure(figsize=(45,30))
fig=px.pie(train, values='TargetValue', names='Country_Region')
fig.update_traces(textposition='inside')
#fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
fig.show()
#We Group Or Add Every Rows With Same Country Region
df_grouped=train.groupby(['Country_Region']).sum()
df_grouped.TargetValue
top5=df_grouped.nlargest(5,'TargetValue')
top5
sns.catplot(y="Population", x="TargetValue",kind="bar", data=top5)
plt.title('Top 5 Target Values',size=20)
plt.show()
top5pop=df_grouped.nlargest(5,'Population')
top5pop
fig = px.treemap(train, path=['Country_Region'], values='TargetValue',
                  color='Population', hover_data=['Country_Region'],
                  color_continuous_scale='matter')
fig.show()
#If ‘coerce’, then invalid parsing will be set as NaN.
#strftime=Return an Index of formatted strings specified by date

dateee= pd.to_datetime(train['Date'], errors='coerce')
train['Date']= dateee.dt.strftime("%Y%m%d").astype(int)
dateee= pd.to_datetime(test['Date'], errors='coerce')
test['Date']= dateee.dt.strftime("%Y%m%d").astype(int)
top2000=train.nlargest(2000,'TargetValue')
top2000
fig, ax = plt.subplots(figsize=(10,10))

h=pd.pivot_table(top2000,values='TargetValue',index=['Country_Region'],columns='Date')

sns.heatmap(h,cmap="RdYlGn",linewidths=0.05)
top2000pop=train.nlargest(2000,'Population')
top2000pop
fig ,ax=plt.subplots(figsize=(20,10))
h=pd.pivot_table(top2000pop,values='TargetValue',index=['Country_Region'],columns="Date")
sns.heatmap(h,cmap="RdYlGn",linewidths=0.005)
train.info()
from sklearn.preprocessing import LabelEncoder
enco = LabelEncoder()

temp=train.iloc[:,0].values
train.iloc[:,0]=enco.fit_transform(temp)

temp=train.iloc[:,4].values
train.iloc[:,4]=enco.fit_transform(temp)
train.info()
test.info()
temp=test.iloc[:,0].values
test.iloc[:,0]=enco.fit_transform(temp)

temp=test.iloc[:,4].values
test.iloc[:,4]=enco.fit_transform(temp)
test.info()
target=train["TargetValue"]
df_train=train.drop(["TargetValue"],axis=1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df_train,target,test_size=0.3,random_state=0)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
pipe = Pipeline([("scaler2",StandardScaler()),("RandomForestRegressor:",RandomForestRegressor())])

pipe.fit(x_train,y_train)

prediction=pipe.predict(x_test)
acc=pipe.score(x_test,y_test)
acc
predict=pipe.predict(test)
output=pd.DataFrame({'id':FID,'TargetValue':predict})
output
a=output.groupby(['id'])['TargetValue'].quantile(q=0.05).reset_index()
b=output.groupby(['id'])['TargetValue'].quantile(q=0.5).reset_index()
c=output.groupby(['id'])['TargetValue'].quantile(q=0.95).reset_index()
a.columns=['Id','q0.05']
b.columns=['Id','q0.5']
c.columns=['Id','q0.95']

total=pd.concat([a,b['q0.5'],c['q0.95']],axis=1)


total
sub=pd.melt(total, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])

print(sub)
#Removing the character "q"
sub['variable']=sub['variable'].str.replace("q","", regex=False)
#Formating ForecastId with in this format "Id_variable"
sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']
sub=sub[['ForecastId_Quantile','TargetValue']]
sub.reset_index(drop=True,inplace=True)


sub.to_csv("submission.csv",index=False)
sub