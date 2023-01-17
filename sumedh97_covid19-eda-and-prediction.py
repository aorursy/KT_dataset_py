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
import pandas as pd

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import plotly.express as px

from datetime import datetime

import plotly.graph_objects as go

%matplotlib inline



train_df=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")

test_df=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")

train_df.head()





train_df.drop('County',axis=1,inplace=True)

train_df.drop('Province_State',axis=1,inplace=True)

train_df.set_index('Id',inplace=True)

train_df.info()
#GETTING TOP 15 INFECTED COUNTRIES(CONFIRMED CASES)



df1=train_df[train_df['Target']=='ConfirmedCases']



train_max_confirmed=pd.DataFrame()

train_max_confirmed['Confirmed_cases']  = df1.groupby('Country_Region')['TargetValue'].max().sort_values(ascending=False)

plot_confirmed= train_max_confirmed.head(15)





plt.style.use("fivethirtyeight")

fig,ax= plt.subplots(figsize=(10,7))

ax.bar(plot_confirmed.index, plot_confirmed['Confirmed_cases'],color='r',label='Confirmed cases',width=0.8,alpha=0.7)

ax.set_xticklabels(train_max_confirmed.index,rotation=80,color='black')

ax.set_ylabel('Confirmed cases')

ax.set_title('Top 15 Infected countries chart')

plt.show()
df2=train_df[train_df['Target']!='ConfirmedCases']

train_max_deaths=pd.DataFrame()

train_max_deaths['Fatalities']  = df2.groupby('Country_Region')['TargetValue'].max().sort_values(ascending=False)



plot_confirmed1= train_max_deaths.head(15)

plt.style.use("fivethirtyeight")

fig,ax= plt.subplots(figsize=(10,7))

ax.bar(plot_confirmed1.index, plot_confirmed1['Fatalities'],color='m',label='Deaths',width=0.8,alpha=0.7)

ax.set_xticklabels(plot_confirmed1.index,rotation=80,color='black')

ax.set_ylabel('Number of deaths')

ax.set_title('Top 15 countries with maximum fatalities')

plt.show()
fig = px.pie(df1, values='TargetValue', names='Country_Region')

fig.update_traces(textposition='inside')

fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

fig.show()
fig = px.pie(df2, values='TargetValue', names='Country_Region')

fig.update_traces(textposition='inside')

fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

fig.show()
plot_confirmed1.sort_values(by='Fatalities',inplace=True)

fig = px.bar(plot_confirmed1,

             x=plot_confirmed1['Fatalities'], y=plot_confirmed1.index,

             title='Mortality rate HIGH: top 15 countries' , text='Fatalities', height=750, orientation='h')

fig.show()
df1.set_index('Date',inplace=True)

df2.set_index('Date',inplace=True)

df3=pd.concat([df1,df2],axis=1,ignore_index=True)

df3.drop([3,5,6,7,8],inplace=True,axis=1)

df3.rename(columns={0:'Country_Region',1:'Population',2:'Weight',4:'Confirmed',9:'Fatalities'},inplace=True)

df3.columns
#ANALYSING TRENDS IN INDIA

df_india = df3[df3['Country_Region'] == "India"].reset_index()

final_india = df_india.groupby('Date')['Date','Population','Weight','Confirmed','Fatalities'].sum().reset_index()

from plotly.subplots import make_subplots

figure = make_subplots(rows = 1, cols = 2, subplot_titles = ("Confirmed","Fatalities"))



a1 = go.Scatter(x=final_india['Date'],y=final_india['Confirmed'], name = "Confirmed", line_color = 'red', mode = 'lines+markers')

a2 = go.Scatter(x=final_india['Date'],y=final_india['Fatalities'], name = "Deaths", line_color = 'blue', mode = 'lines+markers')



figure.append_trace(a1, 1, 1)

figure.append_trace(a2, 1, 2)





figure.update_layout(template="plotly",title_text = ' Spread of Corona Virus over time in India')

figure.show()
#ANALYSING TRENDS IN US

df_US = df3[df3['Country_Region'] == "US"].reset_index()

final_US = df_US.groupby('Date')['Date','Population','Weight','Confirmed','Fatalities'].sum().reset_index()

figure = make_subplots(rows = 1, cols = 2, subplot_titles = ("Confirmed","Fatalities"))



a1 = go.Scatter(x=final_US['Date'],y=final_US['Confirmed'], name = "Confirmed", line_color = 'firebrick', mode = 'lines+markers')

a2 = go.Scatter(x=final_US['Date'],y=final_US['Fatalities'], name = "Deaths", line_color = 'green', mode = 'lines+markers')



figure.append_trace(a1, 1, 1)

figure.append_trace(a2, 1, 2)





figure.update_layout(template="plotly",title_text = ' Spread of Corona Virus over time in US')

figure.show()
#ANALYSING TRENDS IN CHINA

df_China = df3[df3['Country_Region'] == "China"].reset_index()

final_China = df_China.groupby('Date')['Date','Population','Weight','Confirmed','Fatalities'].sum().reset_index()

figure = make_subplots(rows = 1, cols = 2, subplot_titles = ("Confirmed","Fatalities"))



a1 = go.Scatter(x=final_China['Date'],y=final_China['Confirmed'], name = "Confirmed", line_color = 'royalblue', mode = 'lines+markers')

a2 = go.Scatter(x=final_China['Date'],y=final_China['Fatalities'], name = "Deaths", line_color = 'orange', mode = 'lines+markers')



figure.append_trace(a1, 1, 1)

figure.append_trace(a2, 1, 2)





figure.update_layout(template="plotly",title_text = ' Spread of Corona Virus over time in China')

figure.show()
#ANALYSING TRENDS IN FRANCE

df_France = df3[df3['Country_Region'] == "France"].reset_index()

final_France = df_France.groupby('Date')['Date','Population','Weight','Confirmed','Fatalities'].sum().reset_index()

figure = make_subplots(rows = 1, cols = 2, subplot_titles = ("Confirmed","Fatalities"))



a1 = go.Scatter(x=final_France['Date'],y=final_France['Confirmed'], name = "Confirmed", line_color = '#e377c2', mode = 'lines+markers')

a2 = go.Scatter(x=final_France['Date'],y=final_France['Fatalities'], name = "Deaths", line_color = '#8c564b', mode = 'lines+markers')



figure.append_trace(a1, 1, 1)

figure.append_trace(a2, 1, 2)





figure.update_layout(template="plotly",title_text = ' Spread of Corona Virus over time in France')

figure.show()
#ANALYSING TRENDS IN ITALY

df_Italy = df3[df3['Country_Region'] == "Italy"].reset_index()

final_Italy = df_Italy.groupby('Date')['Date','Population','Weight','Confirmed','Fatalities'].sum().reset_index()

figure = make_subplots(rows = 1, cols = 2, subplot_titles = ("Confirmed","Fatalities"))



a1 = go.Scatter(x=final_Italy['Date'],y=final_Italy['Confirmed'], name = "Confirmed", line_color = '#17becf', mode = 'lines+markers')

a2 = go.Scatter(x=final_Italy['Date'],y=final_Italy['Fatalities'], name = "Deaths", line_color = '#2ca02c', mode = 'lines+markers')



figure.append_trace(a1, 1, 1)

figure.append_trace(a2, 1, 2)





figure.update_layout(template="plotly",title_text = ' Spread of Corona Virus over time in Italy')

figure.show()
import seaborn as sns

df3_grouped=df3.groupby(['Country_Region']).sum()

df3_grouped.Confirmed

tot_conf= df3_grouped.nlargest(10,'Confirmed')

tot_deaths=df3_grouped.nlargest(10,'Fatalities')



fig,ax = plt.subplots(figsize=(15,8))

sns.barplot(y="Population", x="Confirmed", data=tot_conf,ax=ax)

plt.title(' top 10 population VS Confirmed Cases',size=25)

plt.show()
fig,ax = plt.subplots(figsize=(15,8))

sns.barplot(y="Population", x="Fatalities", data=tot_conf,ax=ax)

plt.title('top 10  population VS Fatalities',size=25)

plt.show()
fig = px.treemap(df3, path=['Country_Region'], values='Confirmed',

                  color='Population', hover_data=['Country_Region'],

                  color_continuous_scale='Inferno')

fig.show()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

#LabelEncoder can be used to normalize labels.

#fit_trainsform - Fit label encoder and return encoded labels

train_df['Target_numerical'] = le.fit_transform(train_df['Target'])

train_df['Country_numerical']= le.fit_transform(train_df['Country_Region'])





train_df['Date'] = pd.to_datetime(train_df['Date'])





train_df['Dayofweek'] = train_df['Date'].dt.dayofweek

train_df['Day'] = train_df['Date'].dt.day

train_df['Month'] = train_df['Date'].dt.month

train_df.head()




test_df.drop(['Province_State','County'],axis=1,inplace=True)

test_df['Target_numerical'] = le.fit_transform(test_df['Target'])

test_df['Country_numerical']= le.fit_transform(test_df['Country_Region'])





test_df['Date'] = pd.to_datetime(test_df['Date'])





test_df['Dayofweek'] = test_df['Date'].dt.dayofweek

test_df['Day'] = test_df['Date'].dt.day

test_df['Month'] = test_df['Date'].dt.month

test_df.head()
test_df.set_index('ForecastId',inplace=True)
plt.title("Heatmap Correlation of the variables in COVID19 Dataset", fontsize = 15)

sns.heatmap(train_df.corr(), annot=True, fmt=".2f",cmap='YlGnBu',linewidths=0.40)

plt.show()
#TRAIN TEST SPLIT 

from sklearn.model_selection import train_test_split



#Selecting feature columns

col_feat = ['Population', 'Weight','Target_numerical', 'Country_numerical', 'Dayofweek','Day', 'Month']

X = train_df[col_feat] # Features

y = train_df['TargetValue'] # Target variable

 



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
#CHECKING THE SHAPE

print('Training Features Shape:', X_train.shape)

print('Training Labels Shape:',   y_train.shape)

print('Testing Features Shape:',  X_test.shape)

print('Testing Labels Shape:',    y_test.shape)
# Import the model we are using

from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 50 decision trees

model = RandomForestRegressor(n_estimators = 50, random_state = 42)

# Train the model on training data

model.fit(X_train, y_train)
y_predicted= model.predict(X_test)

model.score(X_test,y_test)
fig, ax = plt.subplots()



ax.scatter(y_test, y_predicted, edgecolors=(0, 0, 0))



ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)



ax.set_xlabel('Actual')



ax.set_ylabel('Predicted')



ax.set_title("Visualising goodness of fit")



plt.show()
#WORKING WITH TEST DATASET

test_col = ['Population', 'Weight','Target_numerical', 'Country_numerical', 'Dayofweek','Day', 'Month']

test_data = test_df[test_col]

test_df.head()
# predictions

y_predicted1 = model.predict(test_data)

y_predicted1
#Creating a dataframe with iD and Predicted list

output_df=pd.DataFrame({'id':test_df.index,'TargetValue':y_predicted1})

output_df


q1=output_df.groupby(['id'])['TargetValue'].quantile(q=0.05).reset_index()

q2=output_df.groupby(['id'])['TargetValue'].quantile(q=0.5).reset_index()

q3=output_df.groupby(['id'])['TargetValue'].quantile(q=0.95).reset_index()
q1.columns=['id','q0.05']

q2.columns=['id','q0.5']

q3.columns=['id','q0.95']

q=pd.concat([q1,q2['q0.5'],q3['q0.95']],1)

q['q0.05']=q['q0.05']

q['q0.5']=q['q0.5']

q['q0.95']=q['q0.95']

q
submission_df=pd.melt(q, id_vars=['id'], value_vars=['q0.05','q0.5','q0.95'])

submission_df['variable']=submission_df['variable'].str.replace("q","", regex=False)

submission_df['ForecastId_Quantile']=submission_df['id'].astype(str)+'_'+submission_df['variable']

submission_df['TargetValue']=submission_df['value']

sub_df=submission_df[['ForecastId_Quantile','TargetValue']]

sub_df.reset_index(drop=True,inplace=True)

sub_df.to_csv("submission.csv",index=False)

sub_df.head()