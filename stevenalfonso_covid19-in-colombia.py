import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df1 = pd.read_csv('/kaggle/input/uncover/UNCOVER/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-over-time.csv')
df1 = df1.fillna(0)
colombia_df = df1[df1['country_region']=='Colombia']
df1 = df1[df1['country_region']!='Colombia']
df1 = df1[df1['country_region']!='US']
countries_df1 = df1.country_region.unique()
print('countries:',len(countries_df1))
df1.head()
plt.figure(figsize=(15,15))
#plt.suptitle('Since {} to {}'.format(min(df1['last_update']), max(df1['last_update'])))
plt.subplot(3,1,1)
for i in countries_df1:
    df = df1[df1['country_region']==i]
    plt.plot(np.arange(0,len(df1[df1['country_region']==i]),1),df['confirmed'])#,label=i)
plt.title('{} Countries \n Confirmed'.format(len(countries_df1)))
plt.xlabel('Date')
plt.ylabel('Cases')

plt.subplot(3,1,2)
for i in countries_df1:
    df = df1[df1['country_region']==i]
    plt.plot(np.arange(0,len(df1[df1['country_region']==i]),1),df['deaths'])#,label=i)
plt.title('Deaths')
plt.xlabel('Date')
plt.ylabel('Cases')

plt.subplot(3,1,3)
for i in countries_df1:
    df = df1[df1['country_region']==i]
    plt.plot(df['confirmed'],df['deaths'])#,label=i)
plt.title('Confirmed vs Deaths')
plt.xlabel('Confirmed')
plt.ylabel('Deaths')
plt.show()
x_col = colombia_df[['confirmed']].values
y_col = colombia_df[['deaths']].values
date = np.arange(0,98,1)
names = []
score_ = []
score_up = []
for i in countries_df1:
    df = df1[df1['country_region']==i]
    X = df[['confirmed']].values
    y = df[['deaths']].values
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(x_col)
    if reg.score(x_col, y_col) > 0:
        score_.append(reg.score(x_col, y_col))
        if (reg.score(x_col, y_col) > 0.9 and reg.score(x_col, y_col) < 1.0):
            score_up.append(reg.score(x_col, y_col))
            names.append(i)

x_train, x_test, y_train, y_test = train_test_split(x_col, y_col, test_size=0.5)
reg = LinearRegression()
reg.fit(x_train, y_train)

plt.figure(figsize=(15,5))
plt.plot(np.arange(0,len(score_),1), score_,label='Score')
plt.axhline(y=reg.score(x_test, y_test), color='r', linestyle='-',label='Colombia Score')
plt.axhline(y=np.mean(score_),color='black',linestyle='-',label='Mean Score')
plt.title('{} Countries Mean= {:.5f} Colombia= {:.5f}'.format(len(score_),np.mean(score_),reg.score(x_test,y_test)))
plt.xlabel('Countries')
plt.ylabel('Score')
plt.legend()
plt.show()
plt.figure(figsize=(15,15))
#plt.suptitle('Since {} to {}'.format(min(df1['last_update']), max(df1['last_update'])))
plt.subplot(3,1,1)
for i in names:
    df = df1[df1['country_region']==i]
    plt.plot(np.arange(0,len(df1[df1['country_region']==i]),1),df['confirmed'])#,label=i)
plt.title('{} Countries \n Confirmed'.format(len(names)))
plt.xlabel('Date')
plt.ylabel('Cases')

plt.subplot(3,1,2)
for i in names:
    df = df1[df1['country_region']==i]
    plt.plot(np.arange(0,len(df1[df1['country_region']==i]),1),df['deaths'])#,label=i)
plt.title('Deaths')
plt.xlabel('Date')
plt.ylabel('Cases')

plt.subplot(3,1,3)
for i in names:
    df = df1[df1['country_region']==i]
    plt.plot(df['confirmed'],df['deaths'])#,label=i)
plt.title('Confirmed vs Deaths')
plt.xlabel('Confirmed')
plt.ylabel('Deaths')
plt.show()
x_train, x_test, y_train, y_test = train_test_split(x_col, y_col, test_size=0.5)
reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
n_confirmed = np.arange(0,10000,100).reshape(-1, 1)
yn_pred = reg.predict(n_confirmed)

plt.figure(figsize=(15,5))
plt.scatter(x_test, y_test,marker='.',label='test')
plt.scatter(x_test, y_pred,marker='.',label='test pred')
plt.scatter(n_confirmed,yn_pred,marker='.',label='pred')
plt.xlabel('Confirmed')
plt.ylabel('Deaths')
plt.legend()
plt.show()
frames = [df1, colombia_df]
df1 = pd.concat(frames)
n_confirmed = np.arange(0,100000,100).reshape(-1, 1)#.reshape(100,1)
plt.figure(figsize=(15,10))
names.append('Colombia')
for i in names:
    df = df1[df1['country_region']==i]
    x = df['confirmed'].values.reshape(-1, 1)
    y = df['deaths'].values.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    yn_pred = reg.predict(n_confirmed)
    #plt.plot(n_confirmed,yn_pred,label=i)
    if i == 'Colombia':
        plt.plot(n_confirmed, yn_pred,label=i,linewidth=5,c='black')
    else:
        plt.plot(n_confirmed, yn_pred,label=i)
plt.title('Confirmed vs Deaths predicted')
plt.xlabel('Confirmed')
plt.ylabel('Deaths')
plt.legend()
plt.show()