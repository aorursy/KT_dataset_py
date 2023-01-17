import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import linear_model
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
data = data.drop(['Province/State','Lat','Long'],axis=1)

data = data.groupby('Country/Region').sum()

data=data.sort_values('3/14/20',ascending=False)

data=data.iloc[:20,:]
def alpha(x):

    x= list(x)

    lr = linear_model.LinearRegression()

    def log_(x):

        if x==0: return 0

        else : return np.log10(x)

    lr.fit(np.arange(len(x)).reshape(-1,1),list(map(log_,x)))

    return lr.coef_[0]
lr = linear_model.LinearRegression()

lr.fit(np.arange(len(data.iloc[0,:])).reshape(-1,1),list(map(np.log10,data.iloc[:20,:].drop('China').sum())))

f, ax = plt.subplots(2,2, figsize=(12,10))

ax[0,0].plot(list(data.sum()))

ax[0,0].set_xticklabels(list(data.columns[[0,0,10,20,30,40,50]]))

ax[0,1].plot(list(map(np.log10,data.iloc[:20,:].sum())))

ax[0,1].set_xticklabels(list(data.columns[[0,0,10,20,30,40,50]]))

ax[0,1].set_yticklabels(['0','10','10^2','10^3','10^4','10^5'])

ax[1,0].plot(list(data.drop('China').sum()))

ax[1,0].set_xticklabels(list(data.columns[[0,0,10,20,30,40,50]]))

ax[1,1].plot(list(map(np.log10,data.iloc[:20,:].drop('China').sum())))

ax[1,1].plot(lr.predict(np.arange(len(data.iloc[0,:])).reshape(-1,1)), alpha = 0.5)

ax[1,1].set_xticklabels(list(data.columns[[0,0,10,20,30,40,50]]))

ax[1,1].set_yticklabels(['0','10','10^2','10^3','10^4','10^5'])

ax[0,0].set_title('World Confirmed')

ax[0,1].set_title('World Confirmed - logscale')

ax[1,0].set_title('World Confirmed (except china)')

ax[1,1].set_title('World Confirmed (except china) - logscale')

ax[1,1].text(23,2,'alpha={}'.format(alpha(list(data.drop('China').sum()))))
alpha(data.drop('China').sum())
plt.figure()

for i in range(0,5):

    plt.plot(list(data.iloc[i,:]), label=data.index[i])

plt.legend()

plt.title('top5_confirmed')

plt.xlabel('date')

plt.ylabel('confirmed')

plt.xticks([0,10,20,30,40,50], list(data.columns[[0,10,20,30,40,50]]))
f, ax = plt.subplots(2,2, figsize=(15,10))

ax[0,0].plot(list(data.loc['China']), label=data.index[0])

ax[0,0].legend()

ax[0,0].set_xticklabels(list(data.columns[[0,0,10,20,30,40,50]]))

ax[0,0].set_title('China Confirmed')

for i in range(1,8):

    ax[0,1].plot(list(data.iloc[i,:]), label=data.index[i])

ax[0,1].legend()

ax[0,1].set_xticklabels(list(data.columns[[0,0,10,20,30,40,50]]))

ax[0,1].set_title('Top2-8 Confirmed')

for i in range(8,14):

    ax[1,0].plot(list(data.iloc[i,:]), label=data.index[i])

ax[1,0].legend()

ax[1,0].set_xticklabels(list(data.columns[[0,0,10,20,30,40,50]]))

ax[1,0].set_title('Top9-14 Confirmed')

for i in range(14,20):

    ax[1,1].plot(list(data.iloc[i,:]), label=data.index[i])

ax[1,1].set_xticklabels(list(data.columns[[0,0,10,20,30,40,50]]))

ax[1,1].legend()

ax[1,1].set_title('Top14-20 Confirmed')
f, ax = plt.subplots(2,2, figsize=(15,10))

ax[0,0].plot(list(map(lambda x: np.log10(x),data.loc['China'])), label=data.index[0])

ax[0,0].legend()

ax[0,0].set_xlim(-1,55)

ax[0,0].set_ylim(-0.5,5)

for i in range(1,8):

    ax[0,1].plot(list(map(lambda x: np.log10(x),data.iloc[i,:])), label=data.index[i])

ax[0,1].legend()

ax[0,1].set_xlim(-1,55)

ax[0,1].set_ylim(-0.5,5)

ax[0,1].axvline(x=29,linewidth=0.5, color='red')

for i in range(8,14):

    ax[1,0].plot(list(map(lambda x: np.log10(x),data.iloc[i,:])), label=data.index[i])

ax[1,0].legend()

ax[1,0].set_xlim(-1,55)

ax[1,0].set_ylim(-0.5,5)

ax[1,0].axvline(x=35,linewidth=0.5, color='red')

for i in range(14,20):

    ax[1,1].plot(list(map(lambda x: np.log10(x),data.iloc[i,:])), label=data.index[i])

ax[1,1].legend()

ax[1,1].set_xlim(-1,55)

ax[1,1].set_ylim(-0.5,5)

ax[1,1].axvline(x=38,linewidth=0.5, color='red')

for i in [0,1]:

    for j in [0,1]:

        ax[i,j].set_xticklabels(list(data.columns[[0,0,10,20,30,40,50]]))

        ax[i,j].set_yticklabels(['0','0','10','10^2','10^3','10^4','10^5'])

ax[0,0].set_title('China Confirmed - logscale')

ax[0,1].set_title('Top2-8 Confirmed - logscale')

ax[1,0].set_title('Top9-14 Confirmed - logscale')

ax[1,1].set_title('Top14-20 Confirmed - logscale')

ax[0,1].text(30,4,'pendemic start')

ax[0,1].text(31,3.7,'Jan.20')

ax[1,0].text(36,4,'pendemic start')

ax[1,0].text(37,3.7,'Jan.26')

ax[1,1].text(39,4,'pendemic start')

ax[1,1].text(40,3.7,'Jan.28')
top2_8 = data.iloc[1:8,29:]

top9_14 = data.iloc[8:14,35:].fillna(0)

top15_20 = data.iloc[14:20,38:].fillna(0)

top2_8['alpha'] = [alpha(top2_8.iloc[i,:]) for i in range(len(top2_8))]

top2_8['class'] = np.zeros(len(top2_8))

top9_14['alpha'] = [alpha(top9_14.iloc[i,:]) for i in range(len(top9_14))]

top9_14['class'] =np.zeros(len(top9_14))+1

top15_20['alpha'] = [alpha(top15_20.iloc[i,:]) for i in range(len(top15_20))]

top15_20['class'] =np.zeros(len(top9_14))+2

df = pd.concat([top2_8,top9_14,top15_20])

df['order']= np.arange(2,21)

df = df.sort_values('alpha')

def color_order(x):

    return 1-x/23
fig =plt.figure(figsize=(15,5))

for i in range(len(df)):

    plt.bar(x=df.index[i] , height=df.alpha[i], alpha= color_order(df['order'][i]), color='red')

plt.axhline(y=alpha(data.drop('China').sum()), linewidth=0.8, color='green')

plt.xticks(rotation=40)

plt.title('Top 20 alpha value after pendemic')

plt.text(0.5, 0.19, 'World alpha')