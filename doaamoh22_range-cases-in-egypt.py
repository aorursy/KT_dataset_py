# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



%matplotlib inline

 

plt.style.use('fivethirtyeight')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

df.head()
#Egypt cases

df[df['Country/Region']=='Egypt']
#covert date cols to rows



df2 = pd.melt(df, id_vars=["Province/State", "Country/Region",'Lat','Long'], var_name="Date", value_name="Confirmed")

df2.head()
#get Egypt data only 

Eg=df2[df2['Country/Region']=='Egypt']

Eg.reset_index(inplace=True)
cases=[]

cases.append((Eg['Confirmed'][0]))

for i in range(len(Eg['Confirmed'])-1):

    

    cases.append((Eg['Confirmed'][i+1]- Eg['Confirmed'][i]))

        

Eg.loc.__setitem__((slice(None), ('cases')), cases)

Eg.head()
Eg.loc.__setitem__((slice(None), ('Date')), pd.to_datetime(Eg['Date']))

Eg.loc.__setitem__((slice(None), ('Day')), Eg['Date'].apply(lambda x: x.strftime("%a")))

Eg.loc.__setitem__((slice(None), ('Month')), Eg['Date'].apply(lambda x: x.strftime("%b")))

Eg.head()
#set index for weeks

sorter_days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

sorterIndex_days = dict(zip(sorter_days,range(len(sorter_days))))

sorterIndex_days

Days=Eg.set_index('Day')

Days.head()
Days['Day_id'] = Days.index

Days['Day_id'] = Days['Day_id'].map(sorterIndex_days)

Days.head()
def graph(df,col,xlabel):

    fig, ax = plt.subplots(figsize=(8,6), facecolor='white')

    x=df[col]

    y=df['cases']



    ax.bar(x, y, width=0.3, align='center')

    ax.set_xticks([x for x in df[col]])

    ax.set_xticklabels(df.index)

    ax.set_title('Number of cases confirmed By Days in Egypt {}'.format(xlabel))

    ax.set_xlabel(xlabel)

    ax.set_ylabel('confirmed')



    ax.set_facecolor('white')

    return(True)
df_EG=Days.reset_index()

df_EG=df_EG.groupby(['Month','Day'],as_index=False).sum()

df_EG=df_EG.set_index('Day')

df_EG['Day_id'] = df_EG.index

df_EG['Day_id'] = df_EG['Day_id'].map(sorterIndex_days)

#df_EG.head()

Mar=df_EG[df_EG['Month']=='Mar']

Feb=df_EG[df_EG['Month']=='Feb']

Jan=df_EG[df_EG['Month']=='Jan']
graph(Mar,'Day_id','(March)')
Months=Eg.copy()

Months.head()
Months['day_num']=Months['Date'].apply(lambda x: x.strftime("%d"))

Months.head()
Mar_=Months[Months['Month']=='Mar']

Feb_=Months[Months['Month']=='Feb']

Apr_=Months[Months['Month']=='Apr']



x_Mar=Mar_['day_num'].values

y_Mar=Mar_['cases'].values

x_Feb=Feb_['day_num'].values

y_Feb=Feb_['cases'].values

x_Apr=Apr_['day_num'].values

y_Apr=Apr_['cases'].values



plt.figure(figsize=(10,5))



#plt.figure()

#plt.plot(x, y)

plt.plot(x_Mar, y_Mar, marker='.',linewidth=2, markersize=3)

plt.plot(x_Feb, y_Feb,'green', marker='.',linewidth=2, markersize=3)

plt.plot(x_Apr, y_Apr,'red', marker='.',linewidth=2, markersize=3)



plt.title(f"Increasing number of cases Confirmed By days in Egypt")

plt.xlabel("Day")

plt.ylabel("Confirmed")

#plt.xticks(x, [str(i) for i in y], rotation=90)



#set parameters for tick labels

plt.tick_params(axis='x', which='major')

plt.legend(['Feb','Mar', 'Apr'])

plt.tight_layout()