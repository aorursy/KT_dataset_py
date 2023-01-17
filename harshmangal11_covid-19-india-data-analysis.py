import seaborn as sns

import matplotlib.pyplot as plt
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
col=['Sno','Date','Time','State','IndianNAtional','ForiegnNAtional','Cured','Deaths','Confirmed']

df=pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv",names=col)

df=df.iloc[1:,:]

df
df.describe()
df["Cured"] = df["Cured"].astype(str).astype(int)

df["Deaths"] = df["Deaths"].astype(str).astype(int)

df["Confirmed"] = df["Confirmed"].astype(str).astype(int)

df.info()

df
df['Date']=pd.to_datetime(df['Date'],format='%d/%m/%y')



### Drop the columns which are not usefull

df.tail()
overall_cases=df[['Date','Confirmed','Cured','Deaths']].groupby('Date').sum().reset_index()

overall_cases.tail(10)
def case_daily(critrion,df):

    daily_case=[]

    df_length=len(df)

    for i in range(df_length):

        if i==0:

         daily_case.append(df[critrion].iloc[i])

        else:

         daily_case.append(df[critrion].iloc[i]-df[critrion].iloc[i-1])

    return daily_case
## add new column for new cases daily

overall_cases['Daily_cases']=case_daily(critrion='Confirmed',df=overall_cases)

### new column for Cured Daily

overall_cases['Daily_cured']=case_daily(critrion='Cured',df=overall_cases)

### Daily deaths

overall_cases['Daily_cases']=case_daily(critrion='Deaths',df=overall_cases)



overall_cases['Active_cases']=overall_cases['Confirmed']-(overall_cases['Cured']+overall_cases['Deaths'])

###Death % everday WRT TOTAL

overall_cases['Deaths%']=round((overall_cases['Deaths']/overall_cases['Confirmed'])*100,2)

### Cured EveryDay WRT TOTAL

overall_cases['Cured%']=round((overall_cases['Cured']/overall_cases['Confirmed'])*100,2)

### Growth % Everyday

overall_cases['Growth%']=round((overall_cases['Daily_cases']/overall_cases['Active_cases'])*100,2)
overall_cases.tail()
colors=['red','green','blue','black','orange']

def lineplot(x,y,title,getcolor,y_scale=None):

    f,(ax1)=plt.subplots(1,1,figsize=(12,5))

    plt1=sns.lineplot(x=x,y=y,ax=ax1,legend="full",color=getcolor)

    plt1.set_title(title,fontsize=15)

    plt.xticks(rotation=55)

    if y_scale !=None:

        plt.y_scale(y_scale)

    plt.show()


lineplot(x=overall_cases['Date'],

         y=overall_cases['Confirmed'],

         title="Total Corona Virus Cases in India",getcolor='blue')
#### Active cases in India

lineplot(x=overall_cases['Date'],

        y=overall_cases['Growth%'],

        title="Growth Rate of Coronavirus in India",

        getcolor='magenta')
def barplot(x,y,title,):

    f,(ax1) = plt.subplots(1,1,figsize=(20,10))

    plt1=sns.barplot(x=x,y=y,ax=ax1, saturation=0.8)

    plt1.set_title(title,fontsize=25)

    plt.xticks(rotation=55)

    plt.show()
###Daily cases function call

barplot(overall_cases['Date'].loc[50:].dt.strftime('%d-%b'),

       overall_cases['Daily_cases'].loc[50:],"Daily New Cases In India")
barplot(overall_cases['Date'].loc[50:].dt.strftime('%d-%b'),

       overall_cases['Daily_cured'].loc[50:],"Daily Recoveries in India")
###pie chart

labels=['Recoverd','Deaths']

sizes=[overall_cases['Cured'].max(),overall_cases['Deaths'].max()]

explode=(0,0.5)

fig1,ax1=plt.subplots(figsize=(12,5))

ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=True,startangle=90,frame=True)

ax1.axis('equal')

plt.tight_layout()

plt.title("Recovered v/s Deaths",fontsize=25)

plt.show()