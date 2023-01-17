import pandas as pd
data = pd.read_csv('../input/suicides-in-india/Suicides in India 2001-2012.csv')
data.head()
data.info()
import seaborn as sns

import matplotlib.pyplot as plt
sns.heatmap(data.isna())
sns.countplot(data=data,x='Gender')
sns.countplot(data=data,x='Age_group')
plt.figure(figsize=(15,10))

sns.countplot(data=data,y='State')
plt.figure(figsize=(15,25))

sns.countplot(data=data,y='Type')
plt.figure(figsize=(10,7))

sns.countplot(data=data,x='Type_code')
yearWiseData = []

year = data['Year'].unique()

for i in range(len(year)):

    dic = {

        'Year':year[i],

        'Count':len(data[data['Year']==year[i]])

    }

    yearWiseData.append(dic)
df = pd.DataFrame(data=yearWiseData)
sns.lineplot(data=df,x='Year',y='Count',marker='o',color='Red')
def stateWise(state,cat='all',arg='all',x=12,y=7,orient='x'):

    plt.figure(figsize=(x,y))

    if cat =='all':

        stateData = data[data['State']==state]

    elif arg=='all':

        stateData = data[data['State']==state]

    else:

        stateData = data[(data['State']==state)&(data[cat]==arg)]



    

    if orient == 'x':

        sns.countplot(data=stateData,x=cat)

    else:

        sns.countplot(data=stateData,y=cat)

   

    
stateWise('Gujarat','Type_code')
stateWise('Maharashtra','Type',y=25,orient='y')
def yearWise(year,cat='all',x=12,y=7,orient='x'):

    

    plt.figure(figsize=(x,y))

    yearData = data[data['Year']==year]

    

    if orient == 'x':

        sns.countplot(data=yearData,x=cat)

    else:

        sns.countplot(data=yearData,y=cat)

   

    
yearWise(2004,'Type',y=25,orient='y')