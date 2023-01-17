# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

df = pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv')



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_russia = df[df["Location"].str.contains("Russia")] 



df_russia = df_russia.copy()

#df_russia.loc[mask, "Datum"] = 0



df_russia["day"] = df_russia.iloc[:,4].apply(lambda x: x[8:10])



df_russia["month"] = df_russia.iloc[:,4].apply(lambda x: x[4:7])

#df_russia["mo"] = df_russia["mo"].apply(lambda x: strptime(x,'%b').tm_mon) 

d = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12 }

df_russia.month = df_russia.month.map(d)



df_russia["year"] = df_russia.iloc[:,4].apply(lambda x: x[12:17])



df_russia['Full_date'] = pd.to_datetime(df_russia[['year', 'month', 'day']])



#df_russia20200522 = df_russia[df_russia["Full_date"]== '2020-05-22'] 

#df_russia20200522.resample('M').mean()

#df_russia20200522



df_russia = df_russia.set_index(['Full_date'])

g = df_russia["Unnamed: 0"].groupby(pd.Grouper(freq="Y")).count()

#g.plot.line()

#ax = sns.lineplot(data=g, color="coral", label="Russia")

#df_russia
df_data = df.copy()

df_data["day"] = df_data.iloc[:,4].apply(lambda x: x[8:10])



df_data["month"] = df_data.iloc[:,4].apply(lambda x: x[4:7])



d = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12 }

df_data.month = df_data.month.map(d)



df_data["year"] = df_data.iloc[:,4].apply(lambda x: x[12:17])



df_data['Full_date'] = pd.to_datetime(df_data[['year', 'month', 'day']])

df_data = df_data.set_index(['Full_date'])

#df_data

df_data["Country"] = df_data["Location"].str.split(", ").str[-1]

#df_data["Country"] = df_data["Country"].replace(' ', '')

#df_data
df_data.loc[df_data["Location"].str.contains("USA"),"Country"] = "USA"

df_data.loc[df_data["Location"].str.contains("New Mexico"),"Country"] = "USA"

df_data.loc[df_data["Location"].str.contains("Yellow Sea"),"Country"] = "China"

df_data.loc[df_data["Location"].str.contains("Shahrud Missile Test Site"),"Country"] = "Iran"

df_data.loc[df_data["Location"].str.contains("Pacific Missile Range Facility"),"Country"] = "USA"

df_data.loc[df_data["Location"].str.contains("Barents Sea"),"Country"] = "Russia"

df_data.loc[df_data["Location"].str.contains("Gran Canaria"),"Country"] = "USA"

#df_data
#df_data["Country"].value_counts().plot.bar(figsize=(15, 7))

ax = df_data["Country"].value_counts().plot(kind='bar',figsize=(15, 7),xlabel="Country", ylabel="Overal count of flights by country")

totals = []

#for i in ax.patches:

#    totals.append(i.get_height())

#total = sum(totals)

#for i in ax.patches:

#    # get_x pulls left or right; get_height pushes up or down

#    ax.text(i.get_x()-.03, i.get_height()+.5, \

#            str(round((i.get_height()/total)*100, 2))+'%', fontsize=15,

#                color='dimgrey')







for p in ax.patches:

    ax.annotate(format(p.get_height(), '.1f'), 

                   (p.get_x() + p.get_width() / 2., p.get_height()), 

                   ha = 'center', va = 'center', 

                   xytext = (0, 9), 

                   textcoords = 'offset points')







#for p in ax.patches:

    #ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.008))



plt.show()

#for i, v in enumerate(y):

#    ax.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')

    

#all_countrys.plot.bar(stacked=True, figsize=(15, 7))
g = df_data["Country"].groupby(pd.Grouper(freq="Y")).count()

#g
df_russia = df_data[df_data["Country"].str.contains("Russia")] 

df_USA = df_data[df_data["Country"].str.contains("USA")] 

df_Kazakhstan = df_data[df_data["Country"].str.contains("Kazakhstan")] 

df_France = df_data[df_data["Country"].str.contains("France")] 

df_China = df_data[df_data["Country"].str.contains("China")] 

df_Japan = df_data[df_data["Country"].str.contains("Japan")] 

df_India = df_data[df_data["Country"].str.contains("India")] 

df_Pacific = df_data[df_data["Country"].str.contains("Pacific Ocean")] 

df_Iran = df_data[df_data["Country"].str.contains("Iran")] 

df_Zealand = df_data[df_data["Country"].str.contains("New Zealand")]
russia = pd.DataFrame(df_russia["Country"].groupby(pd.Grouper(freq="Y")).count()).rename(columns = {'Country': 'Russia'}, inplace = False)

USA = pd.DataFrame(df_USA["Country"].groupby(pd.Grouper(freq="Y")).count()).rename(columns = {'Country': 'USA'}, inplace = False)

Kazakhstan = pd.DataFrame(df_Kazakhstan["Country"].groupby(pd.Grouper(freq="Y")).count()).rename(columns = {'Country': 'Kazakhstan'}, inplace = False)

France = pd.DataFrame(df_France["Country"].groupby(pd.Grouper(freq="Y")).count()).rename(columns = {'Country': 'France'}, inplace = False)

China = pd.DataFrame(df_China["Country"].groupby(pd.Grouper(freq="Y")).count()).rename(columns = {'Country': 'China'}, inplace = False)

Japan = pd.DataFrame(df_Japan["Country"].groupby(pd.Grouper(freq="Y")).count()).rename(columns = {'Country': 'Japan'}, inplace = False)

India = pd.DataFrame(df_India["Country"].groupby(pd.Grouper(freq="Y")).count()).rename(columns = {'Country': 'India'}, inplace = False)

Pacific = pd.DataFrame(df_Pacific["Country"].groupby(pd.Grouper(freq="Y")).count()).rename(columns = {'Country': 'Pacific'}, inplace = False)

Iran = pd.DataFrame(df_Iran["Country"].groupby(pd.Grouper(freq="Y")).count()).rename(columns = {'Country': 'Iran'}, inplace = False)

Zealand = pd.DataFrame(df_Zealand["Country"].groupby(pd.Grouper(freq="Y")).count()).rename(columns = {'Country': 'Zealand'}, inplace = False)



all_countrys = pd.concat([russia, USA,Kazakhstan,France,China,Japan,India,Pacific,Iran,Zealand], axis=1, sort=False)

all_countrys=all_countrys.fillna(0)
#plt.figure(figsize=(15, 7))

#sns.palplot(sns.color_palette("husl", 10))

#plt.set_title('Countries')

#data = all_countrys

#sns.palplot(sns.husl_palette(10))

plt.figure(figsize=(20,10))

sns.set(style="whitegrid")

sns.lineplot(data=all_countrys,linewidth=3,dashes=False,hue='label', style='label',markers= ["o","o","o","o","o","o","o","o","o","o"])
x_ticks = []

[x_ticks.append(pd.to_datetime(x).date()) for x in list(all_countrys.index.values)]



all_countrys.index =all_countrys.index.map(lambda t: t.strftime('%Y'))

#all_countrys.reindex(x_ticks)

all_countrys.index.names = ['Year']

#all_countrys
#plt.figure(figsize=(15, 7))

all_countrys.plot.bar(stacked=True, figsize=(15, 7), title="Number of flights")

plt.legend(bbox_to_anchor=(1.0, 1.0155), loc='upper left')

#all_countrys.size().unstack().plot(kind='bar',stacked=True)

plt.show(block=True)
all_countrys_perc = all_countrys.copy()



all_countrys_perc["summa"] = all_countrys_perc.sum(axis = 1, skipna = True)



all_countrys_perc["Russia"] = all_countrys_perc["Russia"]/all_countrys_perc["summa"]*100

all_countrys_perc["USA"] = all_countrys_perc["USA"]/all_countrys_perc["summa"]*100

all_countrys_perc["Kazakhstan"] = all_countrys_perc["Kazakhstan"]/all_countrys_perc["summa"]*100

all_countrys_perc["France"] = all_countrys_perc["France"]/all_countrys_perc["summa"]*100

all_countrys_perc["China"] = all_countrys_perc["China"]/all_countrys_perc["summa"]*100

all_countrys_perc["Japan"] = all_countrys_perc["Japan"]/all_countrys_perc["summa"]*100

all_countrys_perc["India"] = all_countrys_perc["India"]/all_countrys_perc["summa"]*100

all_countrys_perc["Pacific"] = all_countrys_perc["Pacific"]/all_countrys_perc["summa"]*100

all_countrys_perc["Iran"] = all_countrys_perc["Iran"]/all_countrys_perc["summa"]*100

all_countrys_perc["Zealand"] = all_countrys_perc["Zealand"]/all_countrys_perc["summa"]*100

all_countrys_perc = all_countrys_perc.drop(labels='summa', axis=1)

#all_countrys_perc
#plt.figure(figsize=(15, 7))

all_countrys_perc.plot.bar(stacked=True, figsize=(15, 7), title="Percentage of flights by each country")

plt.legend(bbox_to_anchor=(1.0, 1.0155), loc='upper left')

#plt.legend(handles=[p1, p2], title='title', bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)

#all_countrys.size().unstack().plot(kind='bar',stacked=True)

#plt.show(block=True)