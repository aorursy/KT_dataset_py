# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from IPython.display import Image
from IPython.core.display import HTML
# Any results you write to the current directory are saved as output.

import plotly.plotly as py
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import plotly.plotly as py
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
Olympics_Master = pd.read_csv("../input/athlete_events.csv")
Olympics_Region = pd.read_csv("../input/noc_regions.csv")
Olympics_Master.head()
Olympics_Master.describe()
print("Dataset Shape\nRows:{0}\nColumns:{1}".format(Olympics_Master.shape[0],Olympics_Master.shape[1]))
sns.heatmap(Olympics_Master.isnull(), cbar=False)
plt.show()
import matplotlib as plt
x = Olympics_Master.isnull().sum().to_frame()
print(x)
imp_col = ['Age','Height','Weight']
for col in  imp_col:
    Olympics_Master[col] = Olympics_Master[col].fillna(np.mean(Olympics_Master[col]))
    Olympics_Master[col] = np.round(Olympics_Master[col],1)
sns.heatmap(Olympics_Master.isnull(), cbar=False)
df1 = Olympics_Master[Olympics_Master.Season=='Summer'].groupby(['Year']).sum()
df2 = Olympics_Master[Olympics_Master.Season=='Winter'].groupby(['Year']).sum()
df3 = Olympics_Master[Olympics_Master.Season=='Summer'].groupby(['Year'])['NOC'].nunique()
df4 = Olympics_Master[Olympics_Master.Season=='Winter'].groupby(['Year'])['NOC'].nunique()
df5 = Olympics_Master[Olympics_Master.Season=='Summer'].groupby(['Year'])['Event'].nunique()
df6 = Olympics_Master[Olympics_Master.Season=='Winter'].groupby(['Year'])['Event'].nunique()
import matplotlib.pyplot as plt
plt.style.use('dark_background')
fig,ax = plt.subplots(3,1)
fig.tight_layout()
fig.set_figheight(12)
fig.set_figwidth(10)

ax[0].plot(df1.ID, marker = 'o', color = 'red', linestyle = '-')
ax[0].plot(df2.ID, marker = 'o', color = 'blue', linestyle = '-')
ax[0].set_title('Number of Athletes',fontsize=16)
ax[0].set_xlabel('Year',fontsize=14)
ax[0].set_ylabel('Athletes',fontsize=14)
ax[0].legend(['Summer','Winter'])
#ax[0].grid(b=True, which='major', color='white')

ax[1].plot(df3, marker = '*',color='red')
ax[1].plot(df4, marker='*',color='blue')
ax[1].set_title('Number of Nations',fontsize=16)
ax[1].set_xlabel('Year',fontsize=14)
ax[1].set_ylabel('Nations',fontsize=14)
ax[1].legend(['Summer','Winter'])
#ax[1].grid(b=True, which='major', color='white')

ax[2].plot(df5, marker = '*',color='red')
ax[2].plot(df6, marker='*',color='blue')
ax[2].set_title('Number of Athletes',fontsize=16)
ax[2].set_xlabel('Year',fontsize=14)
ax[2].set_ylabel('Events',fontsize=14)
ax[2].legend(['Summer','Winter'])
#ax[2].grid(b=True, which='major', color='white')
plt.show()
df0 = Olympics_Master[Olympics_Master.Sport=='Art Competitions'].groupby(['Year'])['Event'].nunique()
df1 = Olympics_Master[Olympics_Master.Sport=='Art Competitions'].groupby(['Year'])['NOC'].nunique()
df2 = Olympics_Master[Olympics_Master.Sport=='Art Competitions'].groupby(['Year'])['ID'].nunique()
plt.style.use('ggplot')
fig, ax = plt.subplots(3,1)
fig.tight_layout()
fig.set_figheight(12)
fig.set_figwidth(10)

ax[0].plot(df0, marker = 'o', color='black')
ax[0].set_ylabel('Events', fontsize='14', fontdict=dict(weight='bold'))

ax[1].plot(df1, marker = 'o', color='black')
ax[1].set_ylabel('Nations', fontsize='14', fontdict=dict(weight='bold'))

ax[2].plot(df2, marker = 'o', color='black')
ax[2].set_ylabel('Artists', fontsize='14', fontdict=dict(weight='bold'))
ax[2].set_xlabel('Year', fontsize='14', fontdict=dict(weight='bold'))
plt.show()
from matplotlib import rcParams
rcParams['figure.figsize'] = 16,8
#df0 = Olympics_Master[Olympics_Master['Medal'].notnull()].groupby(['Team'])['Medal'].value_counts()
df0 = Olympics_Master[(Olympics_Master['Medal'].notnull()) & (Olympics_Master['Sport'] =='Art Competitions')].groupby(['Team'])['Medal'].value_counts()
df0 = df0.reset_index(level=0)
df0.rename(columns={'Medal': 'Count'}, inplace=True)
df0 = df0.reset_index()
df0.head()
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
df0 = Olympics_Master[(Olympics_Master['Medal'].notnull()) & (Olympics_Master['Sport'] =='Art Competitions')].groupby('Team').count().reset_index()[['Team','Medal']]
df0 = df0.sort_values(by=['Medal'],ascending=False)
fig = sns.barplot(x='Medal', y='Team',data=df0,palette='dark')

fig.set_xlabel("Count",fontsize=16)
fig.set_ylabel("Team",fontsize=16)
fig.set_title('Historical Medal Counts from Art Competitions',fontsize=16)
plt.show()
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
df0 = Olympics_Master[(Olympics_Master['Medal'].notnull()) & (Olympics_Master['Sport'] =='Art Competitions') & (Olympics_Master['Year'] ==1936)].groupby('Team').count().reset_index()[['Team','Medal']]
df0 = df0.sort_values(by=['Medal'],ascending=False)
fig = sns.barplot(x='Medal', y='Team',data=df0,palette='dark')
fig.set_xlabel("Count",fontsize=16)
fig.set_ylabel("Team",fontsize=16)
fig.set_title('Nazi Domination of Art Competitions at the 1936 Olymics',fontsize=16)
plt.show()
original = [1994,1998,2002,2006,2010,2014]
for i in Olympics_Master.index:
    if (Olympics_Master.at[i, 'Year'] in original):
        Olympics_Master.at[i, 'Year'] =   Olympics_Master.at[i, 'Year'] + 2
print(Olympics_Master.Year.unique())
df0 = Olympics_Master[(Olympics_Master.Sex=='M')&(Olympics_Master.Sport !='Art Competitions')].groupby(["Year"])['ID'].nunique()
df0 = df0.reset_index()
df1 = Olympics_Master[(Olympics_Master.Sex=='F')&(Olympics_Master.Sport !='Art Competitions')].groupby(["Year"])['ID'].nunique()
df1 = df1.reset_index()
fig, ax = plt.subplots()
ax.plot(df0.Year, df0.ID,marker ='o',color='blue',label='M')
ax.plot(df1.Year, df1.ID, marker ='o',color='red',label='F')
ax.legend()
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Athletes',fontsize=14)
ax.set_title('Number of Male and Female Olympians Over Time',fontsize=16)
plt.show()
df0 = Olympics_Master[(Olympics_Master.Sport !='Art Competitions')].groupby(["Year", "NOC", "Sex"])['ID'].nunique().to_frame()
df0 = df0.reset_index()

YearsPlt = [2016,1996,1976,1956,1936]  ## To be plotted data for Years
for i in df0.index:
    if (df0.at[i, 'Year'] not in YearsPlt):
        df0.drop([i], inplace=True)
dM = df0[df0.Sex=='M']
dF = df0[df0.Sex=='F']
dF = dF.drop(['Sex'], axis=1)
dM = dM.drop(['Sex'], axis=1)
dF = dF.rename(columns={"ID": "Female"})
dM = dM.rename(columns={"ID": "Male"})

dPlot = pd.merge(dF, dM, how='inner', on=['Year', 'NOC'])
dPlot.head()

fig = sns.lmplot(x='Male',y='Female',data=dPlot,hue='Year',palette='muted',height=8,aspect=1.5,truncate=True)
fig.set(ylim=(-25, 425), xlim=(0, 500))
plt.title("Female vs Male Plympians from partiipating NOCs", fontsize=16)
plt.show()
# Manipulating dataset to calcuate Women Athlete Count and Medal Ratios
df = Olympics_Master[Olympics_Master.Year == 1936].groupby(['Year', 'NOC', 'Sex']).agg({'ID' : 'nunique','Medal' : 'count'})
df = df.reset_index()

dF = df[df.Sex=='F']
dM = df[df.Sex=='M']

dF = dF.rename(columns={"ID": "Athletes_F", "Medal":"Medal_F"})
dM = dM.rename(columns={"ID": "Athletes_M", "Medal":"Medal_M"})

dF = dF.drop(['Sex'], axis=1)
dM = dM.drop(['Sex'], axis=1)

dPlot = pd.merge(dF, dM, how='inner', on=['Year', 'NOC'])
dPlot = dPlot[(dPlot.Athletes_F + dPlot.Athletes_M)>49]

dPlot['Ratio1'] = dPlot.Athletes_F/(dPlot.Athletes_F+dPlot.Athletes_M)
dPlot['Ratio2'] = dPlot.Medal_F/(dPlot.Medal_F+dPlot.Medal_M)
dPlot = dPlot.sort_values('Ratio1')
# dPlot.head()
plt.scatter(x='Ratio1',y='NOC',data=dPlot,color='black',label='Athletes')
plt.scatter(x='Ratio2',y='NOC',data=dPlot,color='gold',label='Medals')
plt.xlim(-0.05,1)
plt.legend(bbox_to_anchor=(1.15, 0.5), fontsize=14)
plt.xticks(np.arange(0, 1, step=0.25))
plt.title('1936 Olympics',fontsize=18)
plt.xlabel('Proportion Female', fontsize=16)
plt.ylabel('NOC', fontsize=16)
plt.show()
df_1936 = Olympics_Master[(Olympics_Master.Year == 1936) & (Olympics_Master.Sex == 'F')].dropna()
df_1936 = df_1936.groupby(['NOC','Medal']).agg({'Medal':'count'})
df_1936 = df_1936.rename(columns = {'Medal':'Count'})
df_1936 = df_1936.reset_index()
# Adding Count zero for .. Tweaking dataset inorder to get stacked bar plot in matplotlib.
for n in df_1936.NOC.unique():
    if(df_1936[(df_1936.NOC == n)&(df_1936.Medal == 'Silver')]).empty:
        df_1936.loc[-1] = [n, 'Silver', 0]  # adding a row
        df_1936.index = df_1936.index + 1  # shifting index
        df_1936 = df_1936.sort_index()  # sorting by index.
        
for n in df_1936.NOC.unique():
    if(df_1936[(df_1936.NOC == n)&(df_1936.Medal == 'Gold')]).empty:
        df_1936.loc[-1] = [n, 'Gold', 0]  # adding a row
        df_1936.index = df_1936.index + 1  # shifting index
        df_1936 = df_1936.sort_index()  # sorting by index.
        
for n in df_1936.NOC.unique():
    if(df_1936[(df_1936.NOC == n)&(df_1936.Medal == 'Bronze')]).empty:
        df_1936.loc[-1] = [n, 'Bronze', 0]  # adding a row
        df_1936.index = df_1936.index + 1  # shifting index
        df_1936 = df_1936.sort_index()  # sorting by index.
        
#Sort Values in Order of NOC        
df_1936 = df_1936.sort_values(by='NOC')

#Setting width and left parameter for stacked barplot
C_Gold =df_1936[df_1936.Medal=='Gold']['Count']
C_Silver = df_1936[df_1936.Medal=='Silver']['Count']
C_Bronze = df_1936[df_1936.Medal=='Bronze']['Count']
B_bottom = [sum(x) for x in zip(df_1936[df_1936.Medal=='Gold']['Count'].tolist(), df_1936[df_1936.Medal=='Silver']['Count'].tolist())]

#creating stacked bar plot
plt.barh(y=(df_1936[df_1936.Medal=='Gold']['NOC']), width=C_Gold,label='Gold',color='#D4AF37')
plt.barh(y=df_1936[df_1936.Medal=='Silver']['NOC'], width=C_Silver, left=C_Gold,label='Silver',color='silver')
plt.barh(y=df_1936[df_1936.Medal=='Bronze']['NOC'], width=C_Bronze,left=B_bottom,label='Bronze',color='#CD7F32')

#Manipulating Labels
plt.legend(fontsize=16)
plt.xlabel('Medal Count', fontsize=14)
plt.ylabel('NOC', fontsize=14)
plt.title('Medal Count For Women at 1936 Olympics',fontsize=16)
plt.show()
# Manipulating dataset to calcuate Women Athlete Count and Medal Ratios
df = Olympics_Master[Olympics_Master.Year == 1976].groupby(['Year', 'NOC', 'Sex']).agg({'ID' : 'nunique','Medal' : 'count'})
df = df.reset_index()

dF = df[df.Sex=='F']
dM = df[df.Sex=='M']

dF = dF.rename(columns={"ID": "Athletes_F", "Medal":"Medal_F"})
dM = dM.rename(columns={"ID": "Athletes_M", "Medal":"Medal_M"})

dF = dF.drop(['Sex'], axis=1)
dM = dM.drop(['Sex'], axis=1)

dPlot = pd.merge(dF, dM, how='inner', on=['Year', 'NOC'])
dPlot = dPlot[(dPlot.Athletes_F + dPlot.Athletes_M)>49]

dPlot['Ratio1'] = dPlot.Athletes_F/(dPlot.Athletes_F+dPlot.Athletes_M)
dPlot['Ratio2'] = dPlot.Medal_F/(dPlot.Medal_F+dPlot.Medal_M)
dPlot = dPlot.sort_values('Ratio1')
# dPlot.head()

plt.scatter(x='Ratio1',y='NOC',data=dPlot,color='black',label='Athletes')
plt.scatter(x='Ratio2',y='NOC',data=dPlot,color='gold',label='Medals')
plt.xlim(-0.05,1)
plt.legend(bbox_to_anchor=(1.15, 0.5), fontsize=14)
plt.xticks(np.arange(0, 1, step=0.25))
plt.title('1976 Olympics',fontsize=18)
plt.xlabel('Proportion Female', fontsize=16)
plt.ylabel('NOC', fontsize=16)
plt.show()
df_1976 = Olympics_Master[(Olympics_Master.Year == 1976) & (Olympics_Master.Sex == 'F')].dropna()
df_1976 = df_1976.groupby(['NOC','Medal']).agg({'Medal':'count'})
df_1976 = df_1976.rename(columns = {'Medal':'Count'})
df_1976 = df_1976.reset_index()

# Adding Count zero for .. Tweaking dataset inorder to get stacked bar plot in matplotlib.
for n in df_1976.NOC.unique():
    if(df_1976[(df_1976.NOC == n)&(df_1976.Medal == 'Silver')]).empty:
        df_1976.loc[-1] = [n, 'Silver', 0]  # adding a row
        df_1976.index = df_1976.index + 1  # shifting index
        df_1976 = df_1976.sort_index()  # sorting by index.
        
for n in df_1976.NOC.unique():
    if(df_1976[(df_1976.NOC == n)&(df_1976.Medal == 'Gold')]).empty:
        df_1976.loc[-1] = [n, 'Gold', 0]  # adding a row
        df_1976.index = df_1976.index + 1  # shifting index
        df_1976 = df_1976.sort_index()  # sorting by index.
        
for n in df_1976.NOC.unique():
    if(df_1976[(df_1976.NOC == n)&(df_1976.Medal == 'Bronze')]).empty:
        df_1976.loc[-1] = [n, 'Bronze', 0]  # adding a row
        df_1976.index = df_1976.index + 1  # shifting index
        df_1976 = df_1976.sort_index()  # sorting by index.
        
#Sort Values in Order of NOC        
df_1976 = df_1976.sort_values(by='NOC')

#Setting width and left parameter for stacked barplot
C_Gold =df_1976[df_1976.Medal=='Gold']['Count']
C_Silver =df_1976[df_1976.Medal=='Silver']['Count']
C_Bronze = df_1976[df_1976.Medal=='Bronze']['Count']
B_bottom = [sum(x) for x in zip(df_1976[df_1976.Medal=='Gold']['Count'].tolist(), df_1976[df_1976.Medal=='Silver']['Count'].tolist())]

#creating stacked bar plot
plt.barh(y=(df_1976[df_1976.Medal=='Gold']['NOC']), width=C_Gold,label='Gold',color='#D4AF37')
plt.barh(y=df_1976[df_1976.Medal=='Silver']['NOC'], width=C_Silver, left=C_Gold,label='Silver',color='silver')
plt.barh(y=df_1976[df_1976.Medal=='Bronze']['NOC'], width=C_Bronze,left=B_bottom,label='Bronze',color='#CD7F32')

#Manipulating Labels
plt.legend(fontsize=16)
plt.xlabel('Medal Count', fontsize=14)
plt.ylabel('NOC', fontsize=14)
plt.title('Medal Count For Women at 1976 Olympics',fontsize=16)
plt.show()
# Manipulating dataset to calcuate Women Athlete Count and Medal Ratios
df = Olympics_Master[Olympics_Master.Year == 2016].groupby(['Year', 'NOC', 'Sex']).agg({'ID' : 'nunique','Medal' : 'count'})
df = df.reset_index()

dF = df[df.Sex=='F']
dM = df[df.Sex=='M']

dF = dF.rename(columns={"ID": "Athletes_F", "Medal":"Medal_F"})
dM = dM.rename(columns={"ID": "Athletes_M", "Medal":"Medal_M"})

dF = dF.drop(['Sex'], axis=1)
dM = dM.drop(['Sex'], axis=1)

dPlot = pd.merge(dF, dM, how='inner', on=['Year', 'NOC'])
dPlot = dPlot[(dPlot.Athletes_F + dPlot.Athletes_M)>49]

dPlot['Ratio1'] = dPlot.Athletes_F/(dPlot.Athletes_F+dPlot.Athletes_M)
dPlot['Ratio2'] = dPlot.Medal_F/(dPlot.Medal_F+dPlot.Medal_M)
dPlot = dPlot.sort_values('Ratio1')
# dPlot.head()

plt.figure(figsize=(12,20))
plt.scatter(x='Ratio1',y='NOC',data=dPlot,color='black',label='Athletes')
plt.scatter(x='Ratio2',y='NOC',data=dPlot,color='gold',label='Medals')
plt.xlim(-0.05,1)
plt.legend(bbox_to_anchor=(1.2, 0.5), fontsize=14)
plt.xticks(np.arange(0, 1, step=0.25))
plt.title('2016 Olympics',fontsize=18)
plt.xlabel('Proportion Female', fontsize=16)
plt.ylabel('NOC', fontsize=16)
plt.show()
df_2016 = Olympics_Master[(Olympics_Master.Year == 2016) & (Olympics_Master.Sex == 'F')].dropna()
df_2016 = df_2016.groupby(['NOC','Medal']).agg({'Medal':'count'})
df_2016 = df_2016.rename(columns = {'Medal':'Count'})
df_2016 = df_2016.reset_index()

# Adding Count zero for .. Tweaking dataset inorder to get stacked bar plot in matplotlib.
for n in df_2016.NOC.unique():
    if(df_2016[(df_2016.NOC == n)&(df_2016.Medal == 'Silver')]).empty:
        df_2016.loc[-1] = [n, 'Silver', 0]  # adding a row
        df_2016.index = df_2016.index + 1  # shifting index
        df_2016 = df_2016.sort_index()  # sorting by index.
        
for n in df_2016.NOC.unique():
    if(df_2016[(df_2016.NOC == n)&(df_2016.Medal == 'Gold')]).empty:
        df_2016.loc[-1] = [n, 'Gold', 0]  # adding a row
        df_2016.index = df_2016.index + 1  # shifting index
        df_2016 = df_2016.sort_index()  # sorting by index.
        
for n in df_2016.NOC.unique():
    if(df_2016[(df_2016.NOC == n)&(df_2016.Medal == 'Bronze')]).empty:
        df_2016.loc[-1] = [n, 'Bronze', 0]  # adding a row
        df_2016.index = df_2016.index + 1  # shifting index
        df_2016 = df_2016.sort_index()  # sorting by index.
        
#Sort Values in Order of NOC        
df_2016 = df_2016.sort_values(by='NOC')

#Setting width and left parameter for stacked barplot
C_Gold =df_2016[df_2016.Medal=='Gold']['Count']
C_Silver =df_2016[df_2016.Medal=='Silver']['Count']
C_Bronze = df_2016[df_2016.Medal=='Bronze']['Count']
B_bottom = [sum(x) for x in zip(df_2016[df_2016.Medal=='Gold']['Count'].tolist(), df_2016[df_2016.Medal=='Silver']['Count'].tolist())]

#creating stacked bar plot
plt.figure(figsize=(12,20))
plt.barh(y=(df_2016[df_2016.Medal=='Gold']['NOC']), width=C_Gold,label='Gold',color='#D4AF37')
plt.barh(y=df_2016[df_2016.Medal=='Silver']['NOC'], width=C_Silver, left=C_Gold,label='Silver',color='silver')
plt.barh(y=df_2016[df_2016.Medal=='Bronze']['NOC'], width=C_Bronze,left=B_bottom,label='Bronze',color='#CD7F32')

#Manipulating Labels
plt.legend(fontsize=16)
plt.xlabel('Medal Count', fontsize=14)
plt.ylabel('NOC', fontsize=14)
plt.title('Medal Count For Women at 2016 Olympics',fontsize=16)
plt.show()

df_1928 = Olympics_Master[Olympics_Master.Games =='1928 Summer'].groupby(['NOC']).agg({'ID':'nunique'})
df_1928 = df_1928.reset_index()
df_1928 = pd.merge(df_1928, Olympics_Region, how='left', on=['NOC'])
#df_1928.head()
df_1928.drop(axis=1, columns=(['notes']),inplace=True)


df_1928 = df_1928.dropna()
df_1928.sort_values(by='NOC',ascending=True,inplace=True)
#df_1928.head()

data = dict(
        type = 'choropleth',
        locations = df_1928['NOC'],
        z = df_1928['ID'],
        colorscale='Reds',
        colorbar = {'title' : 'Athletes'},
      ) 

layout = dict(
    title = '1928 Olympics',
    geo = dict(
        showframe = False,
        projection = {'type':'natural earth'}
    )
)


choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)
df_1972 = Olympics_Master[Olympics_Master.Games =='1972 Summer'].groupby(['NOC']).agg({'ID':'nunique'})
df_1972 = df_1972.reset_index()
df_1972 = pd.merge(df_1972, Olympics_Region, how='left', on=['NOC'])
#df_1972.head()
df_1972.drop(axis=1, columns=(['notes']),inplace=True)


df_1972 = df_1972.dropna()
df_1972.sort_values(by='NOC',ascending=True,inplace=True)
#df_1972.head()

data = dict(
        type = 'choropleth',
        locations = df_1972['NOC'],
        z = df_1972['ID'],
        colorscale='Reds',
        colorbar = {'title' : 'Athletes'},
      ) 

layout = dict(
    title = '1972 Olympics',
    geo = dict(
        showframe = False,
        projection = {'type':'natural earth'}
    )
)


choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)
df_2016 = Olympics_Master[Olympics_Master.Games =='2016 Summer'].groupby(['NOC']).agg({'ID':'nunique'})
df_2016 = df_2016.reset_index()
df_2016 = pd.merge(df_2016, Olympics_Region, how='left', on=['NOC'])
#df_2016.head()
df_2016.drop(axis=1, columns=(['notes']),inplace=True)


df_2016 = df_2016.dropna()
df_2016.sort_values(by='NOC',ascending=True,inplace=True)
#df_2016.head()

data = dict(
        type = 'choropleth',
        locations = df_2016['NOC'],
        z = df_2016['ID'],
        colorscale='Reds',
        colorbar = {'title' : 'Athletes'},
      ) 

layout = dict(
    title = '2016 Olympics',
    height=600, width=900,
    autosize=True,
    geo = dict(
        showframe = True,
        projection = {'type':'mercator'}
    )
)     
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)
sns.boxplot(data=Olympics_Master[Olympics_Master.Year>=1960],y='Height',x='Year',hue='Sex',width=0.5)
plt.show()
sns.boxplot(data=Olympics_Master[Olympics_Master.Year>=1960],y='Weight',x='Year',hue='Sex',width=0.5)
plt.show()
Olympics_India = Olympics_Master[Olympics_Master.NOC=='IND'].reset_index(drop=True)
Olympics_India.head()
print('India first participated at the Olympic Games in {}'.format(Olympics_India['Year'].min()))

print('with a {} athlete (Norman Pritchard) winning two medals- both silver- in athletics.'.format(Olympics_India[Olympics_India.Year == 1900]['ID'].nunique()))

print('Total number if medals: {}'.format(Olympics_India[Olympics_India.Year == 1900]['Medal'].count()))

print('First appearance at Winter Olympics: {}'.format(Olympics_India[Olympics_India.Season == 'Winter']['Year'].min()))
import IPython
url = 'https://en.wikipedia.org/wiki/India_at_the_Olympics#List_of_competitors'
iframe = '<iframe src=' + url + ' width=1200 height=350></iframe>'
IPython.display.HTML(iframe)
India_Summer = Olympics_India[Olympics_India.Season=='Summer']
India_Winter = Olympics_India[Olympics_India.Season=='Winter']
Sports = India_Summer.groupby(['Year']).agg({'Sport': lambda x: len(x.unique())}).reset_index()

Men = India_Summer[India_Summer.Sex=='M'].groupby(['Year']).agg({'ID': lambda x: len(x.unique())}).reset_index()
Men.columns = ['Year','Men']
Women = India_Summer[India_Summer.Sex=='F'].groupby(['Year']).agg({'ID': lambda x: len(x.unique())}).reset_index()
Women.columns = ['Year','Women']

Gold = India_Summer[India_Summer.Medal=='Gold'].groupby(['Year']).agg({'Medal': len}).reset_index()
Gold.columns = ['Year','Gold']
Silver = India_Summer[India_Summer.Medal=='Silver'].groupby(['Year']).agg({'Medal': len}).reset_index()
Silver.columns = ['Year','Silver']
Bronze = India_Summer[India_Summer.Medal=='Bronze'].groupby(['Year']).agg({'Medal': len}).reset_index()
Bronze.columns = ['Year','Bronze']

# This approach doesn't seem to be efficient. I want to do this using the pivot_table function. 
#Will update same once achieved. Looking for suggestions.
from functools import reduce
dfs = [Sports, Men, Women,Gold,Silver,Bronze]
df_final = reduce(lambda left,right: pd.merge(left,right,on='Year',how='outer'), dfs).fillna(0).astype(int)
df_final
Sports = India_Winter.groupby(['Year']).agg({'Sport': lambda x: len(x.unique())}).reset_index()

Men = India_Winter[India_Winter.Sex=='M'].groupby(['Year']).agg({'ID': lambda x: len(x.unique())}).reset_index()
Men.columns = ['Year','Men']
Women = India_Winter[India_Winter.Sex=='F'].groupby(['Year']).agg({'ID': lambda x: len(x.unique())}).reset_index()
Women.columns = ['Year','Women']

Gold = India_Winter[India_Winter.Medal=='Gold'].groupby(['Year']).agg({'Medal': len}).reset_index()
Gold.columns = ['Year','Gold']
Silver = India_Winter[India_Winter.Medal=='Silver'].groupby(['Year']).agg({'Medal': len}).reset_index()
Silver.columns = ['Year','Silver']
Bronze = India_Winter[India_Winter.Medal=='Bronze'].groupby(['Year']).agg({'Medal': len}).reset_index()
Bronze.columns = ['Year','Bronze']

dfs = [Sports, Men, Women,Gold,Silver,Bronze]
df_final = reduce(lambda left,right: pd.merge(left,right,on='Year',how='outer'), dfs).fillna(0).astype(int)
df_final
df1 = Olympics_India[Olympics_India.Season=='Summer'].groupby(['Year']).sum()
df2 = Olympics_India[Olympics_India.Season=='Winter'].groupby(['Year']).sum()

fig,ax = plt.subplots()
plt.style.use('grayscale')
fig.set_figheight(4)
fig.set_figwidth(12)
fig.tight_layout()
ax.plot(df1.ID, marker = 'o', color = 'red', linestyle = '-')
ax.plot(df2.ID, marker = 'o', color = 'blue', linestyle = '-')
ax.set_title('Number of Athletes',fontsize=16)
ax.set_xlabel('Year',fontsize=14)
ax.set_ylabel('Athletes',fontsize=14)
ax.legend(['Summer','Winter'])
plt.show()

