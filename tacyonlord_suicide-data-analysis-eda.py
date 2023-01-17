import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('../input/master.csv')
df.head()
df.info()
df.isnull().sum()
df.drop(['HDI for year', 'country-year'],axis=1,inplace=True)
df['age'].value_counts()
a_map = {

'75+ years' : 5,      

'55-74 years' : 4 ,    

'35-54 years' : 3,    

'25-34 years' : 2,    

'15-24 years' : 1,   

'5-14 years':  0     

 }
df['age'] = df['age'].map(a_map)
#Visualizations start here....

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
sns.set_style('darkgrid')

sns.countplot(df['age'])

#df['age'].plot(kind = 'hist')
sns.countplot(df['sex'])
df['developed'] = 0

df.loc[ df['gdp_per_capita ($)'] > df['gdp_per_capita ($)'].median(), 'developed']  = 1
df.head()
df.describe()
palette = {0 : 'red', 1: 'green'}
df_1 = df.groupby(['year','developed'],as_index=False).sum()

#df_1.drop(df_1[0])

plt.figure(figsize = (10,6)) 

plt.title('No. of suicides vs Year')

sns.lineplot(x = 'year', y ='suicides_no', data =df_1, hue='developed',palette = palette)

sns.scatterplot(x = 'year', y ='suicides_no', data =df_1, hue='developed', palette = palette)

#df_1.head()
df_1 = df.groupby(['year','developed'],as_index=False).mean()

df_1.head()

plt.figure(figsize = (10,6))

plt.title('Suicides/100k people vs Year')

sns.lineplot(x = 'year', y ='suicides/100k pop', data =df_1, hue='developed')

sns.scatterplot(x = 'year', y ='suicides/100k pop', data =df_1, hue='developed')
#A funtion which plots bar graphs for a given feature  and groups them by developed and not-developed 

def graph(feature):

    #define separate dataframes for developed and not developed  

    s_df = df[df['developed']==1].groupby([feature]).sum()['suicides_no']  #Developed

    d_df = df[df['developed']==0].groupby([feature]).sum()['suicides_no']  #Not Developed

    df1 = pd.DataFrame([s_df,d_df])

    df1.index = ['Developed','Not Developed']

    df1.plot(kind='bar') #we can stack them by using stacked = True

    plt.ylabel(feature)

    

    
graph('sex')

graph('age')
def graph2(feature):

    #define separate dataframes for developed and not developed

    #value_counts() counts unique objects in a given feature

    s_df = df[df['developed']==1].groupby([feature])['suicides/100k pop'].mean()  #Developed

    d_df = df[df['developed']==0].groupby([feature])['suicides/100k pop'].mean()  #Not Developed

    df1 = pd.DataFrame([s_df,d_df])

    df1.index = ['Developed','Not Developed']

    df1.plot(kind='bar') #we can stack them by using stacked = True

    plt.ylabel(feature)

    #plt.show()
graph2('age')

graph2('sex')
df.head()
plt.figure(figsize = (15,10))

df1 = df.groupby(['country','year'],as_index=False).sum()

plt.subplot(221)

plt.title('SuicideNos vs Population grouped by country and year')

sns.scatterplot(x = 'population', y = 'suicides_no', data = df1, alpha = 0.5)



plt.subplot(222)

plt.title('SuicideNos vs Population ')

sns.scatterplot(x = 'population', y = 'suicides_no', data = df, alpha = 0.5, hue = 'developed', palette = palette)



plt.subplot(223)

plt.title('Suicide Nos vs Gdp per Capita grouped by country and year')

sns.scatterplot(x = 'gdp_per_capita ($)', y = 'suicides_no', data = df1, alpha = 0.5)



plt.subplot(224)

plt.title('SuicideNos vs Gdp per Capita')

sns.scatterplot(x = 'gdp_per_capita ($)', y = 'suicides_no', data = df, alpha = 0.5, hue = 'developed', 

                palette = palette)
plt.figure(figsize = (15,10))

df1['suicides/100k pop'] = df1['suicides/100k pop']/12

df1['gdp_per_capita ($)'] = df1['gdp_per_capita ($)']/12



plt.subplot(221)

plt.title('Avg Suicide Rate vs Population grouped by country and year')

sns.scatterplot(x = 'population', y = 'suicides/100k pop', data = df1, alpha = 0.5)



plt.subplot(222)

plt.title('Avg Suicide Rate vs Population')

sns.scatterplot(x = 'population', y = 'suicides/100k pop', data = df, alpha = 0.5, hue = 'developed', palette = palette)



plt.subplot(223)

plt.title('Avg Suicide Rate vs Avg Gdp per capita grouped by country and year')

sns.scatterplot(x = 'gdp_per_capita ($)', y = 'suicides/100k pop', data = df1, alpha = 0.5)



plt.subplot(224)

plt.title('Avg Suicide Rate vs Avg Gdp per capita')

sns.scatterplot(x = 'gdp_per_capita ($)', y = 'suicides/100k pop', data = df, alpha = 0.5, hue = 'developed',

                palette = palette)
plt.figure(figsize = (14,7))

plt.subplot(121)

sns.countplot(df.generation)

plt.subplot(122)

df['generation'].value_counts().plot.pie(shadow = True,explode=[0.15,0.15,0.15,0.15,0.15,0.15], autopct='%1.1f%%')
graph('generation')

graph2('generation')
plt.figure(figsize=(8,8))

sns.heatmap(df1.corr(),annot=True)
df1 = df.drop(['developed'],axis=1)

sns.pairplot(df1,hue = 'generation',diag_kind = 'hist',palette = 'husl')