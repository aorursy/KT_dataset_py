import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from mpl_toolkits.basemap import Basemap
import itertools
df=pd.read_csv('../input/FAO.csv',encoding='ISO-8859-1')
df.shape
df.isnull().sum()
df.describe()
df.sample(10)
df.dropna().shape
df.shape
df=df.fillna(0)
df.sample(5)
list(df.columns)[:10]
Years=list(df.columns)[10:]
Countries=list(df.Area.unique())
df['latitude'][df.Area=='Japan'].unique()
df_Element=pd.concat([df.loc[:,'Element'],df.loc[:,Years]],axis=1)
Food_Type=df_Element.groupby('Element')
list(Food_Type.Y2013)
plt.figure(figsize=(20,15))
sns.heatmap(Food_Type.corr())
tot = Food_Type.sum()
tot.apply(lambda x: x/x.sum()*100)
for name, group in Food_Type:
    print(name,group['Y1961'].sum())
pd.DataFrame([Food_Type[year].sum() for year in Years]).plot(kind='bar',figsize=(20,10),color=('rg'),fontsize=14,width=.95,alpha=.5)
plt.yticks(np.arange(0,1.05*10**7,5*10**5))
plt.ylabel('Production in 1000 tonnes')
plt.title('Food & Feed vs Years',fontsize=14)
plt.show()
print('min longitude is',min(df.longitude))
print('max longitude is',max(df.longitude))
print('min latitude is',min(df.latitude))
print('max latitude is',max(df.latitude))
q=df.groupby(['Element','Area']).sum().loc[:,Years]
q=q.reset_index()
q['latitude']=q.apply(lambda row: df['latitude'][df.Area==row['Area']].unique()[0],axis=1)
q['longitude']=q.apply(lambda row: df['longitude'][df.Area==row['Area']].unique()[0],axis=1)
q
def food_map(lon,lat,df,year):
    fig,ax = plt.subplots()
    fig.set_size_inches(12,20)
    plt.gca().set_color_cycle(['crimson','blue'])
    
    m = Basemap(projection='merc', llcrnrlat=-60, urcrnrlat=80, \
            llcrnrlon=-180, urcrnrlon=180, \
            lat_ts=20, \
            resolution='c')
    
    x,y=m(lon, lat)
        
    m.fillcontinents(color='white', alpha =.1) # add continent color
    m.drawcoastlines(color='black', linewidth=0.2)  # add coastlines
    m.drawmeridians(np.arange(-180,180,30),color='lightgrey',alpha=.6,labels=[0,0,0,1], fontsize=10) ## add latitudes
    m.drawparallels(np.arange(-60,100,20),color='lightgrey',alpha=.6,labels=[1,0,0,0], fontsize=10) # add longitude
    
    marker = itertools.cycle(('X','o'))

    for ele,m in zip(['Feed','Food'],marker):
        ax.scatter(x, y, marker =m, label=ele, s=q[year][q.Element==ele]/500 ,alpha=0.2)
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.,fontsize=10)
        plt.title(year,y=1.055)

    return plt.show()
ax=[food_map(list(q.longitude),list(q.latitude),q,year) for year in ['Y1961','Y1971','Y1981','Y1991','Y2001','Y2011','Y2013']]

largest_feed=pd.DataFrame([df[df.Element=='Feed'].groupby(['Area'])[year].sum().nlargest(5) for year in Years])
largest_feed=largest_feed.fillna(0)
largest_feed.head()
plt.figure(figsize=(15,10))
plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow','purple','grey','orange','lightblue','violet','black'])
plt.plot(Years,largest_feed)
plt.xticks(Years, rotation=-300, fontsize=12)
plt.yticks(np.arange(0,710000,25000))
plt.xlabel('Years')
plt.ylabel('Production in 1000 tonnes')
plt.title('Countries Producing the Most Feed')
plt.legend(labels=largest_feed.columns, loc='best',fancybox=True,fontsize=14)
plt.show()
plt.figure(figsize=(10,10))
sns.set(font_scale=1.5)
cmap = sns.cubehelix_palette(25, start=.2, rot=-.5) ## colour maps
sns.heatmap(largest_feed,cmap=cmap,linecolor='w')
largest_food=pd.DataFrame([df[df.Element=='Food'].groupby(['Area'])[year].sum().nlargest(5) for year in Years])
largest_food=largest_food.fillna(0)
largest_food.head()
plt.figure(figsize=(18,12))
plt.rc('axes',prop_cycle=(cycler(color=['red', 'green', 'blue', 'pink','purple','maroon','orange','lightblue','violet','black'])))
plt.plot(Years,largest_food)
plt.xlabel('Years')
plt.ylabel('Production in 1000 tonnes')
plt.title('Countries Producing the Most Food')
plt.xticks(Years, rotation=-300,fontsize=12)
plt.yticks(np.arange(0,2500000,50000))
plt.legend(labels=largest_food.columns, loc='best',fancybox=True,fontsize=14)
plt.show()
plt.figure(figsize=(10,10))
sns.set(font_scale=1.25)
cmap = sns.cubehelix_palette(25, start=.2, rot=-.5) ## colour maps
sns.heatmap(largest_food,cmap=cmap,linecolor='w')
pd.DataFrame([df.groupby(['Item'])[year].sum().nlargest(10) for year in Years])
df_Item=pd.DataFrame([df.groupby(['Item'])[year].sum().nlargest(10) for year in Years])
df_Item=df_Item.fillna(0)
## cycle through colour and line styles
cycle=cycler('linestyle', ['-', '--', ':', '-.'])*cycler('color',['r', 'g', 'b', 'y', 'c', 'k'])
plt.figure(figsize=(20,10))
plt.rc('axes',prop_cycle=cycle)
plt.plot(Years,df_Item)
plt.xticks(rotation=300,fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Years')
plt.ylabel('Production in 1000 tonnes')
plt.title('Top 10 Food & Feed Items produced vs Years')
plt.legend(labels=df_Item.columns, loc='best',fancybox=True,fontsize=14)


plt.show()
df_Item_Feed=pd.DataFrame([df[df.Element=='Feed'].groupby(['Item'])[year].sum().nlargest(10) for year in Years])
df_Item_Feed=df_Item_Feed.fillna(0)
df_Item_Feed.head()
cycle=cycler('linestyle', ['-', '--', ':', '-.'])*cycler('color',['r', 'g', 'b', 'y', 'c', 'k'])
plt.figure(figsize=(20,15))
plt.rc('axes',prop_cycle=cycle)
plt.plot(Years,df_Item_Feed)
plt.xticks(rotation=300,fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Years')
plt.ylabel('Production in 1000 tonnes')
plt.title('Top 10 Feed Items produced vs Years')
plt.legend(labels=df_Item_Feed.columns, loc='best',fancybox=True,fontsize=14)

plt.show()
df_Item_Food=pd.DataFrame([df[df.Element=='Food'].groupby(['Item'])[year].sum().nlargest(10) for year in Years])
df_Item_Food=df_Item_Food.fillna(0)
cycle=cycler('linestyle', ['-', '--', ':', '-.'])*cycler('color',['r', 'g', 'b', 'y', 'c', 'k'])
plt.figure(figsize=(20,10))
plt.rc('axes',prop_cycle=cycle)
plt.plot(Years,df_Item_Food)
plt.xticks(rotation=300,fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Years')
plt.ylabel('Production in 1000 tonnes')
plt.title('Top 10 Food Items produced vs Years')
plt.legend(labels=df_Item_Food.columns, loc='best',fancybox=True,fontsize=14)

plt.show()
print('No of items produced over the year are at',len(df.Item.unique()))
#Attach a text label above each bar displaying its height
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%.2f' %float(height),
                ha='center', va='bottom')
## create function to plot the percentage increase of items between 2 years (Year1, Year2). Element is either 'Food' or 'Feed'

def percentage_increase_plot(Element,Year1,Year2):
    if Element=='Food':
        a=[df_Item_Food.loc[Year1,i] for i in df_Item_Food.loc[Year2].index]
        b=[i for i in df_Item_Food.loc[Year2]]
        Percent_Growth=pd.DataFrame([df_Item_Food.loc[Year2].index,[(y-x)/y*100 for x,y in zip(a,b)]]).transpose()
        Percent_Growth.columns=['Item','Percentage_increase']
        Percent_Growth=Percent_Growth[~Percent_Growth.isin([np.inf,-np.inf])] ## returns inf & -inf as NaN with isin & ~ returns the df that satisfies the condition
    elif Element=='Feed':
        a=[df_Item_Feed.loc[Year1,i] for i in df_Item_Feed.loc[Year2].index]
        b=[i for i in df_Item_Feed.loc[Year2]]
        Percent_Growth=pd.DataFrame([df_Item_Feed.loc[Year2].index,[(y-x)/y*100 for x,y in zip(a,b)]]).transpose()
        Percent_Growth.columns=['Item','Percentage_increase']
        Percent_Growth=Percent_Growth[~Percent_Growth.isin([np.inf,-np.inf])]
    
    Percent_Growth=Percent_Growth.fillna(0)
    ## drop rows wirh 0% increase
    Percent_Growth=Percent_Growth.drop(Percent_Growth[Percent_Growth.Percentage_increase==0].index)
    x=Percent_Growth.Item
    y=Percent_Growth.Percentage_increase
    
    plt.figure(figsize=(15,5))
    autolabel(plt.bar(x,y,color=['darkblue','crimson']))
    plt.title(" ".join([Element,'Percentage Increase from',Year1[1:],'to',Year2[1:]]))
    plt.xlabel(' '.join(['Top 10 Items in',Year2[1:]]))
    plt.ylabel(' '.join(['Percentage Increase from',Year1[1:]]))
    plt.xticks(rotation=330,fontsize=10)
    plt.yticks(fontsize=14)
    
for year in ['Y1961','Y1971','Y1981','Y1991','Y2001','Y2011','Y2012']:
    percentage_increase_plot('Food',year,'Y2013')
    
plt.show()
for year in ['Y1961','Y1971','Y1981','Y1991','Y2001','Y2011','Y2012']:
    percentage_increase_plot('Feed',year,'Y2013')
    plt.show()
