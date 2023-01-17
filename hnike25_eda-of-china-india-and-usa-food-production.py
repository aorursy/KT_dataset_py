import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from matplotlib import cm
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/FAO.csv', encoding='ISO-8859-1')
#Pivoting some columns
columns = ['Area Abbreviation', 'Area Code', 'Area', 'Item Code', 'Item','Element Code', 
           'Element', 'Unit', 'latitude', 'longitude']
df = pd.melt(df,id_vars= columns)

# Dropping some columns
df.drop(columns=['Area Code','Item Code','Area Abbreviation','Unit','Element Code'], axis=1,inplace=True)
# Renaming some columns
df.rename(str.lower, axis = 1, inplace = True)
df.rename({'variable':'year','value':'quantity','area':'country'},axis=1,inplace=True)

# Removing the Y from the numbers in df.year
df.year = df.year.str.replace('Y','')
df.country = df.country.replace ({'China, mainland': 'China','United States of America':'USA',
                                 'United Kingdom':'UK'})
df.head(2)
df_isnull = pd.DataFrame(round((df.isnull().sum().sort_values(ascending=False)/df.shape[0])*100,1)).reset_index()
df_isnull.columns = ['Columns', '% of Missing Data']
df_isnull.style.format({'% of Missing Data': lambda x:'{:.1%}'.format(abs(x))})
cm = sns.light_palette("skyblue", as_cmap=True)
df_isnull = df_isnull.style.background_gradient(cmap=cm)
print('Only the quantity column has a 10.3% missing data')
df_isnull
plt.figure(figsize=(12,5))
labels = df.element.value_counts().index
sizes = df.element.value_counts().values
explode = (0.1,0)
plt.pie(sizes, explode=explode,labels=labels,autopct='%1.1f%%', startangle=45,
        wedgeprops = {'linewidth': 1,},textprops ={'color':'white','fontsize':15,'weight':'bold'})
plt.legend(labels,loc='best',ncol=2,fontsize=12,title ='Element Type')
plt.axis('equal')
plt.title('Element Type Distribution',color = 'black',fontsize =15)
plt.show()
df.dropna(inplace=True)
df_feed = df[df.element == 'Feed']
df_food = df[df.element == 'Food']
def top_five_year (data,text,a,b):
    d3 = pd.DataFrame(columns = ['year','country','quantity','rank'])
    d1 = data.groupby(['year','country'])['quantity'].sum().reset_index()
    for i in range(1961, 2014):
        d2 = d1[d1.year ==str(i)]
        d2['rank'] = d2['quantity'].rank(ascending = False,method ='max')
        d2['rank'] = d2['rank'].astype(int)
        d2 = d2[d2['rank'].isin([1,2,3,4,5])]
        d4 = pd.concat([d3,d2])
        d3 = d4
    d3.reset_index(drop=True,inplace=True)
    d3.sort_index(by='rank')
# Creating a plot 
    f,ax = plt.subplots(figsize=(14,5))
    title_list = []
    for i in d3.country.value_counts().index[a:b]:
        d3[d3.country == i].plot(kind='line',x='year',y='rank',ax=ax,label= i,marker='o')
        title_list.append(i)

    ax.set_xticklabels(['0','1961','1971','1981','1991','2001','2011'])
    ax.legend(loc='best',fontsize=12,title ='Country',ncol=3)
    ax.set_title (title_list[0]+ ' & ' + title_list[1]+ ' & '+ title_list[2]+ ': ' + text +' Rank since 1961',
                  color='black',fontsize=16)
    sns.despine()
    plt.tight_layout()
    return plt.show()
top_five_year(df_food,'Food Production',0,3)
top_five_year(df_feed,'Food Consumption',0,3)
def pct_change (start_pt,current_pt):
    numb = (float(current_pt - start_pt)/start_pt)*100.00
    numb = round(numb,1)
    return numb

def pct_top_cnty (data):
    # To extract the top 5 over 50 years
    ds = data.groupby('country')['quantity'].sum().reset_index().sort_values(by='quantity',ascending=False)
    # Reduce the data set to the top 5 
    ff = data.groupby(['year','country'])['quantity'].sum().reset_index()
    ff2 = ff[ff.country.isin(ds.head().country.values)]

    #Calculate the running percentage 
    d3 = pd.DataFrame(columns = ['year','country','quantity','pct'])
    for i in ds.head().country.values:
        d2 = ff2[ff2.country==i].reset_index()
        pct = [pct_change(d2.quantity[0],qty) for qty in d2.quantity]
        d2['pct'] = pct
        d4 = pd.concat([d3,d2])
        d3 = d4
    return d3

def pct_top_cnty_plot(data,text,a,b):
    f,ax = plt.subplots(figsize=(16,7))
    ds = pct_top_cnty(data).groupby('country')['quantity'].sum().reset_index().sort_values(by='quantity',ascending=False)
    title_list = []
    for i in ds.head().country.values[a:b]:
        pct_top_cnty(data)[pct_top_cnty(data).country == i].plot(kind='line',x='year',y='pct',ax=ax,label= i,marker='o')
        title_list.append(i)
    ax.set_xticklabels(['0','1961','1971','1981','1991','2001','2011'])
    ax.legend(loc='best',fontsize=12,title ='Country',ncol=3)
    ax.set_title (title_list[0]+ ' & ' + title_list[1]+ ': ' + text +' Growth Rate since 1961',
                  color='black',fontsize=16)
    sns.despine()
    return plt.show()
pct_top_cnty_plot(df_food,'Food Production',0,3)
pct_top_cnty_plot(df_feed,'Feed Consumption',0,3)
def top_food_feed_cty (data1,country,data2,n_col):

    d1 = data1[data1.country == country].groupby('item')['quantity'].sum().reset_index().sort_values(by='quantity',ascending=False)
    d2 = data2[data2.country == country].groupby('item')['quantity'].sum().reset_index().sort_values(by='quantity',ascending=False)
    
    f,(ax1,ax2) = plt.subplots(2,1,figsize=(10,10))
    sns.barplot(x = 'quantity', y = 'item', data= d1.head(n_col),orient='h',
                    palette=['seagreen'],ax=ax1)
    sns.barplot(x = 'quantity', y = 'item', data= d2.head(n_col),orient='h',
                    palette=['orange'],ax=ax2)
    axes = [ax1,ax2]
    d = [d1,d2]
    for j in range(len(axes)):
        kwargs= {'length':3, 'width':1, 'colors':'black','labelsize':12,}
        axes[j].tick_params(**kwargs)
        x_axis = axes[j].axes.get_xaxis().set_visible(False)
        text = ['Production','Consumption']
        axes[j].set_title (country+':' + ' Top 10 Items ' + text[j],color='black',fontsize=16)
        f.subplots_adjust(wspace=0.3,hspace=0.2)
        sns.despine(left=False,bottom=True)
    return plt.show()
top_food_feed_cty(df_food,'China',df_feed,10)
top_food_feed_cty(df_food,'USA',df_feed,10)
top_food_feed_cty(df_food,'India',df_feed,10)
def one_item (data,item_name,n_row,color):
    dd = data[data.item==item_name]
    dd2 = dd.groupby('country')['quantity'].sum().reset_index().sort_values(by='quantity',ascending=False)
    
    f, ax = plt.subplots(figsize=(10,5)) 
    sns.barplot(x = 'quantity', y = 'country', data= dd2.head(n_row),orient='h',
                        color=color,ax=ax)
    
    kwargs= {'length':3, 'width':1, 'colors':'black','labelsize':'large'}
    ax.tick_params(**kwargs)
    x_axis = ax.axes.get_xaxis().set_visible(False)
    ax.set_title ('Top 10 '+ item_name + ' Producer',color='black',fontsize=16)
    sns.despine(bottom=True)
    
    return plt.show()
one_item(df_food,'Beer',10,'GoldenRod')
one_item(df_food,'Meat',10,'darkred')
one_item(df_food,'Coffee and products',10,'tan')

