import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from matplotlib import cm
sns.set_style('whitegrid')
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/commodity_trade_statistics_data.csv')

df.rename({'country_or_area':'country'},axis =1, inplace=True)
df.country.replace({'China, Hong Kong SAR':'China'},inplace=True)
f,ax=plt.subplots(figsize=(14,7))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax)
ax.set_title('Missing Values Visualization',fontsize=16,color='black')
plt.show()
total_com = df[(df.commodity == 'ALL COMMODITIES') & (df.year>2000)]

total_com = total_com[total_com.country !='EU-28']

#Drop the weight and quantity columns
total_com.drop(['weight_kg','quantity'],axis=1,inplace=True)
total_com.reset_index(drop=True, inplace=True)

def total_commodity(flow_list):
    dataset = total_com.groupby(['country','flow']).sum().reset_index()
    dataset = dataset[dataset['flow'].isin(flow_list)]
    dataset.trade_usd = round(dataset.trade_usd/(10**9)).astype('int')
    data = []
    for i in flow_list:
        co = dataset[dataset.flow ==i].sort_values(by ='trade_usd',ascending=False)
        data.append(co.head(10))
    f, (ax1,ax2) = plt.subplots(1,2,figsize=(10,7))
    color1 = ['slategray','orange','slategray','slategray','slategray','slategray','slategray','slategray',
             'slategray','slategray']
    color2 = ['orange','slategray','slategray','slategray','slategray','slategray','slategray','slategray',
             'slategray','slategray']
    color=[color1,color2]
    axes = [ax1,ax2]
    for k in range(len(axes)):
        sns.barplot(x = 'trade_usd',y='country',data=data[k],palette=color[k],ax=axes[k],edgecolor=".2")
        axes[k].set_title('Top 10 '+flow_list[k]+'er'+' Since year 2000',fontsize=16)
        kwargs = {'length':3,'width':1,'color':'black','labelsize':14}
        axes[k].tick_params(**kwargs)
        x_axis = axes[k].axes.get_xaxis().set_visible(False)
    #sns.barplot(x = 'trade_usd',y='country',data=data[1],palette='RdBu_r',ax=ax2)
        sns.despine(bottom=True)
        f.subplots_adjust(wspace=0.22,right= 2,hspace=0.4)
        #f.subplots_adjust(wspace=0.22,right=2,hspace=0.4)
    return plt.show()

total_commodity(['Export','Import'])
e = total_com.groupby(['year','flow']).sum().reset_index()#.plot(kind='scatter')
flow_list = ['Export','Import']
e = e[e['flow'].isin(flow_list)]
e.trade_usd = round(e.trade_usd/(10**12)).astype('int')
#plt.legend(loc='best',)
f, ax = plt.subplots(figsize = (14,7))
for i in range(len(flow_list)):
    e[e.flow==flow_list[i]].plot(kind = 'line',x='year',y='trade_usd',
                             marker='o',label = flow_list[i],ax =ax)

ax.axvline(x=2009, color='darkred',linestyle= '-',linewidth=3)
kwargs = {'fontsize':13,'color':'black'}
plt.xlabel('Year',**kwargs)
plt.ylabel('Amounts ($T)',**kwargs)
kg = {'length':3,'width':1,'color':'black','labelsize':12}
ax.tick_params(**kg)
plt.title('World Total Imports & Exports Commodities Since Year 2000',fontsize=14)
plt.legend(loc='best',fontsize=12)
plt.show()
def imp_exp(input):
    data = total_com.groupby(['country','year','flow']).sum().reset_index()#.plot(kind='scatter')
    flow_list = ['Export','Import']
    ctry_list_imp = ['USA', 'China','Germany','Japan','United Kingdom']
    ctry_list_exp = ['USA', 'China','Germany','Japan','France']
    ctry_type = [ctry_list_exp,ctry_list_imp]

    ctry_select = ctry_type[input]
    data = data[data['country'].isin(ctry_select)]

    data.trade_usd = round((data.trade_usd/(10**12)),2).astype('float')
    #plt.legend(loc='best',)
    f, ax = plt.subplots(figsize = (14,7))
    for i in range(len(ctry_list_imp)):
        data[(data.country==ctry_select[i]) & (data.flow==flow_list[input])].plot(kind = 'line',
                                x='year',y='trade_usd',marker='o',label = ctry_select[i],ax =ax)

    ax.axvline(x=2009, color='darkred',linestyle= '-',linewidth=3)
    kwargs = {'fontsize':13,'color':'black'}
    plt.xlabel('Year',**kwargs)
    plt.ylabel('Amounts in $US(Trillions)',**kwargs)
    ax.set_yticklabels(['0','0.5T','1T','1.5T','2T','2.5T','3.5T'])
    kg = {'length':3,'width':1,'color':'black','labelsize':12}
    ax.tick_params(**kg)
    plt.title('Top 5 Countries '+ flow_list[input]+'ers'+' of Commodities Since Year 2000',fontsize=14)
    plt.legend(loc='best',fontsize=12,ncol=3)
    return plt.show()

imp_exp(0)
imp_exp(1)
def clean_data(data,input,countryname):
    df_clean_usa = data[data.country==countryname]
    df_clean_usa = df_clean_usa[(df_clean_usa.commodity != 'ALL COMMODITIES') & (df_clean_usa.flow.isin(input))&
                        (df_clean_usa.commodity != 'Commodities not specified according to kind') &
                               (df_clean_usa.year > 2000)]
    df_clean_usa.reset_index(inplace=True,drop=True)
    df_clean_usa.commodity = [df_clean_usa.commodity[i].split(',')[0] for i in range(len(df_clean_usa.commodity))]
    return df_clean_usa

def usa_max_im_ex(data,input,input1):
    
    df_usa_1 = clean_data(data,input,'USA').groupby(['year','commodity','flow']).sum()['trade_usd'].reset_index()

    df_usa_1.trade_usd = round((df_usa_1.trade_usd/(10**9)),2).astype('float')
    df_usa_1 = df_usa_1[df_usa_1.flow==input1]
    df_usa_1.year = df_usa_1.year.astype('int')
    max_trade_yr = []
    for i in df_usa_1.year.unique():
        max_trade_yr.append(df_usa_1[df_usa_1.year==i].trade_usd.max())

    df_usa_1_max = df_usa_1[df_usa_1.trade_usd.isin(max_trade_yr)]

    #g = sns.pointplot(x="year", y="trade_usd", hue="commodity",size=6, data=df_usa_1_max,fit_reg=False,
                   #aspect=2.5,scatter_kws={'s':200,'edgecolor':'black'},legend_out=False)
    #g = (g.set_axis_labels('Year','Trade_usd'))
    #ax = plt.gca()
    f,ax =plt.subplots(figsize=(15,7))
    sns.barplot(x="year", y="trade_usd", hue="commodity", data=df_usa_1_max,ax=ax,edgecolor=".2")
                 #markers=["o", "x",'*'],size=6,linestyles=["-", "--",'--'])
    ax.set_title('USA: Yearly Most '+ input1 +'ed'+' Commodities since Year 2000',fontsize=16,color='black')
    ax.set_xlabel('Year',fontsize=12)
    ax.set_ylabel('Trade_usd (Trillions US Dollars)',fontsize=12)
    kg = {'length':3,'width':1,'color':'black','labelsize':12}
    ax.tick_params(**kg)
    ax.legend(loc='best',ncol=2,fontsize=12)
    return plt.show()

usa_max_im_ex(df,['Export','Import'],'Export')
usa_max_im_ex(df,['Export','Import'],'Import')
usa_com = clean_data(df,['Import','Export'],'USA')

#Clean the Category Column
#split each row at 'and',pick the first elt in the list...list the first elt again at'_' and everything except first elt
a = [usa_com.category[i].split('and')[0].split('_')[1:] for i in range(len(usa_com.category))]
#Join the first 3 elts from each row 
a=[' '.join(a[i][:4]) for i in range(len(a))]
usa_com.category = a

def cat_year(data,input,input1):
    cat = data.groupby(['year','category','flow']).sum()['trade_usd'].reset_index()
    cat_2008 = cat[(cat.year==input)& (cat.flow==input1)].sort_values('trade_usd',ascending=False)
    cat_2008.trade_usd = round((cat_2008.trade_usd/(10**9)),2).astype('float')
    cat_2008.category.replace({'arms ':'arms-Ammunitions parts'},inplace=True)
    return cat_2008

def trade_diff (input):   
    d1 = cat_year(usa_com,2008,input)
    d2 = cat_year(usa_com,2009,input)

    d_merge = d1.merge(d2,on='category')
    d_merge['trade_usd_diff'] = ((d_merge.trade_usd_y - d_merge.trade_usd_x)/d_merge.trade_usd_x )*100
    d_com = d_merge[d_merge.trade_usd_diff > 0]
    d_com = d_com.sort_values('trade_usd_diff',ascending=False)
    #d = trade_diff('Export')
    d_com.rename({'trade_usd_x':'trade_usd_2008','trade_usd_y':'trade_usd_2009'},axis=1,inplace=True)
    # Graph
    f,ax=plt.subplots(figsize=(15,5))
    color=['slategray','green']
    ax = d_com[['trade_usd_2008','trade_usd_2009']].plot(kind='bar',ax=ax,color=color,ec='black')
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x()+.04, i.get_height()+0.06,
                str(round((i.get_height()), 2))+' B', fontsize=11, color='black')
    ax.set_xticklabels(d_com.category.values.tolist(),rotation=0)
    kwargs = {'length':3,'width':1,'color':'black','labelsize':12}
    ax.tick_params(**kwargs)
    ax.set_ylabel('Trade($Billions)',fontsize=14)
    ax.set_xlabel('Category',fontsize=14)
    ax.legend(loc='upper left',ncol=1,fontsize=12)
    ax.set_title(input +' Categories Trade Increased During The Great Depression',fontsize=14)
    sns.despine()
    return plt.show()

trade_diff('Export')
trade_diff('Import')