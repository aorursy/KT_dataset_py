
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
pop_df = pd.read_csv('../input/population-by-country-2020/population_by_country_2020.csv',index_col=0,usecols=[0,1])


    
#########Prepare data##########
df = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')
df['Country/Region'].replace(to_replace=r'^US$', value='United States', regex=True,inplace=True)
df=df[['Country/Region','Date','Deaths']]
df['Date'] = pd.to_datetime(df['Date'])

df= df.groupby(["Country/Region",'Date'],as_index=False)['Deaths'].sum()
UE=["Austria",
"Belgium",
"Bulgaria",
"Croatia",
"Cyprus",
"Czech Republic",
"Denmark",
"Estonia",
"Finland",
"France",
"Germany",
"Greece",
"Hungary",
"Ireland",
"Italy",
"Latvia",
"Lithuania",
"Luxembourg",
"Malta",
"Netherlands",
"Poland",
"Portugal",
"Romania",
"Slovak Republic",
"Slovenia",
"Spain",
"Sweden",
"United Kingdom"]
#print(df[df["Country/Region"]=="France"])
#df[df["Country/Region"]=="France"]['Deaths']=df[df["Country/Region"] .isin(UE)].groupby(['Date'])['Deaths'].sum()
#print(df[df["Country/Region"]=="France"])
### compute the top 10
latest_g=df['Date'].max()
#latest=pd.to_datetime("2020-04-27")
df=df[df['Date']<=latest_g]

latest_d2=df[df['Date']==latest_g- pd.Timedelta('1 day')].reset_index()

latest_d=df[df['Date']==latest_g].reset_index()
line_styles=['-','--',':','-.-']

#latest_d["Deaths"]=latest_d["Deaths"]-latest_d2["Deaths"]
top_d=latest_d.sort_values(by='Deaths',ascending=False)[0:20]['Country/Region']
#print(top_d)
#top_d=["France","United Kingdom","Italy","Spain","Brazil","Canada","Germany","Belgium","United States"]
#plot dataset
def plot_UE(window_size,top_d,pop=None,field="Deaths",dataset=None,diff=False,latest=None,style="-"):
    if dataset==None:
        dataset=df
    if latest==None:
        latest=latest_g
    i=0
    ax={}
    
    c="UE" 
    c_df=dataset[dataset["Country/Region"].isin(UE)].groupby(['Date'])[field].sum()
    c_df=c_df.diff()

    c_df=c_df.rolling(window=window_size,min_periods=1,center=True, win_type='triang').mean()
    if diff:
        c_df=c_df.diff()
        c_df=c_df.rolling(window=window_size,min_periods=1,center=True, win_type='triang').mean()

    if pop!=None:
        c_df=c_df/pop[i]
    lw=1
    i=i+1
    if latest==None:
        c_df=c_df[c_df.index<=latest]

    ax=c_df.plot(label=c,legend=True, linewidth=lw,title="Numero de mortes por dia (window_size="+str(window_size)+")",figsize=(8,7),style=style)
    ax.set_ylabel("Numero de mortes por dia")
    #if latest!=None:
    #ax.set_xlim(right=(latest+ pd.Timedelta('0 days')))
    ax.set_xlim(right= pd.to_datetime("2020-04-27")+ pd.Timedelta('30 days'))
    #ax.set_ylim(top= 1000)

    ax.grid(b=True)
    return ax
def plot_dataset(window_size,top_d,pop=None,field="Deaths",dataset=None,diff=False,diff2=True,latest=None,style=None,title=None):
    if dataset==None:
        dataset=df
    if latest==None:
        latest=latest_g
    i=0
    ax={}
    if(title==None):
        title="Numero de mortes por dia (window_size="+str(window_size)+")"
    for c in top_d:
        c_df=dataset[dataset["Country/Region"]==c].groupby(['Date'])[field].sum()
        if diff2:
            c_df=c_df.diff()
       
        c_df=c_df.rolling(window=window_size,min_periods=1,center=True, win_type='triang').mean()
        if diff:
            c_df=c_df.diff()
            c_df=c_df.rolling(window=window_size,min_periods=1,center=True, win_type='triang').mean()

        if pop!=None:
            c_df=c_df/pop_df.loc[c][0]
            #c_df=c_df/pop[i]
        lw=3
       
        if style==None:
            _style=line_styles[i//10]
        else:
            _style=style
        i=i+1
        if latest!=None:
            c_df=c_df[c_df.index<=latest]
        if c_df.empty:
            return
        if c== "Brazil":
            lw=9
            ax=c_df.plot(label=c,legend=True, linewidth=lw,figsize=(2*8,2*7),style=_style,title=title)
           
        else:    
            ax=c_df.plot(label=c,legend=True, linewidth=lw,figsize=(2*8,2*7),style=_style,title=title)
        ax.set_ylabel(title)
        #if latest!=None:
        #ax.set_xlim(right=(latest+ pd.Timedelta('0 days')))
        #ax.set_xlim(right= pd.to_datetime("2020-04-27")+ pd.Timedelta('30 days'))
        #ax.set_ylim(top= 1000)
        patches, labels = ax.get_legend_handles_labels()

        ax.legend(patches, labels, loc=2)
        ax.grid(b=True)
    return ax
#top_d=["France"]
#ax=plot_dataset(window_size=14,top_d=top_d,latest=pd.to_datetime("2020-04-27")+ pd.Timedelta('0 days'))
#ax=plot_dataset(window_size=14,top_d=top_d,latest=pd.to_datetime("2020-04-27"))

#ax.set_ylim(top=4000)
#top_d=["France"]
#ax=plot_dataset(window_size=14,top_d=top_d,latest=pd.to_datetime("2020-04-27")+ pd.Timedelta('0 days'))
ax=plot_dataset(window_size=14,top_d=top_d)
#ax=plot_UE(window_size=14,top_d=top_d,style="--")

#plot_UE
#ax.set_ylim(top=4000)


ax=plot_dataset(window_size=14,top_d=top_d,pop=1,title="Numero de mortes por dia per capita")

ax=plot_dataset(window_size=14,top_d=top_d,diff2=False,pop=1,title="Numero de mortes por dia per capita")

ax=plot_dataset(window_size=14,top_d=top_d,diff=True,latest=pd.to_datetime("2020-05-08"),title="Variaçao do numero de mortes por dia")
#ax=plot_UE(window_size=14,top_d=top_d,diff=True,latest=pd.to_datetime("2020-05-08"))



ax=plot_dataset(window_size=14,top_d=top_d,diff=True,pop=1,title="Variaçao do numero de mortes por dia per capita")
#ax=plot_UE(window_size=14,top_d=top_d,diff=True,latest=pd.to_datetime("2020-05-08"))

south_a=["Brazil","Colombia","Argentina","Peru","Venezuela","Chile","Ecuador","Bolivia","Paraguay","Uruguay","Guyana","Suriname"]
south_a_pop=[212322269,50888825,45119153,32933515,28222237,19132249,17608629,11642688,7116303,3471726,785787,  585714]
ax=plot_dataset(window_size=14,top_d=south_a,title="Numero de mortes por dia América do Sul")

south_a=["Brazil","Colombia","Argentina","Peru","Venezuela","Chile","Ecuador","Bolivia","Paraguay","Uruguay"]
#south_a_pop=[212322269,50888825,45119153,32933515,28222237,19132249,17608629,11642688,7116303,3471726]
ax=plot_dataset(window_size=14,top_d=south_a,pop=1,title="Numero de mortes por dia per capita America do Sul")

#df=df[df["Date"]>pd.to_datetime("2020-03-20")]

ax=plot_dataset(window_size=14,top_d=top_d,diff=True,pop=1,title="Variaçao do numero de mortes por dia per capita Top 10")
#ax=plot_UE(window_size=14,top_d=top_d,diff=True,latest=pd.to_datetime("2020-05-08"))

"""
df = pd.read_csv('https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv',
                 parse_dates=['date'],low_memory=False)


df=df[["country_region","date","sub_region_1","retail_and_recreation_percent_change_from_baseline"]].rename(columns={'date': 'Date','retail_and_recreation_percent_change_from_baseline':'mobility',"country_region":"Country/Region"})

df=df[df["sub_region_1"].isnull() ==True  ]
#df=df[df["Date"]>pd.to_datetime("2020-03-20")]
#df2=df[df['Country/Region']=='Brazil'][["Date","mobility"]].set_index("Date")

df["mobility"]=1-df["mobility"]
#ax=plot_dataset(window_size=14,top_d=np.concatenate([top_d,south_a]),field="mobility",diff=False,diff2=False)
ax=plot_dataset(window_size=14,top_d=top_d,field="mobility",diff=False,diff2=False,title="Adesao ao isolamento social Top 10")

#ax.set_ylabel("Adesao ao isolamento social")
#ax.set_title("Adesao ao isolamento social")

#ax.get_legend().remove()""" 
#ax=plot_dataset(window_size=14,top_d=south_a,field="mobility",diff=False,diff2=False,title="Adesao ao isolamento social America do Sul")