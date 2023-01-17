%%capture



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

import pandas_profiling # library for automatic EDA

%pip install autoviz # installing and importing autoviz, another library for automatic data visualization

from autoviz.AutoViz_Class import AutoViz_Class





file_path='../input/netflix-shows/netflix_titles.csv'



NF_data=pd.read_csv(file_path)



from IPython.display import display, HTML

pd.options.display.max_columns = None

display(NF_data.head())

%%capture

print(list(NF_data.columns),'\n')



for k in range(1,len(list(NF_data.columns))):

    print(NF_data[list(NF_data.columns)[k]].unique())
%%capture

report = pandas_profiling.ProfileReport(NF_data, title="Netflix dataset", html={'style': {'full_width': True}}, sort="None")



report.to_notebook_iframe()

display()
%%capture

AV = AutoViz_Class()





report_2 = AV.AutoViz("../input/netflix-shows/netflix_titles.csv")


#Let's clean up the data as we said before.

#First, let's replace NaN directors with "Unknown", which makes more sense:



NF_data['director'] = NF_data['director'].fillna('Unknown')

NF_data['date_added'] = NF_data['date_added'].fillna('January 1, 1900')

NF_data['country'] = NF_data['country'].fillna('Unknown')

NF_data['cast'] = NF_data['cast'].fillna('Unknown')

NF_data['rating'] = NF_data['rating'].fillna('Unknown')



#next, let's create a new column with the new formatted date

#and three more columns with day, month, year



monthN2N={"January":"1","February":"2","March":"3","April":"4","May":"5","June":"6","July":"7","August":"8","September":"9","October":"10","November":"11","December":"12"}



NF_data[['month_added','day_added','year_added']]=NF_data.date_added.str.split(expand=True,)

NF_data['day_added'] = NF_data['day_added'].map(lambda x: x.rstrip(','))

NF_data=NF_data.replace({"month_added":monthN2N})

NF_data['date_added']=NF_data['day_added']+'-'+NF_data['month_added']+'-'+NF_data['year_added']

NF_data['year_added']=NF_data['year_added'].astype(int)

display(NF_data.head())
%%capture

#setting the style for the sns plotting

sns.set_style("whitegrid")

sns.set_palette('Reds')



#getting the data to plot

items=NF_data.groupby('type').count()['show_id']

items.reset_index()



#setting easier names for kwargs

explode = (0, 0.1)

labels = 'Movies','TV Shows'



#barplot

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,8))

sns.barplot(items.index,items.values,palette=sns.color_palette('Reds',n_colors=2),ax = axes[0])

for bar in axes[0].patches:

    x = bar.get_x()

    width = bar.get_width()

    centre = x + width/2.

    bar.set_x(centre - 0.25/2)

    bar.set_width(0.25)

    

    

#pie chart 

axes[0].set_title('Bar chart of available items on netflix')

sns.despine()

sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1.5})

axes[1].pie(items.values, shadow=True, autopct='%1.f%%', explode=explode,colors=sns.color_palette('Reds',n_colors=2),wedgeprops = {'linewidth': 0})

axes[1].legend(loc='best', labels=labels, fontsize='medium')

axes[1].set_title('Pie chart of available items on netflix')

display()
# %%capture

#getting the data to plot for movies



movies=NF_data.loc[NF_data['type']=='Movie']

movies[['duration','b']] = movies['duration'].str.split(n=1, expand=True)

movies=movies.drop(columns=['b'])



#getting the data to plot for shows

shows=NF_data.loc[NF_data['type']=='TV Show']



shows[['duration','b']] = shows['duration'].str.split(n=1, expand=True)

shows['duration']=shows['duration'].astype(int)

shows2=shows

shows=shows.groupby('duration').count()

display(shows2.head())


import numpy as np

import matplotlib.pylab as pl

import matplotlib.gridspec as gridspec

# Cut the window in 2 parts





def show_values_on_bars(axs):

    def _show_on_single_plot(ax):        

        for p in ax.patches:

            _x = p.get_x() + p.get_width() / 2

            _y = p.get_y() + p.get_height()

            value = int(p.get_height())

            ax.text(_x, _y+50, value, ha="center",rotation=90) 



    if isinstance(axs, np.ndarray):

        for idx, ax in np.ndenumerate(axs):

            _show_on_single_plot(ax)

    else:

        _show_on_single_plot(axs)



gs = gridspec.GridSpec(2, 2,height_ratios=[0.15,0.85])



pl.figure(figsize=(16,8))

axe = pl.subplot(gs[0, 0]) # row 0, col 0

sns.boxplot(x=movies.duration.astype(int), data=movies, orient='h',color='red')

axe.set_xlabel('')

axe.set_title('Distribution of movie duration on netflix')



axe = pl.subplot(gs[1, 0]) # row 0, col 1

sns.distplot(movies["duration"], ax=axe,color='r')

axe.set_ylim([0,0.025])



axe.set_xlabel('duration of movie')



axe = pl.subplot(gs[:, 1])

sns.barplot(shows.index,shows.b,ax=axe,palette=sns.color_palette('Reds_r',n_colors=10))

show_values_on_bars(axe)

axe.set_ylim([0,1500])

axe.set_title('Amount of netflix shows relative to seasons')

axe.set_xlabel('seasons')

display()


movies['duration']=movies.duration.astype(int)

shortmovies=movies.sort_values('duration').head()

longmovies=movies.sort_values('duration',ascending=False).head(10)



plt.figure(figsize=(15,5))

ax1=plt.subplot(121)

sns.barplot(longmovies.title,longmovies.duration,palette=sns.color_palette('Reds_r',n_colors=10))

plt.setp( ax1.xaxis.get_majorticklabels(), rotation=-45,ha="left", rotation_mode="anchor" ) 

ax1.set_title('longest movies ranked')

ax1.set_xlabel("")

ax1.set_ylabel("duration (mins)")



ax2=plt.subplot(122)

sns.barplot(shortmovies.title,shortmovies.duration,palette=sns.color_palette('Reds',n_colors=10))

plt.setp( ax2.xaxis.get_majorticklabels(), rotation=-45,ha="left", rotation_mode="anchor" ) 

ax2.set_title('shortest movies ranked')

ax2.set_xlabel("")

ax2.set_ylabel("duration (mins)")



display()
shows2['duration']=shows2.duration.astype(int)

shortshows=shows2.sort_values('duration').head()

longshows=shows2.sort_values('duration',ascending=False).head(10)



plt.figure(figsize=(15,5))

ax1=plt.subplot(121)

sns.barplot(longshows.title,longshows.duration,palette=sns.color_palette('Blues_r',n_colors=10))

plt.setp( ax1.xaxis.get_majorticklabels(), rotation=-45,ha="left", rotation_mode="anchor" ) 

ax1.set_title('longest shows ranked')

ax1.set_xlabel("")

ax1.set_ylabel("duration (seasons)")





ax2=plt.subplot(122)

sns.barplot(shortshows.title,shortshows.duration,palette=sns.color_palette('Blues',n_colors=10))

plt.setp( ax2.xaxis.get_majorticklabels(), rotation=-45,ha="left", rotation_mode="anchor" )

ax2.set_title('longest shows ranked')

ax2.set_xlabel("")

ax2.set_ylabel("duration (seasons)")



display()
#Let's get the data for number of movies (resp.shows) /released/ by year (as in in theaters)



released_data=NF_data.groupby('release_year').count().astype(int)

released_data=released_data[['show_id','type']]



movies_released=NF_data.loc[NF_data['type']=='Movie']

movies_released=movies_released.groupby('release_year').count()



shows_released=NF_data.loc[NF_data['type']=='TV Show']

shows_released=shows_released.groupby('release_year').count()



released_data['total released']=released_data.show_id

released_data['movies released']=movies_released.show_id

released_data['movies released'] = released_data['movies released'].fillna(0)

released_data['movies released']=released_data['movies released'].astype(int)

released_data['shows released']=shows_released.show_id

released_data['shows released'] = released_data['shows released'].fillna(0)

released_data['shows released']=released_data['shows released'].astype(int)

released_data=released_data.drop(columns=['show_id','type'])

released_data['year']=released_data.index



display(released_data.tail())
%%capture

#Let's get the data for number of movies (resp.shows) /added/ by year (as in on netflix)



added_data=NF_data.groupby('year_added').count().astype(int)



added_data=added_data[['show_id','type']]



movies_added=NF_data.loc[NF_data['type']=='Movie']

movies_added=movies_added.groupby('year_added').count()



shows_added=NF_data.loc[NF_data['type']=='TV Show']

shows_added=shows_added.groupby('year_added').count()



added_data['total added']=added_data.show_id

added_data['movies added']=movies_added.show_id

added_data['movies added'] = added_data['movies added'].fillna(0)

added_data['movies added']=added_data['movies added'].astype(int)

added_data['shows added']=shows_added.show_id

added_data['shows added'] = added_data['shows added'].fillna(0)

added_data['shows added']=added_data['shows added'].astype(int)

added_data=added_data.drop(columns=['show_id','type'])

added_data['year']=added_data.index



display(added_data.tail())


fig=plt.figure(figsize=(14,4))

axe=plt.subplot(121)

sns.lineplot(x='year', y='value', hue='variable', 

             data=pd.melt(released_data, ['year']),palette=sns.color_palette('Reds_r',n_colors=3),ax=axe)

axe.set_ylabel("amount of released items")

axe.set_xlim([1925,2019])

axe.set_title("Amount of items released by year separated by type")

axe=plt.subplot(122)

sns.lineplot(x='year', y='value', hue='variable', 

             data=pd.melt(added_data, ['year']),palette=sns.color_palette('Blues_d',n_colors=3),ax=axe)

axe.set_xlim([2008,2019])

axe.set_ylabel("amount of released items")

axe.set_title("Amount of items added to NF by year separated by type")

display()
Ratings_data=NF_data.groupby(['rating','type']).count().sort_values(['title','rating'],ascending=False)

Ratings_data = Ratings_data.reset_index(level=[0,1])

R_d_m=Ratings_data.loc[Ratings_data['type']=='Movie']

R_d_s=Ratings_data.loc[Ratings_data['type']=='TV Show']



popular_ratings=NF_data.groupby('rating').count().sort_values('title')
gs = gridspec.GridSpec(3, 2,height_ratios=[0.2,0.3,0.3])



pl.figure(figsize=(16,16))



axe = pl.subplot(gs[0, :]) # row 0, col 0

sns.barplot(x=popular_ratings.index,y=popular_ratings.title,ax=axe,palette='Reds')

axe.set_title('Overall popularity for ratings')

axe.set_ylabel('amount of tags on items')



axe = pl.subplot(gs[1:, 0]) # row 0, col 0

sns.barplot(x=Ratings_data.title,y=Ratings_data.rating,hue=Ratings_data.type,palette=sns.color_palette(['red','blue']),ax=axe)

display()

axe.set_title('Popularity for ratings separated by type')

axe.set_xlabel('amount of tags on items')



axe = pl.subplot(gs[1, 1]) # row 0, col 0

sns.barplot(y=R_d_m.title,x=R_d_m.rating,palette=sns.color_palette('Reds_r'),ax=axe)

plt.setp( axe.xaxis.get_majorticklabels(), rotation=-45,ha="left", rotation_mode="anchor" )

display()

axe.set_title('Popularity, movie ratings')

axe.set_ylabel('amount of tags on items')



axe = pl.subplot(gs[2, 1]) # row 0, col 0

sns.barplot(y=R_d_s.title,x=R_d_s.rating,palette=sns.color_palette('Blues_r'),ax=axe)

plt.setp( axe.xaxis.get_majorticklabels(), rotation=-45,ha="left", rotation_mode="anchor" )

axe.set_title('Popularity, TV show ratings')

axe.set_ylabel('amount of tags on items')



display()
tags_split=pd.concat([pd.Series(row['show_id'], row['listed_in'].split(','))              

                    for _, row in NF_data.iterrows()]).reset_index()

tags_split=tags_split.rename(columns={tags_split.columns[1]: "show_id"})



tags_split['index']=tags_split['index'].str.strip()





clean_data= NF_data.drop(columns=['listed_in'])



tags_data=clean_data.set_index('show_id').join(tags_split.set_index('show_id')).rename(columns={'index': "tag"})





poptags=tags_data.groupby('tag').count().sort_values('title',ascending=False)

p_t_g=tags_data.groupby(['tag','type']).count().reset_index(level=[0,1])

p_t_m=p_t_g.loc[p_t_g['type']=='Movie'].sort_values('title',ascending=False)

p_t_s=p_t_g.loc[p_t_g['type']=='TV Show'].sort_values('title',ascending=False)





fig=plt.figure(figsize=(16,16))

ax=plt.subplot(111)

sns.barplot(y=poptags.index,x=poptags.title,palette='Reds_r')

ax.set_title('popularity of tags according to number of shows')

ax.set_xlabel('Amount of movies tagged')

ax.set_ylabel('')

display()
plt.figure(figsize=(12,10))

ax1=plt.subplot(211)

sns.barplot(y=p_t_m.head(20).tag,x=p_t_m.title,palette='Reds_r')

ax1.set_title('popularity of tags according to number of shows (Movies)')

ax1.set_xlabel('Amount of movies tagged')

ax1.set_ylabel('')



ax2=plt.subplot(212)

sns.barplot(y=p_t_s.head(20).tag,x=p_t_s.title,palette='Blues_r')

display()

ax2.set_title('popularity of tags according to number of shows (TV Shows)')

ax2.set_xlabel('Amount of movies tagged')

ax2.set_ylabel('')

display()
#getting all release years and durations 

duration_year_m=movies[['release_year','duration']]

duration_year_s=shows2[['release_year','duration','title']]





duration_year_m['duration']=duration_year_m['duration'].astype(int)

duration_year_s['duration']=duration_year_s['duration'].astype(int)

duration_year_m['release_year']=duration_year_m['release_year'].astype(int)

duration_year_s['release_year']=duration_year_s['release_year'].astype(int)







duration_year_s2=duration_year_s.groupby(['duration','release_year']).count().reset_index([0,1])

duration_year_s2=duration_year_s2.pivot_table(values='title', index='duration', columns='release_year', aggfunc='first')

duration_year_s2=duration_year_s2.fillna(0).astype(int)



def q1(x):

            return x.quantile(0.25)



# 90th Percentile

def q2(x):

            return x.quantile(0.75)



duration_year_avg_m=duration_year_m.groupby('release_year').agg(['mean',q1,q2,'max','min'])['duration']

duration_year_avg_s=duration_year_s.groupby('release_year').agg(['mean',q1,q2,'max','min'])['duration']

fig=plt.figure(figsize=(25,15))

plt.subplot(211)

ax1=sns.boxplot(x="release_year", y="duration", data=duration_year_m,palette='Reds')

plt.xticks(rotation=70)

plt.subplot(212)

# sns.scatterplot(x="release_year", y="duration", data=duration_year_s,color='b')

ax2 = sns.heatmap(data=duration_year_s2,annot=True,fmt='.0f',linewidths=.5,cmap="Blues")

display()
fig=plt.figure(figsize=(15,5))

ax1=plt.subplot(121)

sns.lineplot(data=duration_year_avg_m,palette='Reds_r')

ax1.set_xlim([1945,2019])

ax1=plt.subplot(122)

sns.lineplot(data=duration_year_avg_s,palette='Blues_r')

ax1.set_xlim([1985,2019])

display()
#getting all release years and durations 

duration_year_m_AD=movies[['year_added','duration','title']]

duration_year_s_AD=shows2[['year_added','duration','title']]



duration_year_m_AD2=duration_year_m_AD.groupby(['duration','year_added']).count().reset_index([0,1])

duration_year_s_AD2=duration_year_s_AD.groupby(['duration','year_added']).count().reset_index([0,1])



duration_year_m_AD['duration']=duration_year_m_AD['duration'].astype(int)

duration_year_s_AD['duration']=duration_year_s_AD['duration'].astype(int)

duration_year_m_AD['year_added']=duration_year_m_AD['year_added'].astype(int)

duration_year_s_AD['year_added']=duration_year_s_AD['year_added'].astype(int)



duration_year_s_AD2.drop(duration_year_s_AD2.loc[duration_year_s_AD2['year_added']==1900].index, inplace=True)

duration_year_s_AD2=duration_year_s_AD2.pivot_table(values='title', index='duration', columns='year_added', aggfunc='first')

duration_year_s_AD2=duration_year_s_AD2.fillna(0).astype(int)



duration_year_m_AD2.drop(duration_year_m_AD2.loc[duration_year_m_AD2['year_added'].isin([1900,2020])].index, inplace=True)

duration_year_m_AD2=duration_year_m_AD2.pivot_table(values='title', index='duration', columns='year_added', aggfunc='first')

duration_year_m_AD2=duration_year_m_AD2.fillna(0).astype(int)





def q1(x):

            return x.quantile(0.25)



# 90th Percentile

def q2(x):

            return x.quantile(0.75)





duration_year_avg_m_AD=duration_year_m_AD.groupby('year_added').agg(['mean',q1,q2,'max','min'])['duration']



duration_year_avg_s_AD=duration_year_s_AD.groupby('year_added').agg(['mean',q1,q2,'max','min'])['duration']

# display(duration_year_m_AD2)
fig=plt.figure(figsize=(20,8))





ax1=plt.subplot(121)

sns.boxplot(x="year_added", y="duration", data=duration_year_m_AD,palette='Reds')

ax2=plt.subplot(122)

# sns.scatterplot(x="year_added", y="duration", data=duration_year_s_AD2,color='b',size='title')



ax2 = sns.heatmap(data=duration_year_s_AD2,annot=True,fmt='.0f',linewidths=.5,cmap="Blues")

# ax2 = sns.stripplot(x="duration", y="year_added", data=duration_year_s_AD2)

# ax2.set_xlim([2005,2021])

display()
fig=plt.figure(figsize=(15,5))

ax1=plt.subplot(121)

sns.lineplot(data=duration_year_avg_m_AD,palette='Reds_r')

ax1.set_xlim([2008,2019])

ax1=plt.subplot(122)

sns.lineplot(data=duration_year_avg_s_AD,palette='Blues_r')

ax1.set_xlim([2012,2019])

display()
import datetime as dt



RD_AD=NF_data[['type','release_year','year_added','date_added','month_added','day_added']]

RD_AD[['release_year','year_added','month_added','day_added']]=RD_AD[['release_year','year_added','month_added','day_added']].astype(int)



RD_AD['delta']=RD_AD['year_added']-RD_AD['release_year']

RD_AD = RD_AD[RD_AD.year_added != 1900]

RD_AD['DateTime'] = RD_AD[['year_added', 'month_added', 'day_added']].apply(lambda s : pd.Timestamp(dt.datetime(*s)),axis = 1)

RD_AD['DateTime'] = RD_AD.DateTime.astype(np.int64)

display(RD_AD)


import scipy.stats as stats



plt.figure(figsize=(16,10))

plt.subplot(121)

ax1=sns.regplot(x='release_year',y='DateTime',data=RD_AD[RD_AD['type']=='Movie'],color='r')

ax1.set_yticklabels([dt.datetime.fromtimestamp(ts / 1e9).strftime('%m/%Y') for ts in ax1.get_yticks()])

ax1.set_ylabel('Date added to netflix')

ax1.set_xlabel('Release year')

ax1.set_title('Date added vs release date for movies')



plt.subplot(122)

ax2=sns.regplot(x='release_year',y='DateTime',data=RD_AD[RD_AD['type']=='TV Show'],color='b')

ax2.set_yticklabels([dt.datetime.fromtimestamp(ts / 1e9).strftime('%m/%Y') for ts in ax2.get_yticks()])

ax2.set_ylabel('Date added to netflix')

ax2.set_title('Date added vs release date for TV shows')

ax2.set_xlabel('Release year')





# display(RD_AD)

KDE=RD_AD.groupby(['type','delta']).count().reset_index([0,1])

plt.figure(figsize=(15,5))

ax = sns.violinplot(y="type", x="delta", data=KDE,bw=.4,palette=sns.color_palette(['r','b']),cut=0)

# ax = sns.boxplot(y="type", x="delta", data=KDE,color='k')

display()
import calendar



OVERY_m=NF_data.loc[NF_data.type=='Movie']





OVERY_m=OVERY_m.groupby(['year_added','month_added']).country.count().reset_index([0,1])

OVERY_m = OVERY_m[OVERY_m.year_added.isin([1900,2020])==False]

OVERY_m['month_added']=OVERY_m['month_added'].astype(int)

OVERY_m=OVERY_m.sort_values('month_added')

OVERY_m['month_added'] = OVERY_m['month_added'].apply(lambda x: calendar.month_abbr[x])

OVERY_m = OVERY_m.pivot(index='year_added', columns='month_added')





OVERY_s=NF_data.loc[NF_data.type=='TV Show']

OVERY_s=OVERY_s.groupby(['year_added','month_added']).country.count().reset_index([0,1])

OVERY_s = OVERY_s[OVERY_s.year_added.isin([1900,2020])==False]

OVERY_s['month_added']=OVERY_s['month_added'].astype(int)

OVERY_s=OVERY_s.sort_values('month_added')

OVERY_s['month_added'] = OVERY_s['month_added'].apply(lambda x: calendar.month_abbr[x])





OVERY_s = OVERY_s.pivot(index='year_added', columns='month_added')



fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,6))









sns.heatmap(OVERY_m.country, ax=ax1,cmap='Reds')

sns.heatmap(OVERY_s.country, ax=ax2,cmap='Blues')

plt.show()
OVERY_trend=NF_data.groupby(['type','year_added','month_added']).country.count().reset_index([0,1,2])

OVERY_trend[['year_added','month_added']]=OVERY_trend[['year_added','month_added']].astype(int)

OVERY_trend=OVERY_trend.sort_values(['month_added','year_added'])

OVERY_trend = OVERY_trend[OVERY_trend.year_added.isin([2012,2013,2014,2015,2016,2017,2018,2019])]

OVERY_trend['month_added'] = OVERY_trend['month_added'].apply(lambda x: calendar.month_abbr[x])







OVERY_trendT=NF_data.groupby(['type','year_added','month_added']).country.count().reset_index([0,1,2])

# OVERY_trendT[['year_added','month_added']]=OVERY_trend[['year_added','month_added']].astype(int)













OVERY_trendTm=OVERY_trendT.groupby(['type','month_added']).country.agg(['mean']).reset_index([0,1])

OVERY_trendTm['month_added']=OVERY_trendTm['month_added'].astype(int)

OVERY_trendTm=OVERY_trendTm.sort_values(['month_added'])



OVERY_trendTm['metric']='mean'

OVERY_trendTm=OVERY_trendTm.reset_index().drop(columns='index').reset_index().rename(columns={"mean": "val"})













OVERY_trendTs=OVERY_trendT.groupby(['type','month_added']).country.agg(['sum']).reset_index([0,1])

OVERY_trendTs['month_added']=OVERY_trendTs['month_added'].astype(int)

OVERY_trendTs=OVERY_trendTs.sort_values(['month_added'])

OVERY_trendTs['metric']='sum'

OVERY_trendTs=OVERY_trendTs.reset_index().drop(columns='index').reset_index().rename(columns={"sum": "val"})

# OVERY_trendTs['month_added'] = OVERY_trendTs['month_added'].apply(lambda x: calendar.month_abbr[x])











OVERY_tt=pd.concat([OVERY_trendTs,OVERY_trendTm])

OVERY_tt=OVERY_tt.sort_values(['month_added'])

OVERY_tt['month_added'] = OVERY_tt['month_added'].apply(lambda x: calendar.month_abbr[x])





# display(OVERY_tt)






# sns.factorplot(x='month_added',y='country',hue='year_added',data=OVERY_trend[OVERY_trend.type=='Movie'],palette='Reds',ax=ax1)



# sns.factorplot(x='month_added',y='country',hue='year_added',data=OVERY_trend[OVERY_trend.type=='TV Show'],palette='Blues',ax=ax2)





# g = sns.FacetGrid(OVERY_trend, col="type",height=10)

# g.map(sns.catplot,'month_added','country','year_added')



OVERY_trend=OVERY_trend.rename(columns={'country':'number of shows added','year_added':'year added','month_added':'month added'})



sns.catplot(x='month added',y='number of shows added',hue='year added',col='type',data=OVERY_trend,kind='point',palette='Reds',height=6,ylabel='amount of shows uploaded')



display()
ax=sns.lineplot(x='month_added',y='val',hue='type',style='metric',data=OVERY_tt,palette=sns.color_palette(['b','red']),sort=False)

ax.set_ylim([0,500])

display()