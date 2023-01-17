from IPython.display import Image

import os

Image('../input/youtubemarketing/youtube-marketing.jpg')
# import necessary libraries

import os

import pandas as pd

import numpy as np

import json

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm

import collections

import pickle

import gc
# Read all files

json_files=[]

csv_files=[]

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        fname=os.path.join(dirname, filename)

        if fname.endswith('csv'):

            csv_files.append(fname)

        elif fname.endswith('json'):

            json_files.append(fname)

            

            

# Reorder CSV files

country_codes=list(map(lambda string:''.join(list(filter(lambda word:word.isupper(),string))),csv_files))

country_codes,order=zip(*sorted(list(zip(country_codes,range(len(country_codes)))), key=lambda val:val[0]))

csv_files=[csv_files[ind] for ind in order]



# Reorder json files

country_codes=list(map(lambda string:''.join(list(filter(lambda word:word.isupper(),string))),json_files))

country_codes,order=zip(*sorted(list(zip(country_codes,range(len(country_codes)))), key=lambda val:val[0]))

json_files=[json_files[ind] for ind in order]





def initialize_country_dataframe(dataframe,json_fname,country_code):

    '''First, remove duplicate rows from the dataframe, second, map category_id column to actual categories, thrid,

    new column in the dataframe called country_code'''

    

    df=dataframe.copy()

    df.drop_duplicates(inplace=True)

    

    with open(json_fname,'r') as f:

        json_data=json.loads(f.read())



    mapping_dict=dict([(int(dictionary['id']),dictionary['snippet']['title']) for dictionary in json_data['items']])



    df['category']=df['category_id'].replace(mapping_dict)

    del df['category_id']



    df['country_code']=country_code

    

    return df



# Initialize country-by-country dataframe using above written function

dataframes=[]

for ind,code in enumerate(country_codes):

    try:

        df=pd.read_csv(csv_files[ind])

    except:

        df=pd.read_csv(csv_files[ind],engine='python')

                

    df=initialize_country_dataframe(df,json_files[ind],code)

    print(code,df.shape)

    dataframes.append(df)

    

    

# Concatenate individual dataframe to form single main dataframe

dataframe=pd.concat(dataframes)

print(dataframe.shape)





# Remove videos with unknown video id

drop_index=dataframe[dataframe.video_id.isin(['#NAME?','#VALUE!'])].index

dataframe.drop(drop_index, axis=0, inplace=True)
'''# Create feature num_days that indicates the number of days the videos are in trend

video_ids=dataframe.video_id.unique().tolist()

num_days=[]

id_days={}

for vid in tqdm(video_ids):

    days=len(dataframe[dataframe.video_id==vid].trending_date.unique())

    id_days[vid]=days

    num_days.append(days)

    



# Create feature num_countries that indicates the number of countries in the videos trended

video_ids=dataframe.video_id.unique().tolist()

num_countries=[]

id_countries={}

for vid in tqdm(video_ids):

    days=len(dataframe[dataframe.video_id==vid].country_code.unique())

    id_countries[vid]=days

    num_countries.append(days)'''
# Reading pre-calculated dictionaries

with open('/kaggle/input/id-dayspickle/id_days.pickle','rb') as f:

    id_days=pickle.load(f)



with open('/kaggle/input/id-countriespickle/id_countries.pickle','rb') as f:

    id_countries=pickle.load(f)



num_days=id_days.values()

num_countries=id_countries.values()



# Adding feature num_days into the dataframe

def n_days_replace(vid):

    return id_days[vid]



dataframe['num_days']=dataframe.video_id.apply(func=n_days_replace)



# Adding feature num_countries into the dataframe

def n_countries_replace(vid):

    return id_countries[vid]



dataframe['num_countries']=dataframe.video_id.apply(func=n_countries_replace)
# Create feature days_lapse that indicates the number of days before videos are in trend

def unique_video_id(keep='last'):

    '''Removes duplicate videos to keep single record according to trending_date and keep argument.'''

    df=dataframe.copy()

    

    df.sort_values(by=['video_id','trending_date'],axis=0,inplace=True)

    df.drop_duplicates(subset='video_id',keep='last',inplace=True)

    

    return df



df=unique_video_id(keep='first')



def publish_date(string):

    return string.split('T')[0]



df['publish_date']=pd.to_datetime(df.publish_time.apply(func=lambda val:publish_date(val)),format='%Y-%m-%d')

df['trending_date']=pd.to_datetime(df.trending_date,format='%y.%d.%m')

df['days_lapse']=df['trending_date']-df['publish_date']



df.days_lapse=df.days_lapse.apply(func=lambda val:val.days).values

id_days_lapse=dict(zip(df.video_id.values,df.days_lapse.values))



def n_days_lapse_replace(vid):

    return id_days_lapse[vid]



dataframe['days_lapse']=dataframe.video_id.apply(func=n_days_lapse_replace)



# Create feature trend_month that indicates month the videos are in trend

def trend_month(string):

    return int(string.split('.')[2])



dataframe['trend_month']=dataframe.trending_date.apply(func=lambda val:trend_month(val))



# Create feature publish_month that indicates the months that the videos are published in

def publish_month(string):

    return int(string.split('T')[0].split('-')[1])

dataframe['publish_month']=dataframe.publish_time.apply(func=lambda val:publish_month(val))



# Create feature publish_hour that indicates the hours that the videos are published in

def publish_hour(string):

    return int(string.split('T')[1].split(':')[0])



dataframe['publish_hour']=dataframe.publish_time.apply(func=lambda val:publish_hour(val))
trending_days=collections.Counter(num_days)

days,freq=zip(*sorted(trending_days.items(),key=lambda val:val[0]))



fig,[ax1,ax2]=plt.subplots(nrows=2,ncols=1,figsize=(14,10))



cmap = plt.get_cmap('GnBu')

colors=[cmap(i) for i in np.linspace(0, 1, len(days))]

ax1.bar(range(len(days)),np.log(freq),color=colors)

ax1.set_xticks(range(len(days)))

ax1.set_xticklabels(days)    



labels=[str(val) for val in freq]

for ind,val in enumerate(np.log(freq)):

    ax1.text(ind,val+0.1,labels[ind],ha='center')



ax1.set_xticks(range(len(days)))

ax1.set_xticklabels(days)



ax1.set_ylabel('Log frequency')



cum_arr=np.cumsum(freq)

max_val=np.max(cum_arr)

min_val=np.min(cum_arr)



ax2.plot((cum_arr-min_val)/(max_val-min_val))

ax2.set_xticks(range(len(days)))

ax2.set_xticklabels(days)

ax2.set_ylabel('Cumulative proportion of number of videos')

ax2.set_xlabel('For number of days videos are in trend');
trending_countries=collections.Counter(num_countries)

nc,freq=zip(*sorted(trending_countries.items(),key=lambda val:val[0]))



fig,ax=plt.subplots(figsize=(14,6))



cmap = plt.get_cmap('PuBu')

colors=[cmap(i) for i in np.linspace(0, 1, len(nc))]

ax.bar(range(len(nc)),np.log(freq),color=colors)

ax.set_xticks(range(len(nc)))

ax.set_xticklabels(nc)    



labels=[str(val) for val in freq]

for ind,val in enumerate(np.log(freq)):

    ax.text(ind,val+0.1,labels[ind],ha='center')



ax.set_ylabel('Log frequency')

ax.set_xlabel('For number of countries videos are in trend')

ax.set_title('Discrete log frequency plot for the number of countries videos are in trend');
df=unique_video_id(keep='first')



months,counts=zip(*sorted(df.publish_month.value_counts().to_dict().items(), key=lambda val:val[0]))



fig,ax=plt.subplots(figsize=(15,5))



cmap = plt.get_cmap('Set3')

colors=[cmap(i) for i in range(len(months))]



ax.bar(months,counts,color=colors)

ax.set_xticks(range(1,len(months)+1))

ax.set_xticklabels(months)

ax.set_xlabel('Months')

ax.set_ylabel('Number of videos published')

for ind,val in enumerate(counts):

    ax.text(months[ind],val+500,val,ha='center');
yr_months=list(map(lambda val:'-'.join(val.split('T')[0].split('-')[:-1]),df.publish_time.unique()))



yr_months,counts=zip(*sorted(collections.Counter(yr_months).items(),key=lambda val:val[0]))



fig,ax=plt.subplots(figsize=(20,6))



cmap = plt.get_cmap('Reds')

colors=[cmap(i) for i in np.linspace(0, 1, len(counts))]



ax.scatter(range(len(yr_months)),counts,color=colors,)

ax.set_xticks(range(len(yr_months)))

ax.set_xticklabels(yr_months,rotation=90)

ax.set_xlabel('Publsihed-months in the years')

ax.set_ylabel('Number of videos published')

ax.set_title('Frequency plot of number of videos published in the months of the years');
df=unique_video_id(keep='first')



hours,counts=zip(*sorted(df.publish_hour.value_counts().to_dict().items(),key=lambda val:val[0]))



fig,ax=plt.subplots(figsize=(10,6))



cmap = plt.get_cmap('twilight')

colors=[cmap(i) for i in np.linspace(0, 1, len(hours))]



ax.bar(hours,counts,color=colors)

ax.set_xticks(range(len(hours)))

ax.set_xticklabels(hours)

ax.set_xlabel('Hour of a day')

ax.set_ylabel('Number of videos published');
df=unique_video_id(keep='first')



days_lapse=df['days_lapse']

days_lapse_count=days_lapse.value_counts().to_dict()

days,count=zip(*sorted(list(filter(lambda val:val[1]>1,days_lapse_count.items())),key=lambda val:val[0]))



fig,[ax1,ax2]=plt.subplots(figsize=(19,13),nrows=2,ncols=1)



cmap = plt.get_cmap('autumn')

colors=[cmap(i) for i in np.linspace(0, 1, len(days))]



ax1.bar(range(len(days)),np.log(count),width=0.6,color=colors)

ax1.set_xticks(range(len(days)))

ax1.set_xticklabels(days,rotation=45)

ax1.set_ylabel('log of frequency count')

ax1.set_xlabel('Number of days pass before videos are trending')

ax1.set_title('Discrete frequency plot for number of days before videos are trending')





cum_arr=np.cumsum(count)

max_val=np.max(cum_arr)

min_val=np.min(cum_arr)

ax2.plot((cum_arr-min_val)/(max_val-min_val))

ax2.set_xticks(range(len(days)))

ax2.set_xticklabels(days,rotation=45)

ax2.set_ylabel('Cumulative proportion of number of videos')

ax2.set_xlabel('Number of days pass before videos are trending');
df=unique_video_id(keep='first')

print('Total number of unique videos:',df.shape[0])

print('Total number of unique videos that take less than 31 days to be in trend:',df[df.days_lapse<31].shape[0])

fig,ax=plt.subplots(figsize=(20,6))

df[df.days_lapse<31].boxplot(column='days_lapse',by='publish_hour',rot=90,ax=ax)

ax.set_xlabel('Hours')

ax.set_ylabel('days lapse')

ax.set_title('')

fig.suptitle('Box plot for videos that took less than 31 days to be in trend');
df=dataframe.copy()

df.sort_values(by=['video_id','trending_date'],inplace=True)

df.drop_duplicates(subset='video_id',keep='last',inplace=True)



fig=plt.figure(figsize=(20,15))

fig.tight_layout()



ax1=fig.add_subplot(221)

ax2=fig.add_subplot(222)

ax3=fig.add_subplot(212)



df.boxplot(column='views',by='num_countries',ax=ax1)

ax1.set_xlabel('For number of countries videos trend')

ax1.set_ylabel('Final number of views')

ax1.set_title('')



df.boxplot(column='likes',by='num_countries',ax=ax2)

ax2.set_xlabel('For number of countries videos trend')

ax2.set_ylabel('Final number of likes')

ax2.set_title('')





df.boxplot(column='num_days',by='num_countries',ax=ax3)

ax3.set_xlabel('For number of countries videos trend')

ax3.set_ylabel('For number of days videos trend')

ax3.set_title('')



fig.suptitle('Boxplot: grouped by the number of countries');
df_disabled=unique_video_id()

df_disabled=df_disabled[df_disabled.comments_disabled==True]

df_disabled.sort_values(by=['video_id','comments_disabled'],inplace=True)

df_disabled.drop_duplicates(subset='video_id',keep='last',inplace=True)

disabled_dict=df_disabled.country_code.value_counts().to_dict()



df_enabled=unique_video_id()

df_enabled=df_enabled[df_enabled.comments_disabled==False]

df_enabled.sort_values(by=['video_id','comments_disabled'],inplace=True)

df_enabled.drop_duplicates(subset='video_id',keep='first',inplace=True)

enabled_dict=df_enabled.country_code.value_counts().to_dict()



dis_ena_prop={}

for country in disabled_dict.keys():

    dis_ena_prop[country]=disabled_dict[country]/enabled_dict[country]



fig,ax=plt.subplots(figsize=(13,6))



cmap = plt.get_cmap('Set3')

colors=[cmap(i) for i in range(len(days))]



countries=dis_ena_prop.keys()

values=list(dis_ena_prop.values())

ax.bar(countries,np.log(np.array(list(values))+1),color=colors)

ax.set_ylabel('log(1+values) transformed values')

ax.set_xlabel('Country codes')

ax.set_title('Percentage of unique trending videos that have comments section disabled over each country')



for ind,val in enumerate(np.log(np.array(list(values))+1)):

    ax.text(ind,val+0.001,str(round(values[ind]*100,1))+'%',ha='center')
df=unique_video_id()



# 1. Get number of views for each categories

views_per_category=df.groupby(by=['category'],as_index=False).views.sum()



# 2. Get number of views and category names

views_in_million=[int(views/1000000) for views in views_per_category.sort_values(by='views').views.values]

cat_val=views_per_category.sort_values(by='views').category.values



# 3. Normalize number of views for data visualization

relative_vals=views_per_category.sort_values(by='views').views.values

max_val=np.max(relative_vals)

min_val=np.min(relative_vals)

diff=max_val-min_val

ms_val=(relative_vals-min_val)/diff



# 4. Create axes for plotting

fig,ax=plt.subplots(figsize=(20,3))

# 4.1 Add one more axis

bx=ax.twiny()

x=range(len(cat_val))

y=[5]*len(cat_val)



# 5. Plot one category at a time using matplotlib scatter plot

for ind,cat in enumerate(cat_val):

    ax.scatter(x[ind],y[ind],s=ms_val[ind]*10000,cmap='Blues',alpha=0.5,edgecolors="grey", linewidth=2)

ax.set_xticks(range(len(cat_val)))

ax.set_xticks(range(len(cat_val)))

ax.set_yticklabels([])

ax.set_xticklabels(cat_val,rotation=90)

ax.set_xlabel('Video categories',fontsize=16)



# 6. Write number of views in millions on the x-axis above the plot

bx.set_xticks(range(len(views_in_million)))

bx.set_xticklabels(views_in_million,rotation=90)

bx.set_xlabel('Number of views in millions',fontsize=16);
# 1. Replace category 29 by string 'Other'

dataframe.category.replace({29:'Other'},inplace=True)



# 2. Count number of occurances of video category for each category

country_by_category=dataframe.groupby(by='country_code')['category'].value_counts()



# 3. Write function that will plot a pie-chart

def pie_chart(country_code,axis):

    '''Plots a pie_chart for a country_by_category series for a given country code on given axis'''

    cmap = plt.get_cmap('Spectral')

    colors=[cmap(i) for i in np.linspace(0, 1, len(country_by_category[country_code].index))]

    axis.pie(country_by_category[country_code].values,labels=country_by_category[country_code].index,autopct='%.2f',colors=colors,shadow=True)

    axis.set_title(country_code,fontsize=14);



# 4. Plot individual pie-chart for each country

fig,ax=plt.subplots(nrows=5,ncols=2,figsize=(16,20))

for c_i in range(0,len(country_codes),2):

    col=0

    ind=c_i//2

    pie_chart(country_codes[c_i],ax[ind][col])

    pie_chart(country_codes[c_i+1],ax[ind][col+1])



# 5. Plot pie-chart for all countries together

fig,ax=plt.subplots(figsize=(4,4))

cmap = plt.get_cmap('Spectral')

all_countries_prop=dataframe.category.value_counts()

colors=[cmap(i) for i in np.linspace(0, 1, len(all_countries_prop.index))]

ax.pie(all_countries_prop.values,labels=all_countries_prop.index,autopct='%.2f',colors=colors,shadow=True)

ax.set_title('All Countries');
# 1. Create temporary dataframe

df=dataframe[dataframe.comments_disabled==False].copy()



# 2. Keep single record of each video_id the last day video is in trend

df.sort_values(by=['video_id','trending_date'],inplace=True)

df.drop_duplicates(subset='video_id',keep='last',inplace=True)



# 3. Add new column on the proportion

df['odds_of_likes']=df['likes']/(df['views'])



# 4. Create axis for distribution plot

fig1,ax1=plt.subplots(figsize=(20,20))

fig1.tight_layout()



# 5. Plot histogram



df.hist(column='odds_of_likes',by='category',ax=ax1)

ax1.set_title('Distribution plot of the proportion likes per views grouped by the video categories')



# 6. Create axis for box-plot

fig2,ax2=plt.subplots(figsize=(15,7))

fig2.tight_layout()



# 7. Create box-plot

df.boxplot(column='odds_of_likes',by='category',rot=90,ax=ax2)
# 1. Plot correation matrix and pairs of countries with higher than 60% correlation

def get_corr_pairs(category, threshold=0.6):

    '''Get list of pairs of country codes those have greater than or equal to 60% pearson correlation'''

    series={}

    for country in country_codes:

        series[country]=dataframe[(dataframe.category==category) & (dataframe.country_code==country)].groupby(by='trending_date')['views'].sum().values

    

    key_list=series.keys()

    drop_list=[]

    

    for key in key_list:

        if not len(series[key])==205:

            drop_list.append(key)

            

    for key in drop_list:

        del series[key]

    

    df=pd.DataFrame(series)

    

    #fig,ax=plt.subplots(figsize=(8,8))

    corr=df.corr()

    #mask = np.triu(np.ones_like(corr, dtype=np.bool))

    #sns.heatmap(corr,mask=mask,annot=True,ax=ax)

    #fig.suptitle(category);

    

    corr=corr.abs()

    s = corr.unstack()

    so = s.sort_values(kind="quicksort")



    df=so.to_frame().reset_index().rename(columns={0:'corr_val'})

    corr_pairs=df[(df.corr_val>=threshold) & (df.corr_val!=1.0)][['level_0','level_1']].values

    

    list_of_sets=[set((pair[0],pair[1])) for pair in corr_pairs]



    corr_pairs=[]

    for pair in list_of_sets:

        if pair not in corr_pairs:

            corr_pairs.append(pair)

    corr_pairs=[list(pair) for pair in corr_pairs]

    return corr_pairs



# 2. Plot sum of viewers over the series for the country pairs

def plot_patterns(corr_pairs, category, figure_size=(15,20)):

    n_rows=len(corr_pairs)

    fig,ax=plt.subplots(nrows=n_rows, ncols=1,figsize=figure_size)

    fig.tight_layout(pad=5)

    fig.suptitle(category,fontsize=17)

    if n_rows>1:

        for ind,pair in enumerate(corr_pairs):

            country_1=pair[0]

            country_2=pair[1]

            dataframe[(dataframe.category==category) & (dataframe.country_code==country_1)].groupby(by='trending_date')['views'].sum().plot(ax=ax[ind],label=country_1)

            dataframe[(dataframe.category==category) & (dataframe.country_code==country_2)].groupby(by='trending_date')['views'].sum().plot(ax=ax[ind],label=country_2)

            ax[ind].legend()

            ax[ind].set_title(' '.join(pair))

    else:

        for ind,pair in enumerate(corr_pairs):

            country_1=pair[0]

            country_2=pair[1]

            dataframe[(dataframe.category==category) & (dataframe.country_code==country_2)].groupby(by='trending_date')['views'].sum().plot(ax=ax,label=country_1)

            dataframe[(dataframe.category==category) & (dataframe.country_code==country_1)].groupby(by='trending_date')['views'].sum().plot(ax=ax,label=country_2)

            ax.legend()

            ax.set_title(' '.join(pair))



# 3. Plot series for each video category at a time

for category in dataframe.category.unique():

    if category in ['Music', 'Comedy', 'Entertainment', 'News & Politics','People & Blogs', 'Howto & Style', 'Film & Animation', 'Sports']:

        corr_pairs=get_corr_pairs(category)

        if len(corr_pairs)>0:

            if len(corr_pairs)<5:

                figure_size=(20,8)

            else:

                figure_size=(15,20)

            plot_patterns(corr_pairs,category,figure_size)

        else:

            print(category,'does not have series having correlation atleast 0.6')
# 1. Clean tags data



# 1.1 Write function to decontract tags data

import re

def decontraction(phrase):

    '''Decontracts given strings'''

    # specific

    phrase = re.sub(r"won't", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)

    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase



# 1.2 Write function to clean data

def clean_data(string):

    '''Removes all unwanted charetchters from a string'''

    return ' '.join(decontraction(string.lower().replace('"','').replace('/','|')).split('|'))



# 1.3 Use pandas apply function to make use of clean_data funcion for every row of the dataset to clean data

tag_list=dataframe.tags.apply(func=clean_data)



# 2. Plot word cloud



# 2.1 Wirte function to create wordcloud

from wordcloud import WordCloud

def plot_wordcloud(df,country,axis):

    '''Plots wordcloud of tags columns for a given country code on the given axis'''

    tag_list=df[df.country_code==country].tags.apply(func=clean_data)

    text=' '.join(tag_list)

    wordcloud=WordCloud().generate(text)

    

    axis.imshow(wordcloud)

    axis.set_title(country)



# 2.2 Plot word cloud using above written function

fig,ax=plt.subplots(nrows=5,ncols=2,figsize=(20,20))

df=unique_video_id(keep='first')

for c_i in range(0,len(country_codes),2):

    ind=c_i//2

    col=0

    plot_wordcloud(df,country_codes[c_i],ax[ind][col])

    plot_wordcloud(df,country_codes[c_i+1],ax[ind][col+1])
# Convert trending date column into datetime object

dataframe.trending_date=pd.to_datetime(dataframe.trending_date,format='%y.%d.%m')



# Sort dataframe according to the trending date

dataframe.sort_values(by='trending_date',axis=0,inplace=True)



# Create train and test data frames based on trending date where we want predict number of likes of

# the future trending videos

df_train=dataframe[dataframe.trending_date<'2018-05-01'].copy()

df_test=dataframe[dataframe.trending_date>='2018-05-01'].copy()



del df

gc.collect()



# Seperate predictors from the target feature



y_train=df_train.likes.values

y_test=df_test.likes.values



del df_train['likes'], df_test['likes']

gc.collect()



X_train=df_train.copy()

X_test=df_test.copy()



del df_train,df_test

gc.collect()



X_train.shape, X_test.shape, y_train.shape, y_test.shape
fig,ax=plt.subplots()

ax.hist(y_train,bins=500)

ax.set_xlabel('number of likes')

ax.set_title('Histogram of y_train');
# Log transformation of the train-target feature

fig,ax=plt.subplots()

ax.hist(np.log(y_train+1))

ax.set_xlabel('transformed number of likes')

ax.set_title('Histogram of transformed y_train');
X_train['views']=np.log(X_train['views']+1)

X_train['comment_count']=np.log(X_train['comment_count']+1)



X_test['views']=np.log(X_test['views']+1)

X_test['comment_count']=np.log(X_test['comment_count']+1)



train=X_train.copy()

train['likes']=np.log(y_train+1)



test=X_test.copy()
train.plot.scatter(x='views',y='likes');
train['views_cat']=pd.cut(train.views.values,bins=[0.0,7.5,10.0,12.5,15.0,17.5,20.0],labels='a b c d e f'.split(),right=True)



test['views_cat']=pd.cut(test.views.values,bins=[0.0,7.5,10.0,12.5,15.0,17.5,20.0],labels='a b c d e f'.split(),right=True)
train.boxplot(column='likes',by='views_cat');
train.plot.scatter(x='comment_count',y='likes');
train['comment_count_cat']=pd.cut(train.comment_count.values,bins=[-0.5,1,2,4,6,8,10,12,14,20],labels='a b c d e f g h i'.split(),right=True)



test['comment_count_cat']=pd.cut(test.comment_count.values,bins=[-0.5,1,2,4,6,8,10,12,14,20],labels='a b c d e f g h i'.split(),right=True)



train.boxplot(column='likes',by='comment_count_cat');
fig,ax=plt.subplots(figsize=(10,4))

train.boxplot(column='likes',by='comments_disabled',ax=ax)

ax.set_ylabel('log(likes+1)');
fig,ax=plt.subplots(figsize=(10,4))

train.boxplot(column='likes',by='video_error_or_removed',ax=ax)

ax.set_ylabel('log(likes+1)');
fig,ax=plt.subplots(figsize=(10,5))

train.boxplot(column='likes',by='category',ax=ax, rot=90)

ax.set_xlabel('category')

ax.set_ylabel('log(likes+1)');
fig,ax=plt.subplots(figsize=(10,5))

train.boxplot(column='likes',by='country_code',ax=ax)

ax.set_xlabel('country_code')

ax.set_ylabel('log(likes+1)');
fig,ax=plt.subplots(figsize=(10,5))

train.boxplot(column='likes',by='num_countries',ax=ax)

ax.set_xlabel('num_countries')

ax.set_ylabel('log(likes+1)');
fig,ax=plt.subplots(figsize=(10,5))

train.boxplot(column='likes',by='num_days',ax=ax)

ax.set_xticks(range(1,len(train.num_days.unique())+1))

ax.set_xticklabels(sorted(train.num_days.unique()),rotation=90)

ax.set_xlabel('num_days')

ax.set_ylabel('log(likes+1)');
train.num_days[(train.num_days>=8) & (train.num_days<15)]=8

train.num_days[(train.num_days>=15) & (train.num_days<30)]=15

train.num_days[(train.num_days>=30) & (train.num_days<40)]=30



test.num_days[(test.num_days>=8) & (test.num_days<15)]=8

test.num_days[(test.num_days>=15) & (test.num_days<30)]=15

test.num_days[(test.num_days>=30) & (test.num_days<40)]=30



train.num_days.replace(dict(zip(list(range(1,8))+[8,15,30],'a b c d e f g h i j'.split())),inplace=True)



test.num_days.replace(dict(zip(list(range(1,8))+[8,15,30],'a b c d e f g h i j'.split())),inplace=True)



fig,ax=plt.subplots(figsize=(15,6))

train.boxplot(column='likes',by='num_days',ax=ax)

#ax.set_xticks(range(1,len(list(range(8))+[8,15,30])+1))

#ax.set_xticklabels(list(range(8))+[8,15,30],rotation=90)

ax.set_xlabel('num_days')

ax.set_ylabel('log(likes+1)');
train.num_countries.replace(dict(zip(list(range(1,11)),'a b c d e f g h i j'.split())),inplace=True)

train.comments_disabled.replace({False:'false',True:'true'},inplace=True)

train.ratings_disabled.replace({False:'false',True:'true'},inplace=True)

train.video_error_or_removed.replace({False:'false',True:'true'},inplace=True)



test.num_countries.replace(dict(zip(list(range(1,11)),'a b c d e f g h i j'.split())),inplace=True)

test.comments_disabled.replace({False:'false',True:'true'},inplace=True)

test.ratings_disabled.replace({False:'false',True:'true'},inplace=True)

test.video_error_or_removed.replace({False:'false',True:'true'},inplace=True)
# Create train and test data frames for prediction

X_train=train[['comments_disabled','ratings_disabled','video_error_or_removed','category','country_code','num_countries','num_days','days_lapse','views_cat','comment_count_cat']]

X_test=test[['comments_disabled','ratings_disabled','video_error_or_removed','category','country_code','num_countries','num_days','days_lapse','views_cat','comment_count_cat']]
train_rows=X_train.shape[0]

data=pd.concat([X_train,X_test])



data=pd.get_dummies(data)



X_train=data[:train_rows].copy()

X_test=data[train_rows:].copy()



del data

gc.collect()



X_train.shape,X_test.shape
# Baseline linear regression model

from sklearn.linear_model import LinearRegression



lr=LinearRegression()

lr.fit(X_train,np.log(y_train+1))

lr.score(X_train,np.log(y_train+1))
from sklearn import metrics



y_pred=lr.predict(X_test)

y_pred=np.exp(y_pred)-1

metrics.mean_absolute_error(y_test,y_pred)