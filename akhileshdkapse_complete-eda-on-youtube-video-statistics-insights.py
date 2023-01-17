import numpy as np                              # linear algebra
import pandas as pd                             # data processing, CSV file I/O (e.g. pd.read_csv)

import missingno as mss                         # Missing data visualization module for Python
import matplotlib.pyplot as plt                 # Data visualization
import seaborn as sns                           #  Data visualization
sns.set_style('darkgrid')
from datetime import datetime                   # For Time formate operations
import matplotlib.cm as cm                 
from collections import Counter                 # Counter
import string                                   # strings

from wordcloud import WordCloud, STOPWORDS      # WordCloud 
from tqdm import tqdm                           # keep track to iterations



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_json=pd.read_json('/kaggle/input/youtube-new/IN_category_id.json').head()
df_json.head()
df_json['items'][0]
df=pd.read_csv('/kaggle/input/youtube-new/INvideos.csv')
df.head()
def permute(x):
    y=[x.split('.')[0],x.split('.')[2],x.split('.')[1]]
    return '-'.join(y)

def convert_to_datetime(df):
    df['publish_date']=pd.to_datetime(df['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')        # trending date ---> yy|dd|mm
                                                                                                 # publish time  ---> yyyy|mm|dd
    df['publish_date']=df['publish_date'].apply(lambda x: str(x)[2:10])
    
    df["publishing_day"] = df["publish_time"].apply(
    lambda x: datetime.strptime(x[:10], "%Y-%m-%d").date().strftime('%a'))
    
    df.publish_time=df.publish_time.apply(lambda x: str(x)[11:-8])
    df.trending_date=df.trending_date.apply(permute)
    
    return df
def load_data(country_name,fill_catagory=False,change_date=False):
    file=country_name+str('videos.csv')
    root_path='/kaggle/input/youtube-new'
    path=os.path.join(root_path,file)
    df=pd.read_csv(path)
    if fill_catagory:
        title_dict={}
        json_path=os.path.join(root_path,(country_name+str('_category_id.json')))
        json_df=pd.read_json(json_path)
        for dict_ in json_df['items']:
            title_dict[dict_['id']]=dict_['snippet']['title']
        global missing_title_cata   # to keep track of missing id ,which shoudn't map from .csv ->.json file
        missing_title_cata=0
        def apply_title(x,dictionary):
            global missing_title_cata
            try:
                return dictionary[str(x)]
            except:
                missing_title_cata+=1
                return np.nan
        df['Title']=df['category_id'].apply(apply_title,args=(title_dict,))
        
        if change_date:
            df=convert_to_datetime(df)
    
        print('{} Titles missing'.format(missing_title_cata))
    return df
df=load_data('IN',fill_catagory=True,change_date=True)
df.head()
plt.figure(figsize=(18,7))
sns.heatmap( df.isnull() , cmap = 'viridis' , yticklabels= False , cbar = True )
df.info(verbose=0)
df["description"] = df["description"].fillna(value=" ")
def video_per_year(df,name):
    df=df.trending_date.apply(lambda x: str(20) + x[:2]).value_counts().reset_index()
    df.rename(columns={"index": "year", "trending_date": "video_count"},inplace=True)
    sns.barplot(x='year',y='video_count' ,data=df,palette="Set1")
    

video_per_year(df,'in')    
def num_title_posted(df):
    color = ["#06547a","#36796e","#3e8b7e","#6edda2", "#45a4b8", "#b6f5f6", "#2ecc71"]
    df=df.Title.value_counts()
    df=pd.DataFrame(df)
    df.reset_index(level=0,inplace=True)
    plt.figure(figsize=(25,10))
    df.columns=['Title','value_count']
    sns.barplot(y='value_count', x="Title", data=df,palette=sns.color_palette(color))
num_title_posted(df)
def Plot_hist(df,limit=None,factor='views', per_limit=None):
    if per_limit is not None:
        per=(len(df[df[factor]<per_limit][factor])/len(df[factor]) )*100
        print('{0} percent of trended videos got less than {1} {2}'.format(per,per_limit,factor))
    if limit is None:
        plt.figure(figsize=(14,7))
        sns.distplot(df[factor], kde=False, color='#976393')
        plt.ylabel('No. of videos')
    else:
        f,ax=plt.subplots(1,2 ,figsize=(20,7))
        ax[0].set( ylabel="No. of Videos")
        ax[1].set(ylabel="No. of Videos")
        sns.distplot(df[factor], kde=False, color='black', ax=ax[0])
        sns.distplot(df[df[factor] < limit][factor],kde=False, color='b', ax=ax[1])
    
Plot_hist(df,factor='views', limit=0.75e7, per_limit=1.5e6)
Plot_hist(df, factor='likes', limit=2.5e5, per_limit=25000)
# Function for Single Factor analyses
def analyse_by_title(df,factor='views',name=None):
    print(str(factor))
    df=df.groupby(['Title'])
    df=df[factor].mean()
    df.sort_values(ascending=False,inplace=True)
    df=pd.DataFrame(df)
    df.reset_index(level=0,inplace=True)
    plt.figure(figsize=(20,10))
    if name is not None:
        plt.title('Number of {} per Trending Video of each Title category in {}'.format(factor,name))
    else:
        plt.title('Number of {} per Trending Video of each Title category'.format(factor))
    sns.barplot(x=factor, y="Title", data=df)
    
# Function for MULTI Factor analyses
def analyse_by_mulfactor(df,factors,name=None):
    print(factors)
    df=df.groupby(['Title'])
    f, axes = plt.subplots(1, 2, figsize=(30,9))
    palettes=['Blues_r','BuGn_r']
    for i, factor in enumerate(factors):
        df1=df[factor].mean().drop(['Movies','Gaming','Pets & Animals','Travel & Events','Shows','Autos & Vehicles'])
        df1.sort_values(ascending=False,inplace=True)
        df1=pd.DataFrame(df1)
        df1.reset_index(level=0,inplace=True)
        plt.figure(figsize=(20,10))
        if name is not None:
            axes[i].set_title('Number of {} per Trended Video of each Title category/Content in {}'.format(factor.capitalize(),name.capitalize()),
                              fontsize=20 )
        else:
            axes[i].set_title('Number of {} per Trended Video of each Title category'.format(factor))
        sns.barplot(x=factor, y="Title", data=df1,ax=axes[i], palette=palettes[i])
    
analyse_by_mulfactor(df,['views','comment_count'],name='India')
analyse_by_mulfactor(df,['likes','dislikes'],name='India')
#Comments Disabled on Videos

def comment_analysis(df,name):
    labels=['Comments_disalble','Comments_enable']
    Values=[df.comments_disabled.value_counts()[1],df.comments_disabled.value_counts()[0]]
    explode=(0.05,0.2)
    f,ax=plt.subplots(1,2, figsize=(25,7))
    ax[0].pie(Values,explode=explode, labels=labels, autopct='%1.1f%%',
           shadow=True, startangle=90)
    
    df=pd.DataFrame(df.groupby(['Title']).comments_disabled.value_counts())
    df.columns=['Counts']
    df.reset_index(level=[1,0],inplace=True)
    plt.title('Comments status of Trending Videos in {}.'.format(name.capitalize()))
    sns.barplot(x='Counts',y='Title',data=df,hue='comments_disabled',ax= ax[1])
    return df
    
    
comment_analysis(df,'INDIA')
df['Month']=df.publish_date.apply(lambda x: x[3:5]+str(-20)+x[:2])
df.Month.value_counts()
# We don't have much data of May 2017 ,So we can exclude it.
df=df[df.Month!='05-2017']
#no of content type posted each month
plt.figure(figsize=(30,13))
sns.countplot(x='Month', data=df, hue='Title',palette='Set1')
# Put the legend out of the figure
plt.xlabel('mm/yyyy')
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.1)
def Plot_Published_Daytime(df,name):
    Series=df.publishing_day.value_counts()
    
    df=df.publish_time.apply(lambda x: x[:2]).value_counts().sort_index().to_frame()
    df.columns=['Video_Count']; df.reset_index(level=0)
    
    f,ax=plt.subplots(1,2, figsize=(25,7))
    
    ax[0].set_title("DayTime analysis of Video uploads going to Trend in {}".format(name.capitalize()),fontsize=16)
    # Set common labels
    ax[0].set_xlabel('Publishing_hour')
    ax[0].set_ylabel('video Counts')
    ax[0].plot(df.index,df.Video_Count,marker='o',ls='--', color='black',
               markerfacecolor="r",linewidth=2, markersize=10)
    
    ax[1].set_xlabel('Publishing_Day')
    ax[1].set_ylabel('video Counts')
    ax[1].set_title("Publishing day distributions in {}".format(name.capitalize()),fontsize=16)
    sns.barplot(x=Series.index, y=Series.values, ax=ax[1],palette='ocean', order=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
    

Plot_Published_Daytime(df,'India')
# Selecting Random 5 rows from DataFrame
df.sample(5)
plt.figure(figsize=(12,6))
label=[x.replace('_', ' ',3).title() for x in df.select_dtypes(include=['int','bool']).columns.values]
sns.heatmap(df.corr(), annot=True,
           xticklabels=label, yticklabels=label, cmap='Blues')
def Plot_corr(df):
    f,ax=plt.subplots(2,2,figsize=(20,10))
    
    sns.scatterplot(x='views',y='likes', data=df, ax=ax[0][0])
    sns.scatterplot(x='comment_count',y='likes', data=df, ax=ax[0][1])
    sns.scatterplot(x='dislikes',y='likes', data=df, ax=ax[1][0])
    sns.scatterplot(x='comment_count',y='views', data=df, ax=ax[1][1])

Plot_corr(df)
def Title_dis_visulize(df):
    f,axes=plt.subplots(1,2, figsize=(20,8))
    sns.scatterplot(y=df.description.apply(lambda x: len(str(x))),x=df.views, ax=axes[0])
    sns.scatterplot(y=df.title.apply(len),x=df.views, ax=axes[1])
    
    
Title_dis_visulize(df)
df.head()
def Show_Wordcolud(df,content_category):
    print('This may take some Time :-)')
    if len(content_category) !=4:
        raise ValueError('Incomplete List to Plot. Expected len:4,got:{}'.format(len(content_category)))
    else:
        
        f,ax=plt.subplots(2,2, figsize=(26,20))
        i=0
        for content in tqdm(content_category):
            mylist=df[df.Title==str(content)].title.apply(lambda x: x.split())
            mylist = [x for y in mylist for x in y]
            mylist=[x for x in mylist if x not in string.punctuation]
            unique_string=(" ").join(mylist)
            wordcloud = WordCloud(width=1400, height=1200, background_color='white', max_words=180).generate(unique_string)
            ax[i//2][i%2].grid(False)
            ax[i//2][i%2].set_title("WordCloud if Titles in '{}' video content".format(content),fontsize=20)
            ax[i//2][i%2].imshow(wordcloud, aspect='auto')
            i+=1
        plt.show()
 
Show_Wordcolud(df,df.Title.value_counts().index[:4].to_list())
def Plot_top20_channel(df,name):
    df=pd.DataFrame(df.channel_title.value_counts()[:20])
    df=df.reset_index(level=0)
    df.columns=['Channel_Name', 'Trended Video Count']
    plt.figure(figsize=(12,6))
    plt.title('Top 20 Youtube Trending Channel in {}'.format(name))
    sns.barplot(x='Channel_Name', y='Trended Video Count', data=df, palette=sns.cubehelix_palette(20))
    plt.xticks(rotation=70)
Plot_top20_channel(df,'India')    
def Top_Channels(df):
    Content_list=['Gaming', 'Science & Technology', 'Music', 'Entertainment']
    f,ax=plt.subplots(2, 2, figsize=(25,12))
    color=[sns.light_palette((210, 90, 60), input="husl",reverse=True), sns.cubehelix_palette(5,reverse=True),
          sns.light_palette("navy", reverse=True), sns.light_palette("green",reverse=True)]
    for i, content in enumerate(Content_list):
        print('....{}....'.format(content))
        print(df[df.Title==content].channel_title.value_counts()[:5])
        print('---------------------------')
        ax[i//2][i%2].set(xlabel="No. of Videos Trended", ylabel="Channel")
        ax[i//2][i%2].set_title(content.capitalize(),fontsize=22)
        sns.barplot(x=df[df.Title==content].channel_title.value_counts()[:5].values,
                    y=df[df.Title==content].channel_title.value_counts()[:5].index, ax=ax[i//2][i%2], palette=color[i] )
        
Top_Channels(df)

