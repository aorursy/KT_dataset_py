## Requried library
!pip install isodate
import isodate
from googleapiclient.discovery import build
import argparse
from datetime import datetime
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# define developer key, api service name and api version
DEVELOPER_KEY = "AIzaSyBWnXJwk09QE_XIHdGMTTPYfyS6siyDRS8"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
# define a funtion to get data for a query or topic
def search_result(topic):
    data_list = []  # get all data into a list
    
    # search in youtube for a query
    res = youtube.search().list( q = topic, part="id,snippet", maxResults='50').execute()
    
    # to get next page of data in youtube search
    nextPageToken = res.get('nextPageToken')
    while ('nextPageToken' in res):
        data_list.append(res['items'])
        nextPage = youtube.search().list(q = topic, part="id,snippet", maxResults='50',
                                         pageToken=nextPageToken).execute()
        res['items'] = res['items'] + nextPage['items']
        if 'nextPageToken' not in nextPage:
            res.pop('nextPageToken', None)
        else:
            nextPageToken = nextPage['nextPageToken']
        
        if(len((data_list)) == 7):
            break;
            
    return data_list
new = search_result('math')
new
(new[0][0]['id'])
# get the list of video ids from search result
def video_ids_list(data):
    ids_list = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            if(list(data[i][j]['id'].keys())[1] == 'videoId'):
                ids_list.append(data[i][j]['id']['videoId'])
    return ids_list
# get the part of video by searching in youtube using video ids
def data_columns(video_part, video_ids):
    search_data = []
    for i in range(0, len(video_ids)):
        res = (youtube).videos().list(id=video_ids[i], part=video_part).execute()
        search_data += res['items']
    return search_data
# Define a function to make a csv file which contain video title, duration, topic, published at, liked, dislike,
# view, favorite count ect as columns.
def csv_file(video_list, topic):
    
    # get data column topic
    Topic = [topic]*len(video_list)
    
    # get content Details
    contentDetails = data_columns('contentDetails', video_list)
    dur = []       # Duration of video
    for i in range(len(video_list)):
        sec = isodate.parse_duration(contentDetails[i]['contentDetails']['duration']).seconds
        dur.append((sec % 3600) // 60)
    
    
    snippets = data_columns('snippet', video_list)
    Video_Title = []                 # video title 
    Published_At = []                # publish at
    for i in range(len(video_list)):
        dt = datetime.strptime(snippets[i]['snippet']['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
        Published_At.append(str(dt))
        Video_Title.append(snippets[i]['snippet']['title'])
    
    
    stats = data_columns('statistics', video_list)
    title=[ ]        # video title
    liked=[ ]        # likes on video
    disliked=[ ]     # dislike on video
    favorites =[ ]   # favorites vote on video
    views=[ ]        # total view on video
    url=[ ]          # video title
    comment=[ ]   # video title
    timequried = []
    for i in range(len(video_list)):
        #title.append((videos[i])['snippet']['title'])
        #url.append("https://www.youtube.com/watch?v="+(videos[i])['snippet']['resourceId']['videoId'])
        timequried.append(str(datetime.now().time()))
        if 'viewCount' in (list(stats[i]['statistics'].keys())):
            views.append(int((stats[i])['statistics']['viewCount']))
        else:
            views.append(0)
        
        if 'likeCount' in (list(stats[i]['statistics'].keys())):
            liked.append(int((stats[i])['statistics']['likeCount']))
        else:
            liked.append(0)
        
        if 'likeCount' in (list(stats[i]['statistics'].keys())):
            disliked.append(int((stats[i])['statistics']['dislikeCount']))
        else:
            disliked.append(0)
    
        if 'commentCount' in (list(stats[i]['statistics'].keys())):
            comment.append(int((stats[i])['statistics']['commentCount']))
        else:
            comment.append(0)
        
        if 'favoriteCount' in (list(stats[i]['statistics'].keys())):
            favorites.append(int((stats[i])['statistics']['favoriteCount']))
        else:
            favorites.append(0)
    
    
    # make a csv file
    data = {'Topic':Topic, 'Timequried':timequried, 'Video ID':video_list, 'Video_Title':Video_Title,'Published_At':Published_At,'duration(m)':dur,
            'favorite':favorites,'liked':liked,'disliked':disliked,'views':views,'comment':comment}
    df=pd.DataFrame(data)
    return df
math_data = search_result('math')
math_video_list = video_ids_list(math_data)
math_video_list
len(math_video_list)
math_csv = csv_file(math_video_list, 'math')
math_csv.to_csv("math.csv", index=False)
math_csv
math_csv.info()
Enginner_data = search_result('engineering')
Enginner_video_list = video_ids_list(Enginner_data)
Enginner_video_list
engineer_csv = csv_file(Enginner_video_list, 'Enginnering')
engineer_csv.to_csv("engineer.csv", index=False)
engineer_csv
engineer_csv.to_csv("engineer.csv", index=False)
engineer_csv.info()
science_data = search_result('science')
science_list = video_ids_list(science_data)
science_list
len(science_list)
science_csv = csv_file(science_list, 'science')
science_csv.to_csv("science.csv", index=False)
science_csv
science_csv.to_csv("science.csv", index=False)
science_csv.info()
dataframe = pd.concat([math_csv, engineer_csv, science_csv])
dataframe.to_csv('solution.csv', index=False)
dataframe.head()
dataframe.info()
dataframe.describe()
dataframe.isnull().sum()
df = dataframe.drop(['Published_At', 'Timequried', 'Topic', 'Video ID', 'Video_Title', 'favorite'], axis=1)
df_zscore = (df - df.mean())/df.std()
df_zscore
sns.pairplot(df_zscore)
dataframe.hist(alpha=0.8, figsize=(8,6))
plt.tight_layout()
plt.show()
import seaborn as sns
sns.distplot(dataframe['liked'])
sns.distplot(dataframe['disliked'])
sns.distplot(dataframe['comment'])
matrix = dataframe.corr()
f, ax = plt.subplots(figsize=(9,6))
sns.heatmap(matrix, vmax=1, square=True, cmap='BuPu')
dataframe.rank(method ='average') 
dataset = dataframe.drop(['Published_At','Timequried','Topic','Video ID','Video_Title','favorite',
                          'duration(m)','disliked','liked'], axis=1)
dataset
dataset['total_like'] = dataframe['liked'] - dataframe['disliked']
dataset
dataset['totel_point'] = dataset['views'] + dataset['comment'] +  dataset['total_like']
dataset
percent_rank = dataset.rank(method ='average', pct=True) 
percent_rank
percent_rank.max()
def find_rank(value):
    ans = 0
    if(value<=0.2):
        ans = 1
    elif(0.2<value<=0.4):
        ans = 2
    elif(0.4<value<=0.6):
        ans = 3
    elif(0.6<value<=0.8):
        ans = 4
    elif(0.8<value<=1):
        ans = 5
    return ans
list(percent_rank['totel_point'])[1]
rank = []
for i in range(len(dataset['totel_point'])):
    rank.append(find_rank(list(percent_rank['totel_point'])[i]))
rank
dataframe['rank'] = rank
dataframe
dataframe.to_csv("solution.csv", index=False)