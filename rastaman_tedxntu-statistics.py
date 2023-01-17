import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime

import seaborn as sns
api_key="AIzaSyDvn1q21oQsOj4z9M9D3dA6mwNTppklKnw"
from apiclient.discovery import build
youtube = build('youtube','v3', developerKey=api_key)
#bef2014=youtube.search().list(q='TEDxNTU', part='snippet', type='video', channelId='UCsT0YIqwnpJCM-mx7-gSA4Q', maxResults = 50, order ='viewCount',publishedBefore = '2014-08-15T00:00:00-07:00').execute()

#aft2014=youtube.search().list(q='TEDxNTU', part='snippet', type='video', channelId='UCsT0YIqwnpJCM-mx7-gSA4Q', maxResults = 50, order ='viewCount',publishedAfter = '2014-08-15T00:00:00-07:00').execute()
bef_2012=youtube.search().list(q='TEDxNTU', part='snippet', type='video', channelId='UCsT0YIqwnpJCM-mx7-gSA4Q', maxResults = 50, order ='viewCount',publishedBefore = '2012-08-15T00:00:00-07:00').execute()

bet_2012_2013=youtube.search().list(q='TEDxNTU', part='snippet', type='video', channelId='UCsT0YIqwnpJCM-mx7-gSA4Q', maxResults = 50, order ='viewCount',publishedAfter = '2012-08-15T00:00:00-07:01',publishedBefore = '2013-08-15T00:00:00-07:00').execute()

bet_2013_2014=youtube.search().list(q='TEDxNTU', part='snippet', type='video', channelId='UCsT0YIqwnpJCM-mx7-gSA4Q', maxResults = 50, order ='viewCount',publishedAfter = '2013-08-15T00:00:00-07:01',publishedBefore = '2014-08-15T00:00:00-07:00').execute()

bet_2014_2015=youtube.search().list(q='TEDxNTU', part='snippet', type='video', channelId='UCsT0YIqwnpJCM-mx7-gSA4Q', maxResults = 50, order ='viewCount',publishedAfter = '2014-08-15T00:00:00-07:01',publishedBefore = '2015-08-15T00:00:00-07:00').execute()

bet_2015_2016=youtube.search().list(q='TEDxNTU', part='snippet', type='video', channelId='UCsT0YIqwnpJCM-mx7-gSA4Q', maxResults = 50, order ='viewCount',publishedAfter = '2015-08-15T00:00:00-07:01',publishedBefore = '2016-08-15T00:00:00-07:00').execute()

bet_2016_2017=youtube.search().list(q='TEDxNTU', part='snippet', type='video', channelId='UCsT0YIqwnpJCM-mx7-gSA4Q', maxResults = 50, order ='viewCount',publishedAfter = '2016-08-15T00:00:00-07:01',publishedBefore = '2017-08-15T00:00:00-07:00').execute()

bet_2017_2018=youtube.search().list(q='TEDxNTU', part='snippet', type='video', channelId='UCsT0YIqwnpJCM-mx7-gSA4Q', maxResults = 50, order ='viewCount',publishedAfter = '2017-08-15T00:00:00-07:01',publishedBefore = '2018-08-15T00:00:00-07:00').execute()

bet_2018_2019=youtube.search().list(q='TEDxNTU', part='snippet', type='video', channelId='UCsT0YIqwnpJCM-mx7-gSA4Q', maxResults = 50, order ='viewCount',publishedAfter = '2018-08-15T00:00:00-07:01',publishedBefore = '2019-08-15T00:00:00-07:00').execute()

bet_2019_2020=youtube.search().list(q='TEDxNTU', part='snippet', type='video', channelId='UCsT0YIqwnpJCM-mx7-gSA4Q', maxResults = 50, order ='viewCount',publishedAfter = '2019-08-15T00:00:00-07:01',publishedBefore = '2020-08-15T00:00:00-07:00').execute()

bef_2012['items']
title=[]

description=[]

url=[]

url_pt1="https://www.youtube.com/watch?v="

published_year=[]

viewcount=[]

likeCount=[]

dislikeCount=[]

favoriteCount=[]

commentCount=[]





for item in bef_2012['items']:

    

    name=item['snippet']['title']

    title.append(name)

    

    desc=item['snippet']['description']

    description.append(desc)

    

    

    

    urls_pt2 = item['id']['videoId']

    url_full=url_pt1 + urls_pt2

    url.append(url_full)

    

    

    published_in_=item['snippet']['publishedAt'][0:4]

    published_year.append(published_in_)

    

    

    res_tmp=youtube.videos().list(id=item['id']['videoId'],part='statistics').execute()

    

    viewcount_=int(res_tmp['items'][0]['statistics']['viewCount'])

    viewcount.append(viewcount_)

    #print(name)

    #print(res_tmp['items'][0]['statistics'])

    

    try:

        likeCount_=int(res_tmp['items'][0]['statistics']['likeCount'])

        likeCount.append(likeCount_)

    except KeyError:

        likeCount.append(np.nan)

    

    try:

        dislikeCount_=int(res_tmp['items'][0]['statistics']['dislikeCount'])

        dislikeCount.append(dislikeCount_)

    except KeyError:

        dislikeCount.append(np.nan)

        

    try:

        favoriteCount_=int(res_tmp['items'][0]['statistics']['favoriteCount'])

        favoriteCount.append(favoriteCount_)

    except KeyError:

        favoriteCount.append(np.nan)        



    try:

        commentCount_=int(res_tmp['items'][0]['statistics']['commentCount'])

        commentCount.append(commentCount_)

    except KeyError:

        commentCount.append(np.nan)

        





for item in bet_2012_2013['items']:

    

    name=item['snippet']['title']

    title.append(name)

    

    desc=item['snippet']['description']

    description.append(desc)

    

    urls_pt2 = item['id']['videoId']

    url_full=url_pt1 + urls_pt2

    url.append(url_full)

    

    published_in_=item['snippet']['publishedAt'][0:4]

    published_year.append(published_in_)

    

    res_tmp=youtube.videos().list(id=item['id']['videoId'],part='statistics').execute()

    

    viewcount_=int(res_tmp['items'][0]['statistics']['viewCount'])

    viewcount.append(viewcount_)

 

    try:

        likeCount_=int(res_tmp['items'][0]['statistics']['likeCount'])

        likeCount.append(likeCount_)

    except KeyError:

        likeCount.append(np.nan)

    

    try:

        dislikeCount_=int(res_tmp['items'][0]['statistics']['dislikeCount'])

        dislikeCount.append(dislikeCount_)

    except KeyError:

        dislikeCount.append(np.nan)

        

    try:

        favoriteCount_=int(res_tmp['items'][0]['statistics']['favoriteCount'])

        favoriteCount.append(favoriteCount_)

    except KeyError:

        favoriteCount.append(np.nan)        



    try:

        commentCount_=int(res_tmp['items'][0]['statistics']['commentCount'])

        commentCount.append(commentCount_)

    except KeyError:

        commentCount.append(np.nan)



        

for item in bet_2013_2014['items']:

    

    name=item['snippet']['title']

    title.append(name)

    

    desc=item['snippet']['description']

    description.append(desc)

    

    urls_pt2 = item['id']['videoId']

    url_full=url_pt1 + urls_pt2

    url.append(url_full)

    

    published_in_=item['snippet']['publishedAt'][0:4]

    published_year.append(published_in_)

    

    res_tmp=youtube.videos().list(id=item['id']['videoId'],part='statistics').execute()

    

    viewcount_=int(res_tmp['items'][0]['statistics']['viewCount'])

    viewcount.append(viewcount_)

 

    try:

        likeCount_=int(res_tmp['items'][0]['statistics']['likeCount'])

        likeCount.append(likeCount_)

    except KeyError:

        likeCount.append(np.nan)

    

    try:

        dislikeCount_=int(res_tmp['items'][0]['statistics']['dislikeCount'])

        dislikeCount.append(dislikeCount_)

    except KeyError:

        dislikeCount.append(np.nan)

        

    try:

        favoriteCount_=int(res_tmp['items'][0]['statistics']['favoriteCount'])

        favoriteCount.append(favoriteCount_)

    except KeyError:

        favoriteCount.append(np.nan)        



    try:

        commentCount_=int(res_tmp['items'][0]['statistics']['commentCount'])

        commentCount.append(commentCount_)

    except KeyError:

        commentCount.append(np.nan)



for item in bet_2014_2015['items']:

    

    name=item['snippet']['title']

    title.append(name)

    

    desc=item['snippet']['description']

    description.append(desc)

    

    urls_pt2 = item['id']['videoId']

    url_full=url_pt1 + urls_pt2

    url.append(url_full)

    

    published_in_=item['snippet']['publishedAt'][0:4]

    published_year.append(published_in_)

    

    res_tmp=youtube.videos().list(id=item['id']['videoId'],part='statistics').execute()

    

    viewcount_=int(res_tmp['items'][0]['statistics']['viewCount'])

    viewcount.append(viewcount_)

 

    try:

        likeCount_=int(res_tmp['items'][0]['statistics']['likeCount'])

        likeCount.append(likeCount_)

    except KeyError:

        likeCount.append(np.nan)

    

    try:

        dislikeCount_=int(res_tmp['items'][0]['statistics']['dislikeCount'])

        dislikeCount.append(dislikeCount_)

    except KeyError:

        dislikeCount.append(np.nan)

        

    try:

        favoriteCount_=int(res_tmp['items'][0]['statistics']['favoriteCount'])

        favoriteCount.append(favoriteCount_)

    except KeyError:

        favoriteCount.append(np.nan)        



    try:

        commentCount_=int(res_tmp['items'][0]['statistics']['commentCount'])

        commentCount.append(commentCount_)

    except KeyError:

        commentCount.append(np.nan)



for item in bet_2015_2016['items']:

    

    name=item['snippet']['title']

    title.append(name)

    

    desc=item['snippet']['description']

    description.append(desc)

    

    urls_pt2 = item['id']['videoId']

    url_full=url_pt1 + urls_pt2

    url.append(url_full)    

    

    published_in_=item['snippet']['publishedAt'][0:4]

    published_year.append(published_in_)

    

    res_tmp=youtube.videos().list(id=item['id']['videoId'],part='statistics').execute()

    

    viewcount_=int(res_tmp['items'][0]['statistics']['viewCount'])

    viewcount.append(viewcount_)

 

    try:

        likeCount_=int(res_tmp['items'][0]['statistics']['likeCount'])

        likeCount.append(likeCount_)

    except KeyError:

        likeCount.append(np.nan)

    

    try:

        dislikeCount_=int(res_tmp['items'][0]['statistics']['dislikeCount'])

        dislikeCount.append(dislikeCount_)

    except KeyError:

        dislikeCount.append(np.nan)

        

    try:

        favoriteCount_=int(res_tmp['items'][0]['statistics']['favoriteCount'])

        favoriteCount.append(favoriteCount_)

    except KeyError:

        favoriteCount.append(np.nan)        



    try:

        commentCount_=int(res_tmp['items'][0]['statistics']['commentCount'])

        commentCount.append(commentCount_)

    except KeyError:

        commentCount.append(np.nan)



   

for item in bet_2016_2017['items']:

    

    name=item['snippet']['title']

    title.append(name)

    

    desc=item['snippet']['description']

    description.append(desc)

    

    urls_pt2 = item['id']['videoId']

    url_full=url_pt1 + urls_pt2

    url.append(url_full)

    

    published_in_=item['snippet']['publishedAt'][0:4]

    published_year.append(published_in_)

    

    res_tmp=youtube.videos().list(id=item['id']['videoId'],part='statistics').execute()

    

    viewcount_=int(res_tmp['items'][0]['statistics']['viewCount'])

    viewcount.append(viewcount_)

 

    try:

        likeCount_=int(res_tmp['items'][0]['statistics']['likeCount'])

        likeCount.append(likeCount_)

    except KeyError:

        likeCount.append(np.nan)

    

    try:

        dislikeCount_=int(res_tmp['items'][0]['statistics']['dislikeCount'])

        dislikeCount.append(dislikeCount_)

    except KeyError:

        dislikeCount.append(np.nan)

        

    try:

        favoriteCount_=int(res_tmp['items'][0]['statistics']['favoriteCount'])

        favoriteCount.append(favoriteCount_)

    except KeyError:

        favoriteCount.append(np.nan)        



    try:

        commentCount_=int(res_tmp['items'][0]['statistics']['commentCount'])

        commentCount.append(commentCount_)

    except KeyError:

        commentCount.append(np.nan)



        

for item in bet_2017_2018['items']:

    

    name=item['snippet']['title']

    title.append(name)

    

    desc=item['snippet']['description']

    description.append(desc)

    

    urls_pt2 = item['id']['videoId']

    url_full=url_pt1 + urls_pt2

    url.append(url_full)

    

    published_in_=item['snippet']['publishedAt'][0:4]

    published_year.append(published_in_)

    

    res_tmp=youtube.videos().list(id=item['id']['videoId'],part='statistics').execute()

    

    viewcount_=int(res_tmp['items'][0]['statistics']['viewCount'])

    viewcount.append(viewcount_)

 

    try:

        likeCount_=int(res_tmp['items'][0]['statistics']['likeCount'])

        likeCount.append(likeCount_)

    except KeyError:

        likeCount.append(np.nan)

    

    try:

        dislikeCount_=int(res_tmp['items'][0]['statistics']['dislikeCount'])

        dislikeCount.append(dislikeCount_)

    except KeyError:

        dislikeCount.append(np.nan)

        

    try:

        favoriteCount_=int(res_tmp['items'][0]['statistics']['favoriteCount'])

        favoriteCount.append(favoriteCount_)

    except KeyError:

        favoriteCount.append(np.nan)        



    try:

        commentCount_=int(res_tmp['items'][0]['statistics']['commentCount'])

        commentCount.append(commentCount_)

    except KeyError:

        commentCount.append(np.nan)



for item in bet_2018_2019['items']:

    

    name=item['snippet']['title']

    title.append(name)

    

    desc=item['snippet']['description']

    description.append(desc)

    

    urls_pt2 = item['id']['videoId']

    url_full=url_pt1 + urls_pt2

    url.append(url_full)

    

    published_in_=item['snippet']['publishedAt'][0:4]

    published_year.append(published_in_)

    

    res_tmp=youtube.videos().list(id=item['id']['videoId'],part='statistics').execute()

    

    viewcount_=int(res_tmp['items'][0]['statistics']['viewCount'])

    viewcount.append(viewcount_)

 

    try:

        likeCount_=int(res_tmp['items'][0]['statistics']['likeCount'])

        likeCount.append(likeCount_)

    except KeyError:

        likeCount.append(np.nan)

    

    try:

        dislikeCount_=int(res_tmp['items'][0]['statistics']['dislikeCount'])

        dislikeCount.append(dislikeCount_)

    except KeyError:

        dislikeCount.append(np.nan)

        

    try:

        favoriteCount_=int(res_tmp['items'][0]['statistics']['favoriteCount'])

        favoriteCount.append(favoriteCount_)

    except KeyError:

        favoriteCount.append(np.nan)        



    try:

        commentCount_=int(res_tmp['items'][0]['statistics']['commentCount'])

        commentCount.append(commentCount_)

    except KeyError:

        commentCount.append(np.nan)



        

        

        

for item in bet_2019_2020['items']:

    

    name=item['snippet']['title']

    title.append(name)

    

    desc=item['snippet']['description']

    description.append(desc)

    

    urls_pt2 = item['id']['videoId']

    url_full=url_pt1 + urls_pt2

    url.append(url_full)

    

    published_in_=item['snippet']['publishedAt'][0:4]

    published_year.append(published_in_)

    

    res_tmp=youtube.videos().list(id=item['id']['videoId'],part='statistics').execute()

    

    viewcount_=int(res_tmp['items'][0]['statistics']['viewCount'])

    viewcount.append(viewcount_)

 

    try:

        likeCount_=int(res_tmp['items'][0]['statistics']['likeCount'])

        likeCount.append(likeCount_)

    except KeyError:

        likeCount.append(np.nan)

    

    try:

        dislikeCount_=int(res_tmp['items'][0]['statistics']['dislikeCount'])

        dislikeCount.append(dislikeCount_)

    except KeyError:

        dislikeCount.append(np.nan)

        

    try:

        favoriteCount_=int(res_tmp['items'][0]['statistics']['favoriteCount'])

        favoriteCount.append(favoriteCount_)

    except KeyError:

        favoriteCount.append(np.nan)        



    try:

        commentCount_=int(res_tmp['items'][0]['statistics']['commentCount'])

        commentCount.append(commentCount_)

    except KeyError:

        commentCount.append(np.nan)

                

        

df=pd.DataFrame()

df['title']=title

df['description']=description

df['url']=url

df['published_year']=published_year

df['viewcount']=viewcount

df['likeCount']=likeCount

df['dislikeCount']=dislikeCount

df['favoriteCount']=favoriteCount

df['commentCount']=commentCount



df.sort_values(by='viewcount',inplace=True,ascending=False)

df.reset_index(drop=True,inplace = True)





print("Done")
df.to_csv("details.csv", index=False)

print("Total Videos found = " +str(df.shape[0]))
df.head(10)
Total_Count=np.sum(viewcount)

print("Total_Count = "+str(Total_Count))
#https://www.youtube.com/watch?v=U39wbNMtMVw&t=23s