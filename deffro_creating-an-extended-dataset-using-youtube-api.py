import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")

print("Train has {} samples {} features. Only {}? Hah, this is going to change soon.".format(train.shape[0],train.shape[1], train.shape[1]))
train.head()
plt.figure(figsize=(10,6))
sns.countplot(train['is_turkey'].sort_values())
train['durationOfClip'] = train['end_time_seconds_youtube_clip']-train['start_time_seconds_youtube_clip']
plt.figure(figsize=(10,6))
sns.countplot(train['durationOfClip'])
plt.show()
import requests
import json
from apiclient.discovery import build
from apiclient.errors import HttpError

DEVELOPER_KEY = "AIzaSyDS6cbdzv7XhBnPuGOOQQGb_9EHahM9nec"

url = "https://www.googleapis.com/youtube/v3/videos?id={id}&key={api_key}&part=snippet,contentDetails,statistics,status,player,recordingDetails"

train['publishedAt'] = '' #The date and time that the video was published.
train['channelId'] = '' #The ID that YouTube uses to uniquely identify the channel that the video was uploaded to.
train['title'] = '' #The video's title.
train['description'] = '' #The video's description.
train['channelTitle'] = '' #Channel title for the channel that the video belongs to.
train['tags'] = '' #A list of keyword tags associated with the video.
train['categoryId'] = '' #The YouTube video category associated with the video. 
train['liveBroadcastContent'] = '' #Indicates if the video is an upcoming/active live broadcast. Or it's "none" if the video is not an upcoming/active live broadcast.
train['defaultLanguage'] = '' #The language of the text in the video.
train['defaultAudioLanguage'] = '' #The language spoken in the video's default audio track.
train['duration'] = '' #The length of the video. The property value is an ISO 8601 duration.
train['dimension'] = '' #Indicates whether the video is available in 3D or in 2D.
train['definition'] = '' #Indicates whether the video is available in high definition (HD) or only in standard definition.
train['caption'] = '' #Indicates whether captions are available for the video.
train['licensedContent'] = '' #Indicates whether the video represents licensed content, which means that the content was uploaded to a channel linked to a YouTube content partner and then claimed by that partner.
train['projection'] = '' #Specifies the projection format of the video.
train['viewCount'] = '' #The number of times the video has been viewed.
train['likeCount'] = '' #The number of users who have indicated that they liked the video.
train['dislikeCount'] = '' #The number of users who have indicated that they disliked the video.
train['commentCount'] = '' #The number of comments for the video.
train['uploadStatus'] = '' #The status of the uploaded video.
train['privacyStatus'] = '' #The video's privacy status.
train['license'] = '' #The video's license.
train['embeddable'] = '' #This value indicates whether the video can be embedded on another website.
train['publicStatsViewable'] = '' #This value indicates whether the extended video statistics on the video's watch page are publicly viewable.


for i, row in train.iterrows():
    _id = train['vid_id'][i]
    r = requests.get(url.format(id=_id, api_key=DEVELOPER_KEY))
    js = r.json()
    try:
        items = js["items"][0]    
        try:
            train.at[i, 'publishedAt'] = items["snippet"]["publishedAt"]
        except Exception:
            train.at[i, 'publishedAt'] = np.nan

        try:
            train.at[i, 'channelId'] = items["snippet"]["channelId"]
        except Exception:
            train.at[i, 'channelId'] = np.nan
            
        try:
            train.at[i, 'title'] = items["snippet"]["title"]
        except Exception:
            train.at[i, 'title'] = np.nan
            
        try:
            train.at[i, 'description'] = items["snippet"]["description"]
        except Exception:
            train.at[i, 'description'] = np.nan            
            
        try:
            train.at[i, 'channelTitle'] = items["snippet"]["channelTitle"]
        except Exception:
            train.at[i, 'channelTitle'] = np.nan            
            
        try:
            train.at[i, 'tags'] = items["snippet"]["tags"]
        except Exception:
            train.at[i, 'tags'] = np.nan
            
        try:
            train.at[i, 'categoryId'] = items["snippet"]["categoryId"]
        except Exception:
            train.at[i, 'categoryId'] = np.nan            
            
        try:
            train.at[i, 'liveBroadcastContent'] = items["snippet"]["liveBroadcastContent"]
        except Exception:
            train.at[i, 'liveBroadcastContent'] = np.nan  
            
        try:
            train.at[i, 'defaultLanguage'] = items["snippet"]["defaultLanguage"]
        except Exception:
            train.at[i, 'defaultLanguage'] = np.nan    
            
        try:
            train.at[i, 'defaultAudioLanguage'] = items["snippet"]["defaultAudioLanguage"]
        except Exception:
            train.at[i, 'defaultAudioLanguage'] = np.nan    
                                  
            
    except IndexError:
        train.at[i, 'publishedAt'] = np.nan
        train.at[i, 'channelId'] = np.nan
        train.at[i, 'title'] = np.nan
        train.at[i, 'description'] = np.nan
        train.at[i, 'channelTitle'] = np.nan
        train.at[i, 'tags'] = np.nan
        train.at[i, 'categoryId'] = np.nan
        train.at[i, 'liveBroadcastContent'] = np.nan
        train.at[i, 'defaultAudioLanguage'] = np.nan
        
    try:
        train.at[i, 'duration'] = items["contentDetails"]["duration"]
    except Exception:
        train.at[i, 'duration'] = np.nan                
            
    try:
        train.at[i, 'dimension'] = items["contentDetails"]["dimension"]
    except Exception:
        train.at[i, 'dimension'] = np.nan                
            
    try:
        train.at[i, 'definition'] = items["contentDetails"]["definition"]
    except Exception:
        train.at[i, 'definition'] = np.nan                
            
    try:
        train.at[i, 'caption'] = items["contentDetails"]["caption"]
    except Exception:
        train.at[i, 'caption'] = np.nan                
            
    try:
        train.at[i, 'licensedContent'] = items["contentDetails"]["licensedContent"]
    except Exception:
        train.at[i, 'licensedContent'] = np.nan                
            
    try:
        train.at[i, 'projection'] = items["contentDetails"]["projection"]
    except Exception:
        train.at[i, 'projection'] = np.nan                       
        
    try:
        train.at[i, 'viewCount'] = items["statistics"]["viewCount"]
    except Exception:
        train.at[i, 'viewCount'] = np.nan                
            
    try:
        train.at[i, 'likeCount'] = items["statistics"]["likeCount"]
    except Exception:
        train.at[i, 'likeCount'] = np.nan                
            
    try:
        train.at[i, 'dislikeCount'] = items["statistics"]["dislikeCount"]
    except Exception:
        train.at[i, 'dislikeCount'] = np.nan                           
            
    try:
        train.at[i, 'commentCount'] = items["statistics"]["commentCount"]
    except Exception:
        train.at[i, 'commentCount'] = np.nan  
        
    try:
        train.at[i, 'uploadStatus'] = items["status"]["uploadStatus"]
    except Exception:
        train.at[i, 'uploadStatus'] = np.nan  
        
    try:
        train.at[i, 'privacyStatus'] = items["status"]["privacyStatus"]
    except Exception:
        train.at[i, 'privacyStatus'] = np.nan  
        
    try:
        train.at[i, 'license'] = items["status"]["license"]
    except Exception:
        train.at[i, 'license'] = np.nan          
        
    try:
        train.at[i, 'embeddable'] = items["status"]["embeddable"]
    except Exception:
        train.at[i, 'embeddable'] = np.nan  
        
    try:
        train.at[i, 'publicStatsViewable'] = items["status"]["publicStatsViewable"]
    except Exception:
        train.at[i, 'publicStatsViewable'] = np.nan          
        
        
train.head()
train.info()