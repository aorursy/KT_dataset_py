%%capture

!pip install google-api-python-client
import json

import csv

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





from apiclient.discovery import build



from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("api_key") # Use your youtube api key

secret_value_1 = user_secrets.get_secret("channel_id") # Use your youtube channel ID
## Get upload playlist id

yt = build('youtube', 'v3', developerKey= secret_value_0)

req = yt.channels().list(id= secret_value_1, part= 'contentDetails').execute()



def get_channel_videos(secret_value_1):

    # get Uploads playlist id

    res = yt.channels().list(id=secret_value_1, 

                                  part='contentDetails').execute()

    playlist_id = res['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    

    videos = []

    next_page_token = None

    

    while 1:

        res = yt.playlistItems().list(playlistId=playlist_id, 

                                           part='snippet', 

                                           maxResults=50,

                                           pageToken=next_page_token).execute()

        videos += res['items']

        next_page_token = res.get('nextPageToken')

        

        if next_page_token is None:

            break

    

    return videos



videos = get_channel_videos(secret_value_1)

print(f'Total number of video are: {len(videos)}')



## get all video from youtube channel in json file

all_Yt_Details = []



for i, video in enumerate(videos):

    ytDetails = {

        "ID" : i+1,

        "Video_ID" : video['snippet']['resourceId']['videoId'],

        "URL" : f"https://www.youtube.com/watch?v={video['snippet']['resourceId']['videoId']}",

        "Title" : video['snippet']['title'],

        "Published Date": video['snippet']['publishedAt'].split("T")[0]



    }

    all_Yt_Details.append(ytDetails)



with open('youtube_data.json', 'w') as f:

    json.dump(all_Yt_Details, f, indent=4)



## get all video from youtube channel in excel file

data = pd.read_json('/kaggle/working/youtube_data.json')

data.to_csv('youtube_data.csv', index= False)



print("File is downloaded")