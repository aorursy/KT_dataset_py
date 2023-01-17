import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# install the following packages in your local notebook



#!pip install google_auth_oauthlib

#!pip install googleapiclient
import os

from googleapiclient.discovery import build

from google_auth_oauthlib.flow import InstalledAppFlow



# The CLIENT_SECRETS_FILE variable specifies the name of a file that contains

# the OAuth 2.0 information for this application, including its client_id and

# client_secret.







###########################################

CLIENT_SECRETS_FILE = "client_secret.json" #This is the name of your JSON file, which should be store locally in a folder. This is your access credential file the fucntion is trying to call

###########################################

# Here is an example client_secrets.json

# here is not it should look like once you open it

#{

#  "installed": {

#    "client_id": "837647042410-75ifg...usercontent.com",

#    "client_secret":"asdlkfjaskd",

#    "redirect_uris": ["http://localhost", "urn:ietf:wg:oauth:2.0:oob"],

#    "auth_uri": "https://accounts.google.com/o/oauth2/auth",

#    "token_uri": "https://accounts.google.com/o/oauth2/token"

#  }

#}







# This OAuth 2.0 access scope allows for full read/write access to the

# authenticated user's account and requires requests to use an SSL connection.

SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']

API_SERVICE_NAME = 'youtube'

API_VERSION = 'v3'



def get_authenticated_service():

  flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)

  credentials = flow.run_console()

  return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)



os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

service = get_authenticated_service()
query = 'Donald Trump impeachment' #querying subject, same as advanced search option in Youtube



query_results = service.search().list(

        part = 'snippet',

        q = query,

#       channelId='string', if you want to query a specific channel

#        videoDuration='any', # long/med/short

        #location='string', specify location of the origin video

        #locationRadius='string', the serach radius of the origin videos

        order = 'relevance', # You can consider using viewCount

        maxResults = 8,

        type = 'video', # Channels might appear in search results

        relevanceLanguage = 'en',

        safeSearch = 'moderate',

    

        ).execute()
# Get Video IDs setup



video_id = []

channel = []

video_title = []

video_desc = []

for item in query_results['items']:

    video_id.append(item['id']['videoId'])

    channel.append(item['snippet']['channelTitle'])

    video_title.append(item['snippet']['title'])

    video_desc.append(item['snippet']['description'])
# Retrive top comments from vide_title



video_id_top=[]

channel_top = []

video_title_top = []

video_desc_top = []

comments_top = []

comment_id_top = []

reply_count_top = []

like_count_top = []



from tqdm import tqdm # just a visual process bar



for i, video in enumerate(tqdm(video_id, ncols = 100)):

# digging into reponse can derive even more information about the user that commented on the video

    response = service.commentThreads().list(

                         

                    part = 'snippet',

                    videoId = video,

                    maxResults = 100, # Only take top 100 comments...

                    order = 'relevance', #... ranked on relevance

                    textFormat = 'plainText',

                    ).execute()

    

    comments_temp = []

    comment_id_temp = []

    reply_count_temp = []

    like_count_temp = []

    for item in response['items']:

        comments_temp.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])

        comment_id_temp.append(item['snippet']['topLevelComment']['id'])

        reply_count_temp.append(item['snippet']['totalReplyCount'])

        like_count_temp.append(item['snippet']['topLevelComment']['snippet']['likeCount'])

    comments_top.extend(comments_temp)

    comment_id_top.extend(comment_id_temp)

    reply_count_top.extend(reply_count_temp)

    like_count_top.extend(like_count_temp)

    

    video_id_top.extend([video_id[i]]*len(comments_temp))

    channel_top.extend([channel[i]]*len(comments_temp))

    video_title_top.extend([video_title[i]]*len(comments_temp))

    video_desc_top.extend([video_desc[i]]*len(comments_temp))

    

query_top = [query] * len(video_id_top)
# Read into Dataframe



import pandas as pd



output_dict = {

        'Query': query_top,

        'Channel': channel_top,

        'Video Title': video_title_top,

        'Video Description': video_desc_top,

        'Video ID': video_id_top,

        'Comment': comments_top,

        'Comment ID': comment_id_top,

        'Replies': reply_count_top,

        'Likes': like_count_top,

        }



output_df = pd.DataFrame(output_dict, columns = output_dict.keys())