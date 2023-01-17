# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



!pip install tweepy

from textblob import TextBlob

import sys, tweepy

import matplotlib.pyplot as plt



import time



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
consumerkey= "6BTTP8DcPwynxqietcTlswnzr"

consumersecret= "dFPOJkeK6Lvas0YiAIlZXqahqIKybgOlvIoKQAg8e7E3Mz8fBu"

accesstoken= "3326102839-amGQNbPwYJIGPzFCWNS2cxitZv2pS51Hng8VDvJ"

accesstokensecret= "0Td9RrB9BNInvi1MNKqcCO76ppfg32V4pTBPcnhhmi1eJ"



auth=tweepy.OAuthHandler(consumerkey,consumersecret)

auth.set_access_token(accesstoken,accesstokensecret)

api=tweepy.API(auth)
searchTerm = "lockdown"

noOfSearchTerms = 100000
#tweets=tweepy.Cursor(api.search,q=searchTerm, wait_on_rate_limit = True, lang="en").items(noOfSearchTerms)
user = []

tweet = []



now = time.time()



try:

    for i in tweepy.Cursor(api.search,q=searchTerm,

                           include_entities=True,

                           monitor_rate_limit=True, 

                           wait_on_rate_limit=True,

                           wait_on_rate_limit_notify = True,

                           retry_count = 10, #retry 5 times

                           retry_delay = 5, #seconds to wait for retry

                           lang="en").items(noOfSearchTerms):

        

        if time.time()-now > 32400:

            break

        try:

            user.append(i.user.screen_name)

            tweet.append(i.text)        

        except:

            #log.error(e)

            c+=1

            if c % 100 == 0:  # first request completed, sleep 5 sec

                time.sleep(5)

except:

    print("",end="")

    

print((time.time()-now)/3600)
d={'user':user, 'tweet':tweet}

df=pd.DataFrame(d)

df.drop_duplicates(subset="tweet", keep='first', inplace=True, ignore_index=True) 
df
df.to_csv('tweets.csv', index=False)