import json
import csv
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

train_json = open('../input/train/train.json', mode='r').read()
train_pyt = json.loads(train_json)
csv_out = open('train_tweets.csv', mode='w') #opens csv file
writer = csv.writer(csv_out) #create the csv writer object
 
fields = ['post_created_at','tweet_id','text','hashtag_texts','hashtag_count','user_mention_name',
         'tweet_display_url','tweet_url_count', 'in_reply_to_user_id','in_reply_to_screen_name', 'user_id_str',
          'user_name', 'user_screen_name','location','profile_description',
          'followers_count', 'friends_count', 'listed_counts', 'acct_created_at', 'favourites_count','time_zone', 'geo_enabled',
         'verified', 'statuses_count', 'language', 'is_traslation_enabled','default_profile', 'following','follow_request_sent',
         'notification', 'translator_type','geo', 'coordinates','place','is_quote_status','retweet_count',
          'favorite_count', 'favorited','retweeted', 'post_lang' ] #fields
writer.writerow(fields) #writes field
for line in train_pyt:
    #writes a row and gets the fields from the json object
    #screen_name and followers/friends are found on the second level hence two get methods
    writer.writerow([line.get('created_at'),
                     line.get('id'),
                     line.get('text').encode('unicode_escape'), #unicode escape to fix emoji issue
                     line.get('entities').get('hashtags'),
                     len(line.get('entities').get('hashtags')),
                     line.get('entities').get('user_mentions'),
                     line.get('entities').get('urls'),
                     len(line.get('entities').get('urls')),
                     line.get('in_reply_to_status_id_str'),
                     line.get('in_reply_to_screen_name'),
                     line.get('user').get('id_str'),
                     line.get('user').get('name'),
                     line.get('user').get('screen_name'),
                     line.get('user').get('location'),
                     line.get('user').get('description').encode('unicode_escape'),
                     line.get('user').get('followers_count'),
                     line.get('user').get('friends_count'),
                     line.get('user').get('listed_count'),
                     line.get('user').get('created_at'),
                     line.get('user').get('favorites_count'),
                    line.get('user').get('time_zone'),
                    line.get('user').get('geo_enabled'),
                    line.get('user').get('verified'),
                    line.get('user').get('statuses_count'),
                    line.get('user').get('lang'),
                    line.get('user').get('is_traslation_enabled'),
                    line.get('user').get('default_profile'),
                    line.get('user').get('following'),
                    line.get('user').get('follow_request_sent'),
                    line.get('user').get('notification'),
                    line.get('user').get('translator_type'),
                    line.get('geo'),
                    line.get('coordinates'),
                    line.get('place'),
                    line.get('is_quote_status'),
                    line.get('retweet_count'),
                    line.get('favorite_count'),
                    line.get('favorited'),
                    line.get('retweeted'),
                    line.get('lang'),
                    ])

csv_out.close()
line.get('text').encode('unicode_escape'), #unicode escape to fix emoji issue
#Test data
test_json = open('../input/test_questions/test_questions.json', mode='r').read()
test_pyt = json.loads(test_json)

test_csv_out = open('test_tweets.csv', mode='w') #opens csv file
writer2 = csv.writer(test_csv_out) #create the csv writer object

writer2.writerow(fields) #writes field
for line in test_pyt:
     writer2.writerow([line.get('created_at'),
                     line.get('id'),
                     line.get('text').encode('unicode_escape'), #unicode escape to fix emoji issue
                     line.get('entities').get('hashtags'),
                     len(line.get('entities').get('hashtags')),
                     line.get('entities').get('user_mentions'),
                     line.get('entities').get('urls'),
                     len(line.get('entities').get('urls')),
                     line.get('in_reply_to_status_id_str'),
                     line.get('in_reply_to_screen_name'),
                     line.get('user').get('id_str'),
                     line.get('user').get('name'),
                     line.get('user').get('screen_name'),
                     line.get('user').get('location'),
                     line.get('user').get('description').encode('unicode_escape'),
                     line.get('user').get('followers_count'),
                     line.get('user').get('friends_count'),
                     line.get('user').get('listed_count'),
                     line.get('user').get('created_at'),
                     line.get('user').get('favorites_count'),
                    line.get('user').get('time_zone'),
                    line.get('user').get('geo_enabled'),
                    line.get('user').get('verified'),
                    line.get('user').get('statuses_count'),
                    line.get('user').get('lang'),
                    line.get('user').get('is_traslation_enabled'),
                    line.get('user').get('default_profile'),
                    line.get('user').get('following'),
                    line.get('user').get('follow_request_sent'),
                    line.get('user').get('notification'),
                    line.get('user').get('translator_type'),
                    line.get('geo'),
                    line.get('coordinates'),
                    line.get('place'),
                    line.get('is_quote_status'),
                    line.get('retweet_count'),
                    line.get('favorite_count'),
                    line.get('favorited'),
                    line.get('retweeted'),
                    line.get('lang'),
                    ])

test_csv_out.close()

line.get('text').encode('unicode_escape'), #unicode escape to fix emoji issue

