import tweepy

from textblob import TextBlob

import nltk

nltk.download

import datetime

import xlsxwriter



consumer_key = "###############"

consumer_secret = "###############"

access_token = "###############"

access_token_secret = "###############"



auth = tweepy.OAuthHandler(consumer_key,consumer_secret)

auth.set_access_token(access_token,access_token_secret)



tweets = []

target = io.open("mytweets.txt", 'w', encoding='utf-8')

target2 = io.open("sarcasm.xlsx", 'w', encoding='utf-8')

api = tweepy.API(auth)

startDate = datetime.datetime(2016, 6, 1, 0, 0, 0)

endDate = datetime.datetime(2017, 1, 1, 0, 0, 0)



count=200

a=str('জ্যাম')

b=str('বাংলাদেশ')

c=str('এর চেয়ে')

public_tweets = api.search(a, count = count)+ api.search(b, count = count)+ api.search(c, count = count)

#public_tweets = tweepy.Cursor(api.search('#sarcasm', )

workbook = xlsxwriter.Workbook(a + ".xlsx")

worksheet = workbook.add_worksheet()

row = 0



for tweet in public_tweets:

    

        analysis=TextBlob(tweet.text)

        print(analysis)

        print('\n')

        

        #for saving .text format

        tweets.append(tweet)

        parsed_tweet['text'] = analysis

        if "http" not in tweet.text:

            line = re.sub("[^A-Za-z]", " ", tweet.text)

            target.write(line+"\n")



worksheet.write_string(row, 0, str(tweet.id))

worksheet.write_string(row, 1, str(tweet.created_at))

worksheet.write(row, 2, tweet.text)

worksheet.write_string(row, 3, str(tweet.in_reply_to_status_id))

row += 1



workbook.close()



import tweepy

from textblob import TextBlob

import nltk

nltk.download

import datetime

import xlsxwriter



consumer_key = "###############"

consumer_secret = "###############"

access_token = "###############"

access_token_secret = "###############"



auth = tweepy.OAuthHandler(consumer_key,consumer_secret)

auth.set_access_token(access_token,access_token_secret)



tweets = []

target = io.open("mytweets.txt", 'w', encoding='utf-8')

target2 = io.open("sarcasm.xlsx", 'w', encoding='utf-8')

api = tweepy.API(auth)





count=200

a=str('sarcasm')

b=str('sarcastic')

c=str('irony')

public_tweets = api.search(a, count = count)+ api.search(b, count = count)+ api.search(c, count = count)

#public_tweets = tweepy.Cursor(api.search('#sarcasm', )

workbook = xlsxwriter.Workbook(a + ".xlsx")

worksheet = workbook.add_worksheet()

row = 0



for tweet in public_tweets:

    

    if tweet.lang == "en":

        analysis=TextBlob(tweet.text)

        print(analysis)

        print('\n')

        tweets.append(tweet)

        parsed_tweet['text'] = analysis

        if "http" not in tweet.text:

            line = re.sub("[^A-Za-z]", " ", tweet.text)

            target.write(line+"\n")



    worksheet.write_string(row, 0, str(tweet.id))

    worksheet.write_string(row, 1, str(tweet.created_at))

    worksheet.write(row, 2, tweet.text)

    worksheet.write_string(row, 3, str(tweet.in_reply_to_status_id))

    row += 1



workbook.close()

            

            