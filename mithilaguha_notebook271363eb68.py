!pip install GetOldTweets3
import GetOldTweets3 as got
import pandas as pd

#option4
import time
from datetime import datetime, date, timedelta

#def get_tweets(username, top_only, start_date, end_date, count):
def get_tweets(username, top_only, start_date, end_date):
    '''
    Downloads all tweets from a certain month in three sessions in order to avoid sending too many requests. 
    Date format = 'yyyy-mm-dd'. timedelta takes in different between two dates. 
    '''
    since = datetime.strptime(start_date, '%Y-%m-%d')
    until= datetime.strptime(end_date, '%Y-%m-%d')
    part1 = since + timedelta(days = 50)
    part2 = since + timedelta(days = 150)
    part3 = since + timedelta(days = 250)
    part4 = since + timedelta(days = 350)
    part5 = since + timedelta(days = 450)
    part6 = since + timedelta(days = 572)
    part7 = since + timedelta(days = 684)
    part8 = since + timedelta(days = 796)
    part9 = since + timedelta(days = 808)
    part10 = since + timedelta(days = 900)
    part11 = since + timedelta(days = 950)
    part12 = since + timedelta(days = 1050)
    part13 = since + timedelta(days = 1100)
    part14 = since + timedelta(days = 1200)
    
    print ('starting first download')
    now = datetime.now()
    print(str(now))
        # specifying tweet search criteria 
    first = got.manager.TweetCriteria().setUsername(username)\
                          .setTopTweets(top_only)\
                          .setSince(since.strftime('%Y-%m-%d'))\
                          .setUntil(part1.strftime('%Y-%m-%d'))\
                         # .setMaxTweets(count)
    
    firstdownload = got.manager.TweetManager.getTweets(first)
    
    # creating list of tweets with the tweet attributes 
    # specified in the list comprehension
    first_list = [[tw.text,
                tw.date,
                tw.retweets,
                tw.favorites,    
                tw.hashtags] for tw in firstdownload]

    
    # creating dataframe, assigning column names to list of
    # tweets corresponding to tweet attributes
    tweets_df_1 = pd.DataFrame(first_list, columns = ['Text','Date','Retweets','favorites','HashTags'])
    tweets_df_1 = tweets_df_1[tweets_df_1.Retweets > 0]
    tweets_df_1 = tweets_df_1[~tweets_df_1['Text'].str.startswith('Hi')]
    tweets_df_1 = tweets_df_1[~tweets_df_1['Text'].str.startswith('Hey')]
    tweets_df_1 = tweets_df_1[~tweets_df_1['Text'].str.startswith('Hello')]
    tweets_df_1.to_csv('{}-1.csv'.format(username, sep=','))
 
    #tweets_df_1.to_csv("%s_1.csv" % start_date)
    print('Finished it.')
    now = datetime.now()
    print(str(now))
    print('Next one in 10 mins.')   
    time.sleep(900)
    
    print ('starting second download')
     # specifying tweet search criteria 
    second = got.manager.TweetCriteria().setUsername(username)\
                          .setTopTweets(top_only)\
                          .setSince(part1.strftime('%Y-%m-%d'))\
                          .setUntil(part2.strftime('%Y-%m-%d'))\
                          #.setMaxTweets(count)
    
    seconddownload = got.manager.TweetManager.getTweets(second)
    
    # creating list of tweets with the tweet attributes 
    # specified in the list comprehension
    second_list = [[tw.text,
                tw.date,
                tw.retweets,
                tw.favorites,    
                tw.hashtags] for tw in seconddownload]

    
    # creating dataframe, assigning column names to list of
    # tweets corresponding to tweet attributes
    tweets_df_2 = pd.DataFrame(second_list, columns = ['Text','Date','Retweets','favorites','HashTags'])
    tweets_df_2 = tweets_df_2[tweets_df_2.Retweets > 0]
    tweets_df_2 = tweets_df_2[~tweets_df_2['Text'].str.startswith('Hi')]
    tweets_df_2 = tweets_df_2[~tweets_df_2['Text'].str.startswith('Hey')]
    tweets_df_2 = tweets_df_2[~tweets_df_2['Text'].str.startswith('Hello')]
    tweets_df_2.to_csv('{}-2.csv'.format(username, sep=','))
 
    #tweets_df_2.to_csv("%s_2.csv" % start_date)
    print('Finished it.')
    now = datetime.now()
    print(str(now))
    print('Next one in 10 mins.')   
    time.sleep(900)
    
    print('starting third download')
        # specifying tweet search criteria 
    third = got.manager.TweetCriteria().setUsername(username)\
                          .setTopTweets(top_only)\
                          .setSince(part2.strftime('%Y-%m-%d'))\
                          .setUntil(part3.strftime('%Y-%m-%d'))\
                          #.setMaxTweets(count)
    
    thirddownload = got.manager.TweetManager.getTweets(third)
    
    # creating list of tweets with the tweet attributes 
    # specified in the list comprehension
    third_list = [[tw.text,
                tw.date,
                tw.retweets,
                tw.favorites,    
                tw.hashtags] for tw in thirddownload]

    
    # creating dataframe, assigning column names to list of
    # tweets corresponding to tweet attributes
    tweets_df_3 = pd.DataFrame(third_list, columns = ['Text','Date','Retweets','favorites','HashTags'])
    tweets_df_3 = tweets_df_3[tweets_df_3.Retweets > 0]
    tweets_df_3 = tweets_df_3[~tweets_df_3['Text'].str.startswith('Hi')]
    tweets_df_3 = tweets_df_3[~tweets_df_3['Text'].str.startswith('Hey')]
    tweets_df_3 = tweets_df_3[~tweets_df_3['Text'].str.startswith('Hello')]
    #tweets_df_1.to_csv('{}-{}k-recenttweets.csv'.format(username, int(count/1000)), sep=',')
    tweets_df_3.to_csv('{}-3.csv'.format(username, sep=','))
    #tweets_df_3.to_csv("%s_3.csv" % start_date)
    print('Finished it.')
    now = datetime.now()
    print(str(now))
    print('Next one in 10 mins.')   
    time.sleep(900)
    
    print('starting fourth download')
        # specifying tweet search criteria 
    fourth = got.manager.TweetCriteria().setUsername(username)\
                          .setTopTweets(top_only)\
                          .setSince(part3.strftime('%Y-%m-%d'))\
                          .setUntil(part4.strftime('%Y-%m-%d'))\
                          #.setMaxTweets(count)
    
    fourthdownload = got.manager.TweetManager.getTweets(fourth)
    
    # creating list of tweets with the tweet attributes 
    # specified in the list comprehension
    fourth_list = [[tw.text,
                tw.date,
                tw.retweets,
                tw.favorites,    
                tw.hashtags] for tw in fourthdownload]

    
    # creating dataframe, assigning column names to list of
    # tweets corresponding to tweet attributes
    tweets_df_4 = pd.DataFrame(fourth_list, columns = ['Text','Date','Retweets','favorites','HashTags'])
    tweets_df_4 = tweets_df_4[tweets_df_4.Retweets > 0]
    tweets_df_4 = tweets_df_4[~tweets_df_4['Text'].str.startswith('Hi')]
    tweets_df_4 = tweets_df_4[~tweets_df_4['Text'].str.startswith('Hey')]
    tweets_df_4 = tweets_df_4[~tweets_df_4['Text'].str.startswith('Hello')]
    #tweets_df_1.to_csv('{}-{}k-recenttweets.csv'.format(username, int(count/1000)), sep=',')
    tweets_df_4.to_csv('{}-4.csv'.format(username, sep=','))
    #tweets_df_1.to_csv("%s_1.csv" % start_date)
    print('Finished it.')
    now = datetime.now()
    print(str(now))
    print('Next one in 10 mins.')   
    time.sleep(900)
    
    print('starting fifth download')
     # specifying tweet search criteria 
    fifth = got.manager.TweetCriteria().setUsername(username)\
                          .setTopTweets(top_only)\
                          .setSince(part4.strftime('%Y-%m-%d'))\
                          .setUntil(part5.strftime('%Y-%m-%d'))\
                         # .setMaxTweets(count)
    
    fifthdownload = got.manager.TweetManager.getTweets(fifth)
    
    # creating list of tweets with the tweet attributes 
    # specified in the list comprehension
    fifth_list = [[tw.text,
                tw.date,
                tw.retweets,
                tw.favorites,    
                tw.hashtags] for tw in fifthdownload]

    
    # creating dataframe, assigning column names to list of
    # tweets corresponding to tweet attributes
    tweets_df_5 = pd.DataFrame(fifth_list, columns = ['Text','Date','Retweets','favorites','HashTags'])
    tweets_df_5 = tweets_df_5[tweets_df_5.Retweets > 0]
    tweets_df_5 = tweets_df_5[~tweets_df_5['Text'].str.startswith('Hi')]
    tweets_df_5 = tweets_df_5[~tweets_df_5['Text'].str.startswith('Hey')]
    tweets_df_5 = tweets_df_5[~tweets_df_5['Text'].str.startswith('Hello')]
    #tweets_df_1.to_csv('{}-{}k-recenttweets.csv'.format(username, int(count/1000)), sep=',')
    tweets_df_5.to_csv('{}-5.csv'.format(username, sep=','))
    #tweets_df_2.to_csv("%s_2.csv" % start_date)
    print('Finished it.')
    now = datetime.now()
    print(str(now))
    print('Next one in 10 mins.')   
    time.sleep(900)
    
    print('starting sixth download')
        # specifying tweet search criteria 
    sixth = got.manager.TweetCriteria().setUsername(username)\
                          .setTopTweets(top_only)\
                          .setSince(part5.strftime('%Y-%m-%d'))\
                          .setUntil(part6.strftime('%Y-%m-%d'))\
                        #  .setMaxTweets(count)
    
    sixthdownload = got.manager.TweetManager.getTweets(sixth)
    
    # creating list of tweets with the tweet attributes 
    # specified in the list comprehension
    sixth_list = [[tw.text,
                tw.date,
                tw.retweets,
                tw.favorites,    
                tw.hashtags] for tw in sixthdownload]

    
    # creating dataframe, assigning column names to list of
    # tweets corresponding to tweet attributes
    tweets_df_6 = pd.DataFrame(sixth_list, columns = ['Text','Date','Retweets','favorites','HashTags'])
    tweets_df_6 = tweets_df_6[tweets_df_6.Retweets > 0]
    tweets_df_6 = tweets_df_6[~tweets_df_6['Text'].str.startswith('Hi')]
    tweets_df_6 = tweets_df_6[~tweets_df_6['Text'].str.startswith('Hey')]
    tweets_df_6 = tweets_df_6[~tweets_df_6['Text'].str.startswith('Hello')]
    #tweets_df_1.to_csv('{}-{}k-recenttweets.csv'.format(username, int(count/1000)), sep=',')
    tweets_df_6.to_csv('{}-6.csv'.format(username, sep=','))
    #tweets_df_3.to_csv("%s_3.csv" % start_date)
    print('Finished it.')
    now = datetime.now()
    print(str(now))
    print('Next one in 10 mins.')   
    time.sleep(900)
    
    print ('starting seventh download')
        # specifying tweet search criteria 
    seventh = got.manager.TweetCriteria().setUsername(username)\
                          .setTopTweets(top_only)\
                          .setSince(part6.strftime('%Y-%m-%d'))\
                          .setUntil(part7.strftime('%Y-%m-%d'))\
                         # .setMaxTweets(count)
    
    seventhdownload = got.manager.TweetManager.getTweets(seventh)
    
    # creating list of tweets with the tweet attributes 
    # specified in the list comprehension
    seventh_list = [[tw.text,
                tw.date,
                tw.retweets,
                tw.favorites,    
                tw.hashtags] for tw in seventhdownload]

    
    # creating dataframe, assigning column names to list of
    # tweets corresponding to tweet attributes
    tweets_df_7 = pd.DataFrame(seventh_list, columns = ['Text','Date','Retweets','favorites','HashTags'])
    tweets_df_7 = tweets_df_7[tweets_df_7.Retweets > 0]
    tweets_df_7 = tweets_df_7[~tweets_df_7['Text'].str.startswith('Hi')]
    tweets_df_7 = tweets_df_7[~tweets_df_7['Text'].str.startswith('Hey')]
    tweets_df_7 = tweets_df_7[~tweets_df_7['Text'].str.startswith('Hello')]
    #tweets_df_1.to_csv('{}-{}k-recenttweets.csv'.format(username, int(count/1000)), sep=',')
    tweets_df_7.to_csv('{}-7.csv'.format(username, sep=','))
    #tweets_df_1.to_csv("%s_1.csv" % start_date)
    print('Finished it.')
    now = datetime.now()
    print(str(now))
    print('Next one in 10 mins.')   
    time.sleep(900)
    
    print ('starting 8th download')
     # specifying tweet search criteria 
    eighth = got.manager.TweetCriteria().setUsername(username)\
                          .setTopTweets(top_only)\
                          .setSince(part7.strftime('%Y-%m-%d'))\
                          .setUntil(part8.strftime('%Y-%m-%d'))\
                          #.setMaxTweets(count)
    
    eighthdownload = got.manager.TweetManager.getTweets(eighth)
    
    # creating list of tweets with the tweet attributes 
    # specified in the list comprehension
    eighth_list = [[tw.text,
                tw.date,
                tw.retweets,
                tw.favorites,    
                tw.hashtags] for tw in eighthdownload]

    
    # creating dataframe, assigning column names to list of
    # tweets corresponding to tweet attributes
    tweets_df_8 = pd.DataFrame(eighth_list, columns = ['Text','Date','Retweets','favorites','HashTags'])
    tweets_df_8= tweets_df_8[tweets_df_8.Retweets > 0]
    tweets_df_8= tweets_df_8[~tweets_df_8['Text'].str.startswith('Hi')]
    tweets_df_8= tweets_df_8[~tweets_df_8['Text'].str.startswith('Hey')]
    tweets_df_8= tweets_df_8[~tweets_df_8['Text'].str.startswith('Hello')]
    #tweets_df_1.to_csv('{}-{}k-recenttweets.csv'.format(username, int(count/1000)), sep=',')
    tweets_df_8.to_csv('{}-8.csv'.format(username, sep=','))
    #tweets_df_2.to_csv("%s_2.csv" % start_date)
    print('Finished it.')
    now = datetime.now()
    print(str(now))
    print('Next one in 10 mins.')   
    time.sleep(900)
    
    print ('starting ninth download')
        # specifying tweet search criteria 
    ninth = got.manager.TweetCriteria().setUsername(username)\
                          .setTopTweets(top_only)\
                          .setSince(part8.strftime('%Y-%m-%d'))\
                          .setUntil(part9.strftime('%Y-%m-%d'))\
                          #.setMaxTweets(count)
    
    ninthdownload = got.manager.TweetManager.getTweets(ninth)
    
    # creating list of tweets with the tweet attributes 
    # specified in the list comprehension
    ninth_list = [[tw.text,
                tw.date,
                tw.retweets,
                tw.favorites,    
                tw.hashtags] for tw in ninthdownload]

    
    # creating dataframe, assigning column names to list of
    # tweets corresponding to tweet attributes
    tweets_df_9 = pd.DataFrame(ninth_list, columns = ['Text','Date','Retweets','favorites','HashTags'])
    tweets_df_9 = tweets_df_9[tweets_df_9.Retweets > 0]
    tweets_df_9 = tweets_df_9[~tweets_df_9['Text'].str.startswith('Hi')]
    tweets_df_9 = tweets_df_9[~tweets_df_9['Text'].str.startswith('Hey')]
    tweets_df_9 = tweets_df_9[~tweets_df_9['Text'].str.startswith('Hello')]
    #tweets_df_1.to_csv('{}-{}k-recenttweets.csv'.format(username, int(count/1000)), sep=',')
    tweets_df_9.to_csv('{}-9.csv'.format(username, sep=','))
    #tweets_df_3.to_csv("%s_3.csv" % start_date)
    print('Finished it.')
    now = datetime.now()
    print(str(now))
    print('Next one in 10 mins.')   
    time.sleep(900)
    
    print ('starting 10th download')
        # specifying tweet search criteria 
    tenth = got.manager.TweetCriteria().setUsername(username)\
                          .setTopTweets(top_only)\
                          .setSince(part9.strftime('%Y-%m-%d'))\
                          .setUntil(part10.strftime('%Y-%m-%d'))\
                          #.setMaxTweets(count)
    
    tenthdownload = got.manager.TweetManager.getTweets(tenth)
    
    # creating list of tweets with the tweet attributes 
    # specified in the list comprehension
    tenth_list = [[tw.text,
                tw.date,
                tw.retweets,
                tw.favorites,    
                tw.hashtags] for tw in tenthdownload]

    
    # creating dataframe, assigning column names to list of
    # tweets corresponding to tweet attributes
    tweets_df_10 = pd.DataFrame(tenth_list, columns = ['Text','Date','Retweets','favorites','HashTags'])
    tweets_df_10 = tweets_df_10[tweets_df_10.Retweets > 0]
    tweets_df_10 = tweets_df_10[~tweets_df_10['Text'].str.startswith('Hi')]
    tweets_df_10 = tweets_df_10[~tweets_df_10['Text'].str.startswith('Hey')]
    tweets_df_10 = tweets_df_10[~tweets_df_10['Text'].str.startswith('Hello')]
    #tweets_df_1.to_csv('{}-{}k-recenttweets.csv'.format(username, int(count/1000)), sep=',')
    tweets_df_10.to_csv('{}-10.csv'.format(username, sep=','))
    #tweets_df_1.to_csv("%s_1.csv" % start_date)
    print('Finished it.')
    now = datetime.now()
    print(str(now))
    print('Next one in 10 mins.')   
    time.sleep(900)
    
    print ('starting eleventh download')
    # specifying tweet search criteria 
    eleventh = got.manager.TweetCriteria().setUsername(username)\
                          .setTopTweets(top_only)\
                          .setSince(part10.strftime('%Y-%m-%d'))\
                          .setUntil(part11.strftime('%Y-%m-%d'))\
                          #.setMaxTweets(count)
    
    eleventhdownload = got.manager.TweetManager.getTweets(eleventh)
    
    # creating list of tweets with the tweet attributes 
    # specified in the list comprehension
    eleventh_list = [[tw.text,
                tw.date,
                tw.retweets,
                tw.favorites,    
                tw.hashtags] for tw in tenthdownload]

    
    # creating dataframe, assigning column names to list of
    # tweets corresponding to tweet attributes
    tweets_df_11 = pd.DataFrame(eleventh_list, columns = ['Text','Date','Retweets','favorites','HashTags'])
    tweets_df_11 = tweets_df_11[tweets_df_11.Retweets > 0]
    tweets_df_11 = tweets_df_11[~tweets_df_11['Text'].str.startswith('Hi')]
    tweets_df_11 = tweets_df_11[~tweets_df_11['Text'].str.startswith('Hey')]
    tweets_df_11 = tweets_df_11[~tweets_df_11['Text'].str.startswith('Hello')]
    #tweets_df_1.to_csv('{}-{}k-recenttweets.csv'.format(username, int(count/1000)), sep=',')
    tweets_df_11.to_csv('{}-11.csv'.format(username, sep=','))
    #tweets_df_1.to_csv("%s_1.csv" % start_date)
    print('Finished it.')
    now = datetime.now()
    print(str(now))
    print('Next one in 10 mins.')   
    time.sleep(900)
    
    print ('starting twelvth download')
    # specifying tweet search criteria 
    twelvth = got.manager.TweetCriteria().setUsername(username)\
                          .setTopTweets(top_only)\
                          .setSince(part11.strftime('%Y-%m-%d'))\
                          .setUntil(part12.strftime('%Y-%m-%d'))\
                          #.setMaxTweets(count)
    
    twelvthdownload = got.manager.TweetManager.getTweets(twelvth)
    
    # creating list of tweets with the tweet attributes 
    # specified in the list comprehension
    twelvth_list = [[tw.text,
                tw.date,
                tw.retweets,
                tw.favorites,    
                tw.hashtags] for tw in twelvthdownload]

    
    # creating dataframe, assigning column names to list of
    # tweets corresponding to tweet attributes
    tweets_df_12 = pd.DataFrame(twelvth_list, columns = ['Text','Date','Retweets','favorites','HashTags'])
    tweets_df_12 = tweets_df_12[tweets_df_12.Retweets > 0]
    tweets_df_12 = tweets_df_12[~tweets_df_12['Text'].str.startswith('Hi')]
    tweets_df_12 = tweets_df_12[~tweets_df_12['Text'].str.startswith('Hey')]
    tweets_df_12 = tweets_df_12[~tweets_df_12['Text'].str.startswith('Hello')]
    #tweets_df_1.to_csv('{}-{}k-recenttweets.csv'.format(username, int(count/1000)), sep=',')
    tweets_df_12.to_csv('{}-12.csv'.format(username, sep=','))
    print('Finished it.')
    now = datetime.now()
    print(str(now))
    print('Next one in 10 mins.')   
    time.sleep(900)
    
    print ('starting thirteenth download')
    # specifying tweet search criteria 
    thirteenth = got.manager.TweetCriteria().setUsername(username)\
                          .setTopTweets(top_only)\
                          .setSince(part12.strftime('%Y-%m-%d'))\
                          .setUntil(part13.strftime('%Y-%m-%d'))\
                          #.setMaxTweets(count)
    
    thirteenthdownload = got.manager.TweetManager.getTweets(thirteenth)
    
    # creating list of tweets with the tweet attributes 
    # specified in the list comprehension
    thirteenth_list = [[tw.text,
                tw.date,
                tw.retweets,
                tw.favorites,    
                tw.hashtags] for tw in thirteenthdownload]

    
    # creating dataframe, assigning column names to list of
    # tweets corresponding to tweet attributes
    tweets_df_13 = pd.DataFrame(thirteenth_list, columns = ['Text','Date','Retweets','favorites','HashTags'])
    tweets_df_13 = tweets_df_13[tweets_df_13.Retweets > 0]
    tweets_df_13 = tweets_df_13[~tweets_df_13['Text'].str.startswith('Hi')]
    tweets_df_13 = tweets_df_13[~tweets_df_13['Text'].str.startswith('Hey')]
    tweets_df_13 = tweets_df_13[~tweets_df_13['Text'].str.startswith('Hello')]
    #tweets_df_1.to_csv('{}-{}k-recenttweets.csv'.format(username, int(count/1000)), sep=',')
    tweets_df_13.to_csv('{}-13.csv'.format(username, sep=','))
    #tweets_df_1.to_csv("%s_1.csv" % start_date)
    print('Finished it.')
    now = datetime.now()
    print(str(now))
    print('Next one in 10 mins.')   
    time.sleep(900)
    
    print ('starting fourteenth download')
    # specifying tweet search criteria 
    fourteenth = got.manager.TweetCriteria().setUsername(username)\
                          .setTopTweets(top_only)\
                          .setSince(part13.strftime('%Y-%m-%d'))\
                          .setUntil(part14.strftime('%Y-%m-%d'))\
                          #.setMaxTweets(count)
    
    fourteenthdownload = got.manager.TweetManager.getTweets(fourteenth)
    
    # creating list of tweets with the tweet attributes 
    # specified in the list comprehension
    fourteenth_list = [[tw.text,
                tw.date,
                tw.retweets,
                tw.favorites,    
                tw.hashtags] for tw in fourteenthdownload]

    
    # creating dataframe, assigning column names to list of
    # tweets corresponding to tweet attributes
    tweets_df_14 = pd.DataFrame(fourteenth_list, columns = ['Text','Date','Retweets','favorites','HashTags'])
    tweets_df_14 = tweets_df_14[tweets_df_14.Retweets > 0]
    tweets_df_14 = tweets_df_14[~tweets_df_14['Text'].str.startswith('Hi')]
    tweets_df_14 = tweets_df_14[~tweets_df_14['Text'].str.startswith('Hey')]
    tweets_df_14 = tweets_df_14[~tweets_df_14['Text'].str.startswith('Hello')]
    #tweets_df_1.to_csv('{}-{}k-recenttweets.csv'.format(username, int(count/1000)), sep=',')
    tweets_df_14.to_csv('{}-14.csv'.format(username, sep=','))
    #tweets_df_1.to_csv("%s_1.csv" % start_date)
    print('Finished it.')
    now = datetime.now()
    print(str(now))
    print('Next one in 10 mins.')   
    time.sleep(900)
    
    print ('starting final download')
    # specifying tweet search criteria 
    final = got.manager.TweetCriteria().setUsername(username)\
                          .setTopTweets(top_only)\
                          .setSince(part14.strftime('%Y-%m-%d'))\
                          .setUntil(until.strftime('%Y-%m-%d'))\
                          #.setMaxTweets(count)
    
    finaldownload = got.manager.TweetManager.getTweets(final)
    
    # creating list of tweets with the tweet attributes 
    # specified in the list comprehension
    final_list = [[tw.text,
                tw.date,
                tw.retweets,
                tw.favorites,    
                tw.hashtags] for tw in finaldownload]

    
    # creating dataframe, assigning column names to list of
    # tweets corresponding to tweet attributes
    tweets_df_15 = pd.DataFrame(final_list, columns = ['Text','Date','Retweets','favorites','HashTags'])
    tweets_df_15 = tweets_df_15[tweets_df_15.Retweets > 0]
    tweets_df_15= tweets_df_15[~tweets_df_15['Text'].str.startswith('Hi')]
    tweets_df_15 = tweets_df_15[~tweets_df_15['Text'].str.startswith('Hey')]
    tweets_df_15 = tweets_df_15[~tweets_df_15['Text'].str.startswith('Hello')]
    #tweets_df_1.to_csv('{}-{}k-recenttweets.csv'.format(username, int(count/1000)), sep=',')
    tweets_df_15.to_csv('{}-15.csv'.format(username, sep=','))
    #tweets_df_2.to_csv("%s_2.csv" % start_date)
    
    df=pd.concat([tweets_df_1,tweets_df_2,tweets_df_3,tweets_df_4,tweets_df_5,tweets_df_6,tweets_df_7,tweets_df_8,tweets_df_9,tweets_df_10,tweets_df_11,tweets_df_12,tweets_df_13,tweets_df_14,tweets_df_15])
    df.to_csv('{}-recenttweets.csv'.format(username, sep=','))
    return df
    print('Finished. Now sleeping for 10mins before being ready for the next scraping')
    now = datetime.now()
    print(str(now))
    time.sleep(900)          


#for option 4
# Input username(s) to scrape tweets and name csv file
username = 'AmericanAir' 
top_only = True
start_date="2015-01-01"
end_date="2020-08-24"
#count = 40000
# Calling function to turn username's past x amount of tweets into a CSV file
#get_tweets(username, top_only, start_date, end_date, count)
get_tweets(username, top_only, start_date, end_date)
