import re

def cleanTweet(tweet_file):
    # open the tweet file for reading
    f = open(tweet_file, 'r')
    tweets = []
    
    # split the tweets by newline characters
    #
    for line in f:
        for j in line.split(r"\n"):
            tweets.append(j)
    
    # Cleansing tweets by using regular expressions
    #
    tweet_filter=[]
    for tweet in tweets:
        if tweet is not None:
            tweet_parse = tweet.split(':')
            if len(tweet_parse) > 1:
                tweet = tweet_parse[-1]
                tweet = re.sub(r'\\x[0-9a-f][0-9a-f]', "", tweet)
                tweet = re.sub('@(\w)+', "", tweet)
                tweet = re.sub('&amp', "", tweet)
                tweet = re.sub('//t.co/(\w)+', "",tweet)
                tweet = re.sub('//t.c', "",tweet)
                tweet = re.sub(r'\\',"", tweet)
                tweet = re.sub('#',"", tweet)
                tweet = re.sub("\'", " " ,tweet)
                tweet_filter.append(tweet.lower())

    # Create a list of words from cleansed tweets
    newlist =[]
    for i in tweet_filter:
        s = i.split()
        for k in s:
            newlist.append(k)
    #print(repr(newlist))
    return newlist
def SentimentAnalysis(newlist):
    (stopcount, positivecount,negativecount,others,sentiment) = (0,0,0,0,0)
    
    # Create sets of positive, negative and stop words
    # Sets allow fast lookup
    #
    PositiveWords = set(open('../input/positive.txt','r',encoding = "ISO-8859-1").read().splitlines())
    NegativeWords = set(open('../input/negative.txt','r',encoding = "ISO-8859-1").read().splitlines())
    StopWords = set(open('../input/stop.txt', 'r', encoding = "ISO-8859-1").read().splitlines())
    
    # Iterate over the list and increment relevant counts
    for word in newlist:
        if  word in StopWords:
            stopcount +=1
        elif word in NegativeWords: 
            negativecount  +=1
            sentiment -= 1
        elif word in PositiveWords:
            positivecount +=1
            sentiment += 1
        else :
            others +=1        

    print("Sentiment value: " + str(sentiment))
    print("Number of stop count:" + str(stopcount))
    print("Number of positive count:" + str(positivecount))
    print("Number of negative count:" + str(negativecount))
    print("Number of others count:" + str(others))

    totalcount = len(newlist)
    print("Ratio of positive words to total:" +str(round(positivecount/totalcount,2)))
    print("Ratio of negative words to total:" +str(round(negativecount/totalcount,2)))
    print("Ratio of stop words to total:" +str(round(stopcount/totalcount,2)))
    print("Ratio of others words to total:" +str(round(others/totalcount,2)))
    print("Ratio of positive words to negative:" +str(round(positivecount/negativecount,2)))

SentimentAnalysis(cleanTweet('../input/Trump.txt'))
SentimentAnalysis(cleanTweet('../input/Melo.txt'))
