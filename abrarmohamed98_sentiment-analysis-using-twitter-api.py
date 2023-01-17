!pip install tweepy
import tweepy
consumer_key ='6arcWV7MGs5BNY5uNp0P2oEde'

consumer_secret='k70mOp5INbBK4URXxCz2Ajrfw6SfUmtcMwhQRCBeYXBxBDIAdH'

access_token='1175082985172942848-Xzf6Mxg1D0aYs7C4GdejTukLB3tEnA'

access_token_secret='Hlgi1odFXO2ToUGTHiT2e8V93ef2C7EXlEdfH1M1lApla'
auth=tweepy.OAuthHandler(consumer_key,consumer_secret)

auth.set_access_token(access_token, access_token_secret)

api=tweepy.API(auth) #API instance
search_words = "#demonetisation"

date_since = "2016-11-16"
new_search = search_words + " -filter:retweets" 

tweets = tweepy.Cursor(api.search,

                       q=new_search,

                       lang="en",

                       since=date_since).items(3000)

tweets
x=[]
for tweet in tweets:

    x.append(tweet.text)
x
import re
len(x)
corpus=[]

for i in range(len(x)):

    #print(x[i])

    review=re.sub('@\w* ',"", x[i])

    review=re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', review)

    review=review.lower()

    corpus.append(review)
corpus
!pip install afinn
from afinn import Afinn
af = Afinn()
y=[]

for i in range(len(corpus)):

    a=af.score(corpus[i])

    y.append(a)
y
len(y)
n=0;

p=0;

ne=0;

for i in range(len(y)):

    if y[i]<0:

        n=n+1;

    if y[i]>0:

        p=p+1;

    else:

        ne=ne+1;

t=len(y)

print('Total number of tweets extracted = ',t)

print('Total number of positive tweets = ',p)

print('Total number of neutral tweets = ',ne)

print('Total number of negative tweets = ',n)