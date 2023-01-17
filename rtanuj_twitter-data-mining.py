import tweepy
consumer_key = "jpV6WraK3c4rjTBmSGmZK7IXn"
consumer_secret = "s8GpWTLh3UImoOCwHOfUZHYOCbHJfPEglVNAa6DwmYfYHDnIfk"
access_token = "1203569245365272577-dKhsLvNOUtiRbtaLoQPW8LDNEoPrnX"
access_token_secret = "NvpXo4sS0vUWxJd3rgcGw6iU6n5x2G6fvlnHlQJJjB2ch"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
name = "tanujrajkumar"
tweetCount = 5
results = api.user_timeline(id=name, count=tweetCount)

for tweet in results:
    print(tweet.text)