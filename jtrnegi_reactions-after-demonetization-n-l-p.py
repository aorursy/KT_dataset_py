import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("../input/demonetization-tweets.csv",encoding="ISO-8859-1")
df.head(5)
#remove the unwanted columns
df.drop(['Unnamed: 0', 'X', 'favorited', 'favoriteCount', 'replyToSN',
       'created', 'truncated', 'replyToSID', 'id', 'replyToUID',
       'statusSource'],axis=1,inplace=True)
df.iloc[df['retweetCount'].idxmax()]['text'] #IDXMAX : GIVES THE INDEX OF THE MAXIMUM VALUE
import nltk
from nltk.probability import FreqDist
user_list = df['screenName'].tolist()
max_user = FreqDist(user_list)
max_user.plot(10)
import string,re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('brown')
#THE CLEANING FUNCTION
def clean_text(tweets):
    tweets = word_tokenize(tweets)#SEPERATE EACH WORD
    tweets = tweets[4:] #to remove RT@
    tweets= " ".join(tweets)#JOIN WORDS
    tweets=re.sub('https','',tweets)#REMOVE HTTPS TEXT WITH BLANK
    tweets = [char for char in tweets if char not in string.punctuation]#REMOVE PUNCTUATIONS 
    tweets = ''.join(tweets)#JOIN THE LETTERS
    tweets = [word for word in tweets.split() if word.lower() not in stopwords.words('english')]#REMOVE COMMON ENGLISH WORDS(I,YOU,WE...)
    return " ".join(tweets)

df['cleaned_text']=df['text'].apply(clean_text) #adding clean text to dataframe
clean_term = []
for terms in df['cleaned_text']:
    clean_term += terms.split(" ")
cleaned = FreqDist(clean_term)
cleaned.plot(10)
from textblob import TextBlob
#CREAING A FUNCTION TO GET THE POLARITY
def sentiments(tweets):
    tweets = TextBlob(tweets)
    pol = tweets.polarity #GIVES THE VALUE OF POLARITY
    return pol

df["polarity"] = df["cleaned_text"].apply(sentiments) #APPLY FUNCTION ON CLEAN TWEETS
df.head(5)
print("THE AVERAGE POLARITY",np.mean(df["polarity"])) #gives the average sentiments of people
print("THE MOST -VE TWEET :",df.iloc[df['polarity'].idxmin()]['text'])# most positive
print("THE MOST +VE TWEET :",df.iloc[df['polarity'].idxmax()]['text'])#most negetive

