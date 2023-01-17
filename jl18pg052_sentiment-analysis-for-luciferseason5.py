import nltk

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import re

import nltk

from wordcloud import WordCloud

nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

from textblob import TextBlob
data=pd.read_csv("../input/twitter-data-for-luciferseason5/Twitter data for LuciferSeason5.csv")
data.head()
data2=pd.DataFrame(data=data["tweet"],columns=["tweet"])

pd.set_option('display.max_colwidth',None)

data2.head()
def cleaned_text(text):

    clean=re.sub("http\S+","",text)

    clean=re.sub("pic.twitter\S+","",clean)

    clean=re.sub("#\S+","",clean)

    clean=clean.lower()

    clean=re.sub("@\S+","",clean)

    clean=re.sub("[^a-z]"," ",clean)

    clean=re.sub("can t","can not",clean)

    clean=re.sub("don t","do not",clean)

    clean=re.sub("pleaseee","please",clean)

    clean=re.sub("plss","please",clean)

    clean=re.sub("haven t","have not",clean)

    clean=re.sub("you re","you are",clean)

    clean=re.sub("aren t","are not",clean)

    clean=re.sub("there s","there is",clean)

    clean=re.sub("isn t","is not",clean)

    clean=re.sub("it s","it is",clean)

    clean=re.sub(r"\s+[a-z]\s+"," ",clean)

    clean=re.sub(r"\s+[a-z]\s+"," ",clean)

    clean=re.sub("winzzzzzzz","win",clean)

    clean=re.sub("pleaseeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","please",clean)

    clean=re.sub("ishould","should",clean)

    clean=re.sub("omg","oh my god",clean)

    clean=re.sub("yessss","yes",clean)

    clean=re.sub("gonna","going to",clean)

    clean=re.sub("iseem","seem",clean)

    clean=re.sub("maybe","may be",clean)

    clean=re.sub("dont","do not",clean)

    clean=re.sub("wouldnt","would not",clean)

    clean=re.sub("imma","i am going to",clean)

    clean=re.sub("btw","by the way",clean)

    clean=re.sub("breath","breathe",clean)

    clean=re.sub("fanfic","fan fiction",clean)

    clean=re.sub("yall","you all",clean)

    clean=re.sub("cannot","can not",clean)

    clean=re.sub("eachother","each other",clean)

    clean=re.sub("hubbys","husbands",clean)

    clean=re.sub("frees","free",clean)

    clean=re.sub("cking","fucking",clean)

    clean=re.sub("dyiiiiiiiiiiing","dying",clean)

    clean=re.sub("wheennn","when",clean)

    clean=re.sub("pls","please",clean)

    clean=re.sub("sofar","so far",clean)

    clean=re.sub("soooon","soon",clean)

    clean=re.sub("fufucking","fucking",clean)

    clean=re.sub("sooooo","so",clean)

    clean=re.sub("plz","please",clean)

    clean=re.sub("pleeeaaaasssseeeee","please",clean)

    clean=re.sub("fvfucking","fucking",clean)

    clean=re.sub("soooooo","so",clean)

    clean=re.sub("listenning","listening",clean)

    clean=re.sub("yeeeeesss","yes",clean)

    clean=re.sub("seaon","season",clean)

    clean=re.sub("pleasee","please",clean)

    clean=re.sub("awesomeeeeeeeee","awesome",clean)

    clean=re.sub("waitttt","wait",clean)

    clean=re.sub("fking","fucking",clean)

    clean=re.sub("isoo","so",clean)

    clean=re.sub("lmao","laughing my ass off",clean)

    clean=re.sub("srsly","seriously",clean)

    clean=re.sub("yaaaaaassss","yes",clean)

    clean=re.sub("wanna","want to",clean)

    clean=re.sub("f ck","fuck",clean)

    clean=re.sub("guy","",clean)

    clean=re.sub("freakin","freaking",clean)

    clean=re.sub("tbh","to be honest",clean)

    clean=re.sub("soooo","so",clean)

    clean=re.sub("neighborhood","neighbourhood",clean)

    clean=re.sub("needdd","need",clean)

    clean=re.sub("cant","cannot",clean)

    clean=re.sub("isad","sad",clean)

    clean=re.sub("netflixxxx","netflix",clean)

    clean=re.sub("ppl","people",clean)

    clean=re.sub("sooo","so",clean)

    clean=re.sub("bbq","barbecue",clean)

    clean=re.sub("areally","really",clean)

    clean=re.sub("frifucking","freaking",clean)

    clean=clean.lstrip()

    clean=re.sub("\s{2,}"," ",clean)

    return clean

data2["cleaned_tweets"]=data2["tweet"].apply(cleaned_text)
data2.head()
data2['Number_of_words'] = data2['cleaned_tweets'].apply(lambda x:len(str(x).split()))
plt.style.use('ggplot')

plt.figure(figsize=(14,6))

sns.distplot(data2['Number_of_words'],kde = False,color="red")

plt.title("Frequency distribution of number of words for each tweet", size=20)
nltk.download("stopwords")

from nltk.corpus import stopwords

stop=stopwords.words('english')

data2["cleaned_clean_tweets"]=data2["cleaned_tweets"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
print(stop)
data2["cleaned_clean_tweets"]=data2["cleaned_clean_tweets"].apply(lambda x: nltk.word_tokenize(x))
def word_lemmatizer(text):

    lem_text = [WordNetLemmatizer().lemmatize(i,pos='v') for i in text]

    return lem_text

data2["cleaned_clean_tweets"]=data2["cleaned_clean_tweets"].apply(lambda x: word_lemmatizer(x))

data2["cleaned_clean_tweets"]=data2["cleaned_clean_tweets"].apply(lambda x: ' '.join(x))
plt.style.use('ggplot')

plt.figure(figsize=(14,6))

freq=pd.Series(" ".join(data2["cleaned_clean_tweets"]).split()).value_counts()[:20]

freq.plot(kind="bar")

plt.title("20 most frequent words",size=20)
cloud=WordCloud(colormap="Dark2").generate(str(data2["cleaned_clean_tweets"]))

fig=plt.figure(figsize=(12,18))

plt.axis("off")

plt.imshow(cloud,interpolation='bilinear')
def analyze_sentiment(tweet):

    analysis = TextBlob(tweet)

    if analysis.sentiment.polarity > 0:     #### For positive sentiment

        return 1

    elif analysis.sentiment.polarity == 0:  ### Neutral

        return 0

    else:

        return -1                           #### Negative sentiment



data2['SA'] = data2["cleaned_clean_tweets"].apply(lambda x: analyze_sentiment(x))
descending_order=data2["SA"].value_counts().sort_values(ascending=False).index

sns.catplot(x="SA",data=data2,kind="count",height=5,aspect=2,order=descending_order)
another_method = data2.groupby('SA').count()['cleaned_clean_tweets'].reset_index().sort_values(by='cleaned_clean_tweets',ascending=False)

another_method.style.background_gradient(cmap='Purples')
print(len(data2[data2["SA"]==1])/len(data2["SA"])*100)  ##positive

print(len(data2[data2["SA"]==0])/len(data2["SA"])*100)  ##neutral

print(len(data2[data2["SA"]==-1])/len(data2["SA"])*100) ##negative
cloud=WordCloud(colormap="Dark2").generate(str(data2[data2["SA"]==1]["cleaned_clean_tweets"]))

fig=plt.figure(figsize=(12,14))

plt.axis("off")

plt.imshow(cloud,interpolation='bilinear')
cloud=WordCloud(colormap="Dark2").generate(str(data2[data2["SA"]==-1]["cleaned_clean_tweets"]))

fig=plt.figure(figsize=(12,18))

plt.axis("off")

plt.imshow(cloud,interpolation='bilinear')