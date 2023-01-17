import pandas as pd

import re

import seaborn as sns

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import nltk.corpus

nltk.download("stopwords")

from nltk.corpus import stopwords

nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

from textblob import TextBlob
data=pd.read_csv("../input/tweets-for-georgefloydfuneral/George Floyd Twitter Data.csv")

data.head()
data_copy=data.copy()
def cleaned_data(text):

    clean=re.sub("http\S+","",text)

    clean=clean.lower()

    clean=re.sub("\d","",clean)

    clean=re.sub("[^a-z]"," ",clean)

    clean=re.sub("\s{2,}"," ",clean)

    clean=clean.lstrip()

    clean=re.sub("don t","do not",clean)

    clean=re.sub("amp","",clean)

    clean=re.sub("i m","i am",clean)

    clean=re.sub("didn t","did not",clean)

    clean=re.sub("couldn t","could not",clean)

    clean=re.sub("son s","son",clean)

    #clean=re.sub("n","and",clean)

    clean=re.sub("pu sy","pussy",clean)

    clean=re.sub("fu ker","fucker",clean)

    clean=re.sub("can t","can not",clean)

    clean=re.sub("sh t","shit",clean)

    #clean=re.sub("inj","injustice",clean)

    clean=re.sub("you re","you are",clean)

    clean=re.sub("doesn t","does not",clean)

    clean=re.sub("wasn t","was not",clean)

    #clean=re.sub("presi","president",clean)

    clean=re.sub("let s","let us",clean)

    clean=re.sub("people s","people",clean)

    clean=re.sub("birble","bible",clean)

    clean=re.sub("servi","service",clean)

    clean=re.sub("it s","it is",clean)

    clean=re.sub("we re","we are",clean)

    clean=re.sub("ameri","america",clean)

    clean=re.sub("b c","because",clean)

    #clean=re.sub("cha","change",clean)

    clean=re.sub("sacri","sacrifice",clean)

    clean=re.sub("cant","can not",clean)

    clean=re.sub("today s","today",clean)

    clean=re.sub("floyd s","floyd",clean)

    clean=clean=re.sub("nationalguard","national guard",clean)

    clean=re.sub("minori","minority",clean)

    clean=re.sub("georgeflyod s","georgefloyd",clean)

    #clean=re.sub("p","protest",clean)

    #clean=re.sub("nd","second",clean)

    clean=re.sub("shouldn t","should not",clean)

    #clean=re.sub("racis","racism",clean)

    clean=re.sub("how d","how did",clean)

    #clean=re.sub("mu","music",clean)

    clean=re.sub("wanna","want to",clean)

    clean=re.sub("addre","address",clean)

    clean=re.sub("isince","since",clean)

    clean=re.sub("didid","did",clean)

    clean=re.sub("ya ll","you all",clean)

    clean=re.sub("ican notbreathe","i can not breathe",clean)

    clean=re.sub("caign","campaign",clean)

    clean=re.sub("caigning","campaigning",clean)

    clean=re.sub("didemocrats","democrats",clean)

    clean=re.sub("i amean","i mean",clean)

    clean=re.sub("americacans","americans",clean)

    clean=re.sub("concealin","concealing",clean)

    clean=re.sub("notell","tell",clean)

    clean=re.sub("what s","what is",clean)

    clean=re.sub("americaca","america",clean)

    clean=re.sub("\s{2,}"," ",clean)

    return clean

data["cleaned_tweets"]=data["Tweets"].apply(cleaned_data)
data.head(10)
data.drop(columns=["Unnamed: 0"],inplace=True)
data.head()
stop=stopwords.words('english')

data["cleaned_tweets"]=data["cleaned_tweets"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
data.head()
data["clean_cleaned"]=data["cleaned_tweets"].apply(lambda x: nltk.word_tokenize(x))
data.head()
def word_lemmatizer(text):

    lem_text = [WordNetLemmatizer().lemmatize(i,pos='v') for i in text]

    return lem_text

data["clean_cleaned_tweets"]=data["clean_cleaned"].apply(lambda x: word_lemmatizer(x))

data["clean_cleaned_tweets"]=data["clean_cleaned_tweets"].apply(lambda x: ' '.join(x))
plt.figure(figsize=(12,6))

freq=pd.Series(' '.join(data["cleaned_tweets"]).split()).value_counts()[:20]

freq.plot(kind="bar")

plt.title("Most frequent words in the cleaned tweets",size=20)
cloud=WordCloud(colormap="OrRd_r",width=700,height=350).generate(str(data["clean_cleaned_tweets"]))

fig=plt.figure(figsize=(12,18))

plt.axis("off")

plt.imshow(cloud)
def analyze_sentiment(tweet):

    analysis = TextBlob(tweet)

    if analysis.sentiment.polarity > 0:     #### For positive sentiment

        return 1

    elif analysis.sentiment.polarity == 0:  ### Neutral

        return 0

    else:

        return -1                           #### Negative sentiment

data['SA'] = data["clean_cleaned_tweets"].apply(lambda x: analyze_sentiment(x))
data.head()
print(len(data[data["SA"]==0]))  ## Neutral

print(len(data[data["SA"]==-1])) ## Negative

print(len(data[data["SA"]==1]))  ## Positive
print(len(data[data["SA"]==0])/len(data["SA"])*100)

print(len(data[data["SA"]==-1])/len(data["SA"])*100)

print(len(data[data["SA"]==1])/len(data["SA"])*100)
cloud=WordCloud().generate(str(data[data["SA"]==-1]["clean_cleaned_tweets"]))

fig=plt.figure(figsize=(12,14))

plt.axis("off")

plt.imshow(cloud)
cloud=WordCloud().generate(str(data[data["SA"]==1]["clean_cleaned_tweets"]))

fig=plt.figure(figsize=(12,14))

plt.axis("off")

plt.imshow(cloud)
descending_order=data["SA"].value_counts().sort_values(ascending=False).index

sns.catplot(x="SA",data=data,kind="count",height=5,aspect=2,order=descending_order)