#import libraries

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud,STOPWORDS

import nltk as nlp

import re



import warnings

warnings.filterwarnings('ignore')
dataset = pd.read_csv("../input/Tweets.csv")

dataset.head(3)
dataset.columns
data = dataset[["airline_sentiment","text","airline","retweet_count"]]

data.head()
sns.set()

plt.figure(figsize=(7,7))

sns.countplot(x=data["airline_sentiment"],palette="Set2")

plt.title("Sentiment Dist.")

plt.show()
sns.set()

plt.figure(figsize=(7,7))

sns.countplot(y=data["airline"],palette="Set2")

plt.title("Airlines Dist.")

plt.show()
j=1

plt.subplots(figsize=(20,4),tight_layout=True)

for i in data["airline"].unique():

        x = data[data["airline"]==i]

        plt.subplot(1, 6, j)

        sns.countplot(x["airline_sentiment"],palette="Set2")

        plt.xticks(rotation=45)

        plt.title(i)

        j +=1

plt.show()
lemma = nlp.WordNetLemmatizer()

def preprocess(x):

    x = str(x)

    x = re.sub("[^a-zA-z]", " ",x)

    x = x.lower()

    x = nlp.word_tokenize(x)

    #x = [i for i in x if not i in set(stopwords.words("english"))] #slowly

    x = [lemma.lemmatize(i) for  i in x]

    x = " ".join(x)

    return x



data.text = data.text.apply(preprocess)

data.text[0:10]
allcomments = " ".join(data.text)

wordcloud = WordCloud(width = 800, height = 800, 

                    background_color ='white', 

                    stopwords = STOPWORDS, 

                    min_font_size = 12).generate(allcomments) 

      

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud)

plt.title("All Tweets Wordcount")

plt.show()
#data=data[["airline_sentiment","text"]]

#data.head()
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

data["airline_sentiment"] = encoder.fit_transform(data["airline_sentiment"])

print(encoder.classes_)

data.head()
# convert to categorical Categority by using one hot tecnique 

df_dummy = data.copy()

df_dummy.airline = pd.Categorical(df_dummy.airline)

x = df_dummy[['airline']]

del df_dummy['airline']

dummies = pd.get_dummies(x, prefix = 'airline')

data = pd.concat([df_dummy,dummies], axis=1)

data.head()
#normalize retweet count

_max = data.retweet_count.describe()[7]

data.retweet_count = [i/_max for i in data.retweet_count]
#Encode Words

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words = "english")

encoded_X = vectorizer.fit_transform(data.text).toarray()

print(encoded_X.shape)

print("Features First 100:",vectorizer.get_feature_names()[:100])
data2 = data.copy()

del data2["text"]

data2 = pd.concat([pd.DataFrame(encoded_X),data2], axis=1)

data2.head()
X = data2.drop(["airline_sentiment"],axis=1)

y = data2.airline_sentiment
#Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=22)

print("Train :",X_train.shape)

print("Test  :",X_test.shape)
sns.set()

plt.subplots(figsize=(10,5),tight_layout=True)

plt.subplot(1,2,1)

sns.countplot(y_train)

plt.title("Train Dist.")

plt.subplot(1,2,2)

sns.countplot(y_test)

plt.title("Test Dist.")

plt.show()
#Classification

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)
#Result

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



rf_acc = accuracy_score(y_test, pred)

rf_f1 = f1_score(y_test, pred, average="micro")



print("Random Forest")

print("Accuracy : %",round(rf_acc*100,2))

print("F1 Score : %",round(rf_f1*100,2))