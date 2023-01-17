# Import Libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Warnings

import warnings

warnings.filterwarnings('ignore')



# Styles

plt.style.use('ggplot')

sns.set_style('whitegrid')



plt.rcParams['font.family'] = 'serif'

plt.rcParams['font.serif'] = 'Ubuntu'

plt.rcParams['font.monospace'] = 'Ubuntu Mono'

plt.rcParams['font.size'] = 10

plt.rcParams['axes.labelsize'] = 10

plt.rcParams['xtick.labelsize'] = 8

plt.rcParams['ytick.labelsize'] = 8

plt.rcParams['legend.fontsize'] = 10

plt.rcParams['figure.titlesize'] = 12

plt.rcParams['patch.force_edgecolor'] = True



# Text Preprocessing

import nltk

# nltk.download("all")

from nltk.corpus import stopwords

import string

from nltk.tokenize import word_tokenize



import spacy

nlp = spacy.load("en")
messages = pd.read_csv("./../input/spam.csv", encoding = 'latin-1')



# Drop the extra columns and rename columns



messages = messages.drop(labels = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)

messages.columns = ["category", "text"]
display(messages.head(n = 10))
# Lets look at the dataset info to see if everything is alright



messages.info()
messages["category"].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), autopct = '%1.1f%%', shadow = True)

plt.ylabel("Spam vs Ham")

plt.legend(["Ham", "Spam"])

plt.show()
topMessages = messages.groupby("text")["category"].agg([len, np.max]).sort_values(by = "len", ascending = False).head(n = 10)

display(topMessages)
spam_messages = messages[messages["category"] == "spam"]["text"]

ham_messages = messages[messages["category"] == "ham"]["text"]



spam_words = []

ham_words = []



# Since this is just classifying the message as spam or ham, we can use isalpha(). 

# This will also remove the not word in something like can't etc. 

# In a sentiment analysis setting, its better to use 

# sentence.translate(string.maketrans("", "", ), chars_to_remove)



def extractSpamWords(spamMessages):

    global spam_words

    words = [word.lower() for word in word_tokenize(spamMessages) if word.lower() not in stopwords.words("english") and word.lower().isalpha()]

    spam_words = spam_words + words

    

def extractHamWords(hamMessages):

    global ham_words

    words = [word.lower() for word in word_tokenize(hamMessages) if word.lower() not in stopwords.words("english") and word.lower().isalpha()]

    ham_words = ham_words + words



spam_messages.apply(extractSpamWords)

ham_messages.apply(extractHamWords)
from wordcloud import WordCloud
#Spam Word cloud



spam_wordcloud = WordCloud(width=600, height=400).generate(" ".join(spam_words))

plt.figure( figsize=(10,8), facecolor='k')

plt.imshow(spam_wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
#Ham word cloud



ham_wordcloud = WordCloud(width=600, height=400).generate(" ".join(ham_words))

plt.figure( figsize=(10,8), facecolor='k')

plt.imshow(ham_wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
# Top 10 spam words



spam_words = np.array(spam_words)

print("Top 10 Spam words are :\n")

pd.Series(spam_words).value_counts().head(n = 10)
# Top 10 Ham words



ham_words = np.array(ham_words)

print("Top 10 Ham words are :\n")

pd.Series(ham_words).value_counts().head(n = 10)
messages["messageLength"] = messages["text"].apply(len)

messages["messageLength"].describe()
f, ax = plt.subplots(1, 2, figsize = (20, 6))



sns.distplot(messages[messages["category"] == "spam"]["messageLength"], bins = 20, ax = ax[0])

ax[0].set_xlabel("Spam Message Word Length")



sns.distplot(messages[messages["category"] == "ham"]["messageLength"], bins = 20, ax = ax[1])

ax[0].set_xlabel("Ham Message Word Length")



plt.show()
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("english")



def cleanText(message):

    

    message = message.translate(str.maketrans('', '', string.punctuation))

    words = [stemmer.stem(word) for word in message.split() if word.lower() not in stopwords.words("english")]

    

    return " ".join(words)



messages["text"] = messages["text"].apply(cleanText)

messages.head(n = 10)    
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(encoding = "latin-1", strip_accents = "unicode", stop_words = "english")

features = vec.fit_transform(messages["text"])

print(features.shape)
def encodeCategory(cat):

    if cat == "spam":

        return 1

    else:

        return 0

        

messages["category"] = messages["category"].apply(encodeCategory)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, messages["category"], stratify = messages["category"], test_size = 0.2)
from sklearn.model_selection import cross_val_score

from sklearn.metrics import fbeta_score



from sklearn.naive_bayes import MultinomialNB

gaussianNb = MultinomialNB()

gaussianNb.fit(X_train, y_train)



y_pred = gaussianNb.predict(X_test)



print(fbeta_score(y_test, y_pred, beta = 0.5))