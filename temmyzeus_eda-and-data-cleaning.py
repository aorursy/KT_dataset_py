import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

#impore re for regular expression
import re

# import punctuations
from string import punctuation

#importing stopwords
from nltk.corpus import stopwords

#import the tokenizer
from nltk.tokenize import RegexpTokenizer

#importing the Stemmer
from nltk.stem import PorterStemmer,SnowballStemmer

#importing the Lemmatizer
from nltk.stem import WordNetLemmatizer

# import wordcloud for text visualization
from wordcloud import WordCloud
import chardet
with open("../input/sms-spam-collection-dataset/spam.csv" , "rb") as f:
    enc = chardet.detect(f.read(1000000))
    
print(enc)
data = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv" , encoding="Windows-1252")
data.head()
print(f"Shape of Data: {data.shape}")
data.isna().sum()
# P.S: The display keyword only works in the jupyter notebook enviroment and would raise an error in an actual python file
display(data[data["Unnamed: 2"].notnull()].head())
display(data[data["Unnamed: 3"].notnull()].head())
display(data[data["Unnamed: 4"].notnull()].head())
#for every line where feature "Unnamed: 2" is not_null, we add Feature "Unnamed: 2" to "Text" feature
data["v2"][data["Unnamed: 2"].notna()] = data["v2"][data["Unnamed: 2"].notna()] + data["Unnamed: 2"][data["Unnamed: 2"].notna()]
#for every line where feature "Unnamed: 3" is not_null, we add Feature "Unnamed: 3" to "Text" feature
data["v2"][data["Unnamed: 3"].notna()] = data["v2"][data["Unnamed: 3"].notna()] + data["Unnamed: 3"][data["Unnamed: 3"].notna()]
#for every line where feature "Unnamed: 4" is not_null, we add Feature "Unnamed: 4" to "Text" feature
data["v2"][data["Unnamed: 4"].notna()] = data["v2"][data["Unnamed: 4"].notna()] + data["Unnamed: 4"][data["Unnamed: 4"].notna()]
#drop feature "A","B", and "C"
data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"] , axis=1, inplace=True)
data.columns = ["label","text"]#rename columns
data.head()
data["label"].value_counts().plot(kind="bar")
plt.show()
fig , ax = plt.subplots(nrows=6 , ncols=1,figsize=(10,50))
def word_length_count(text):
    return len(text.split())
data["word_length"] = data["text"].apply(word_length_count)
sns.kdeplot(data["word_length"][data["label"] == "ham"] , shade=True , label="ham" , ax = ax[0])
sns.kdeplot(data["word_length"][data["label"] == "spam"] , shade=True , label="spam" , ax=ax[0])
ax[0].set_title("Distribution of Words Count", fontsize=14)
ax[0].set_xlabel("Number of Words in Text" , fontsize=14)
ax[0].set_ylabel("Frequency Rate" , fontsize=14)

def char_length_count(text):
    return len(text)
data["characters_count"] = data["text"].apply(char_length_count)
sns.kdeplot(data["characters_count"][data["label"] == "ham"] , shade=True , label="ham" , ax = ax[1])
sns.kdeplot(data["characters_count"][data["label"] == "spam"] , shade=True , label="spam" , ax=ax[1])
ax[1].set_title("Distribution of Characters Count", fontsize=14)
ax[1].set_xlabel("Number of Characters in Text" , fontsize=14)
ax[1].set_ylabel("Frequency Rate" , fontsize=14)

def punct_count(text):
    return len([char for char in text if char in punctuation])
data["punctuation_count"] = data["text"].apply(punct_count)
sns.kdeplot(data["punctuation_count"][data["label"] == "ham"] , shade=True , label="ham" , ax = ax[2])
sns.kdeplot(data["punctuation_count"][data["label"] == "spam"] , shade=True , label="spam" ,ax = ax[2])
ax[2].set_title("Distribution of Punctuations Count", fontsize=14)
ax[2].set_xlabel("Number of Punctuations in Text" , fontsize=14)
ax[2].set_ylabel("Frequency Rate" , fontsize=14)

def numbers_count(text):
    return len([char for char in text if char.isnumeric()])
data["numbers_count"] = data["text"].apply(numbers_count)
sns.kdeplot(data["numbers_count"][data["label"] == "ham"] , shade=True , label="ham" , ax = ax[3])
sns.kdeplot(data["numbers_count"][data["label"] == "spam"] , shade=True , label="spam",ax = ax[3])
ax[3].set_title("Distribution of Numerical Characters Count", fontsize=14)
ax[3].set_xlabel("Number of Numerical String in Text" , fontsize=14)
ax[3].set_ylabel("Frequency Rate" , fontsize=14)

def stopwords_counts(text):
    return len([word for word in text.split() if word in stopwords.words("english")])
data["stopwords_count"] = data["text"].apply(stopwords_counts)
sns.kdeplot(data["stopwords_count"][data["label"] == "ham"] , shade=True , label="ham" , ax = ax[4])
sns.kdeplot(data["stopwords_count"][data["label"] == "spam"] , shade=True , label="spam" , ax = ax[4])
ax[4].set_title("Distribution of Stopwords Count", fontsize=14)
ax[4].set_xlabel("Number of Stopwords in Text" , fontsize=14)
ax[4].set_ylabel("Frequency Rate" , fontsize=14)

def non_stopwords_counts(text):
    return len([word for word in text.split() if word not in stopwords.words("english")])
data["non_stopwords_count"] = data["text"].apply(non_stopwords_counts)
sns.kdeplot(data["non_stopwords_count"][data["label"] == "ham"] , shade=True , label="ham" , ax = ax[5])
sns.kdeplot(data["non_stopwords_count"][data["label"] == "spam"] ,  shade=True , label="spam" , ax = ax[5])
ax[5].set_title("Distibution of Non-Stopwords Count", fontsize=14)
ax[5].set_xlabel("Number of Non-Stopwords in Text" , fontsize=14)
ax[5].set_ylabel("Frequency Rate" , fontsize=14)
plt.show()
for _ in data["text"]:
    print(_)
#change text to lowercase
data["text"] = data["text"].str.lower()
display(data[data["text"].str.contains("http")].head())
display(data[data["text"].str.contains("www")].head())
# write a function to remove web links
def remove_html(text):
    text = re.sub(r"http://.+\.com" , "" , text)
    text = re.sub(r"http://\S+" , "" , text)
    text = re.sub(r"http://[A-Za-z0-9./?= *&:-]+" , "" , text)
    text = re.sub(r"http.+\.com" , "" , text)
    text = re.sub(r"http//[A-Za-z0-9./?= *&:-]+" , "" ,text)
    text = re.sub(r"www[.A-Za-z0-9/+-]+" , "" , text)
    return text
data["text"] = data["text"].apply(remove_html)
#Remove numbers and words which are joined with numbers such that the number removal makes them useless
def remove_numbers(text):
    text = re.sub(r"[å£A-Za-z0-9]*\d+[A-Za-z0-9]*" , "" , text)
    return text
data["text"] = data["text"].apply(remove_numbers)
# to view some datapoints with charcters such as "å"
data[data["text"].str.contains("å")].sample(5)
def remove_fancy_text(word):
    if word.isascii() == False:
        return ""
    else:
        return word
data["text"] = data["text"].apply(lambda x: " ".join([remove_fancy_text(word) for word in x.split()]))
for _ in data["text"]:
    print(_)
data[data["text"].str.contains("@")].head()
def remove_mails(text):
    text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+(\.[a-zA-Z]{2,5})" , "" , text)
    return text
data["text"] = data["text"].apply(remove_mails)
# view amils with "@" in them 
data[data["text"].str.contains("@")]
tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S=+')
data["text"] = data["text"].apply(lambda x : tokenizer.tokenize(x))
def remove_stopwords(text):
    return [word for word in text if word not in stopwords.words("english")]
data["text"] = data["text"].apply(remove_stopwords)
def remove_puncts(text):
    return [word for word in text if word not in punctuation]
data["text"] = data["text"].apply(remove_puncts)
data["text_length"] = data["text"].apply(lambda x: len(x))
data.head()
data[data["text_length"] > data["non_stopwords_count"]]
data["text"] = data["text"].apply(lambda x: [word for word in x if len(word) >= 2])
lemma = WordNetLemmatizer()
data["text"] = data["text"].apply(lambda x: " ".join([lemma.lemmatize(word) for word in x]))
# stemmer = SnowballStemmer(language="english")
# data["text"].apply(lambda x: " ".join([stemmer.stem(word) for word in x]))
spam_corpus = ""
for text in data[data["label"] == "spam"]["text"]:
    spam_corpus += " " + text
ham_corpus = ""
for text in data[data["label"] == "ham"]["text"]:
    ham_corpus += " " + text
spam_wordcloud = WordCloud(background_color="white").generate(spam_corpus)
ham_wordcloud = WordCloud(background_color="white").generate(ham_corpus)
fig,ax= plt.subplots(nrows=1,ncols=2,figsize=(20,100))
ax[0].imshow(spam_wordcloud)
ax[0].set_title("Spam Text" , fontsize=20)
ax[0].axis(False)
ax[0].grid(False)
plt.savefig("Spam Text")
ax[1].imshow(ham_wordcloud)
ax[1].set_title("Ham Text" , fontsize=20)
ax[1].axis(False)
ax[1].grid(False)
plt.show()
