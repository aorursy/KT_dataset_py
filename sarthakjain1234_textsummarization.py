import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from nltk.stem.porter import PorterStemmer 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#Keras preprocessing models for text
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
#Load in data
news_summary = pd.read_csv("../input/news-summary/news_summary_more.csv", encoding='utf-8')
news_summary.head()
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}
print(contraction_mapping)
sample_text = "Democratic candidate Joe Biden returns to his native Pennsylvania for events in Bucks County and Luzerne County, two areas crucial to winning \
the important state. Sen. Kamala D. Harris (D-Calif.) has a speech in Cleveland, former president Barack Obama has a drive-in rally in Miami and singer \
Cher is holding a Biden event in Las Vegas."

complex_text = "Hey there mate!!? #I heard you're do##ing (Well) 'cause / I'd mightn't've been calling, seeing, or doing watching Y'all or Her but I needed to. How are you doing??"

punct_list = set(punctuation)

def preprocess_text(text, link):
    text = text.lower()
    split_text = text.split(" ")
    
    split_text = [word for word in split_text if word not in STOPWORDS]
    contracted_text = list()
    
    for word in split_text:
        if word in contraction_mapping:
            word = contraction_mapping[word]
            contracted_text.append(word)
        else:
            contracted_text.append(word)
    
    better_text = list()
    for word in contracted_text:
        word_text = "".join(ch for ch in word if ch not in punct_list)
        better_text.append(word_text)
    
    better_text = [word for word in better_text if len(word) > link]
    
    return (" ".join(better_text)).strip()

    

text = preprocess_text(complex_text, link = 3)
print(text, "\n")
sample_text = preprocess_text(sample_text, link = 3)
print(sample_text)


news_summary["text"] = news_summary["text"].apply(lambda x: preprocess_text(x, link = 3))
news_summary["headlines"] = news_summary["headlines"].apply(lambda x: '_START_ ' + preprocess_text(x, link = 1) + ' _END_' )
news_summary.head(10)
for ii in range(5):
    print("The preprocessed text: ", news_summary["text"][ii])
    print("The headlines: ", news_summary["headlines"][ii])
    print("\n")
max_length_text = 0
for row in news_summary["text"]:
    if len(row) >= max_length_text:
        max_length_text = len(row)
        
max_length_headlines = 0     
for row in news_summary["headlines"]:
    if len(row) >= max_length_headlines:
        max_length_headlines = len(row)

print(max_length_text)
print(max_length_headlines)
#Lets split the data
x_train, x_val, y_train, y_val = train_test_split(news_summary["text"], news_summary["headlines"], test_size = 0.1, random_state = 0, shuffle = True)
#Lets define the assertions:
assert len(x_train) == len(y_train)
assert len(x_val) == len(x_val)
print(len(x_train))
print(len(x_val))
x_vectorizer = Tokenizer()
x_vectorizer.fit_on_texts(list(x_train))

#convert summary sequences into integer sequences
x_train =  x_vectorizer.texts_to_sequences(x_train) 
x_val   =   x_vectorizer.texts_to_sequences(x_val)

#padding zero upto maximum length
x_train =  pad_sequences(x_train,  maxlen=max_length_text, padding='post') 
x_val = pad_sequences(x_val, maxlen=max_length_text, padding='post')



y_vectorizer = Tokenizer()
y_vectorizer.fit_on_texts(list(y_train))

#convert summary sequences into integer sequences
y_train =  y_vectorizer.texts_to_sequences(y_train) 
y_val   =   y_vectorizer.texts_to_sequences(y_val)

#padding zero upto maximum length
y_train =  pad_sequences(y_train,  maxlen=max_length_text, padding='post') 
y_val = pad_sequences(y_val, maxlen=max_length_text, padding='post')
