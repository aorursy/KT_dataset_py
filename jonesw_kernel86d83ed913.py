import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))
# import dataset

# column is index of data, ignore

df = pd.read_csv('../input/train.txt', index_col=0)

df.head()
from nltk.tokenize import TweetTokenizer



# data cleansing

# replace "class" as tone: Neg -> False(-ve); Pos -> True(+ve)

df['class'] = [True if i == 'Pos' else False for i in df['class']]

# add word count

tknzr = TweetTokenizer()

df['word_count'] = [len(tknzr.tokenize(record)) for record in df['text']]
df.head()
print("Total Neg comment:", len(df[df['class']==False]))

print("Total Pos comment:", len(df[df['class']==True]))
sns.boxplot(y="word_count", x='class', data=df) # relationship between word count and class
from wordcloud import WordCloud, STOPWORDS



sw = set(STOPWORDS)

wc = WordCloud(max_words=50, background_color="white",margin=2, stopwords=sw)
plt.imshow(wc.generate(' '.join(df[df['class'] == False]['text']))) 

plt.axis("off")

plt.title("Frequent Words appeared in Neg Text")
plt.imshow(wc.generate(' '.join(df[df['class'] == True]['text'])))

plt.axis("off")

plt.title("Frequent Words appeared in Pos Text")
training_data = df.iloc[:, 0:2]

training_data.tail()
# data cleansing

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer 

from nltk.stem import PorterStemmer 

import re





def remove_numbers(text):

    '''function for remove 0-9 in text'''

    return re.sub(r'[0-9\.]+', '', text) 



def remove_stopwords(text):

    '''a function for removing the stopword'''

    # extracting the stopwords from nltk library

    sw = stopwords.words('english')

    sw.extend(

        ["ford", "focus", "br", "quot", "good", "great", "new"]

    )

    # removing the stop words and lowercasing the selected words

    text = [word.lower() for word in text.split() if word.lower() not in sw]

    # joining the list of words with space separator

    return " ".join(text)



def lemmatize(text):

    '''function for lemmatize'''    

    lemmatizer = WordNetLemmatizer()

#     ps = PorterStemmer()

#     text = " ".join([ps.stem(i) for i in text.split(" ")])

    return " ".join([lemmatizer.lemmatize(i) for i in text.split(" ")])



# Apply data cleaning

# punctuation is already cleaned before obtain

training_data['text'] = training_data['text'].apply(remove_numbers)

training_data['text'] = training_data['text'].apply(remove_stopwords)

training_data['text'] = training_data['text'].apply(lemmatize)

training_data['text'] = training_data['text'].apply(remove_stopwords)
training_data.tail()
wc2 = WordCloud(max_words=20, background_color="white",margin=2)
import nltk
neg_text = ' '.join(training_data[training_data['class'] == False]['text'])

neg_token = nltk.word_tokenize(neg_text)

plt.imshow(wc2.generate(neg_text))

plt.axis("off")

plt.title("Frequent Words appeared in Pos Text")
pos_text = ' '.join(training_data[training_data['class'] == True]['text'])

pos_token = nltk.word_tokenize(pos_text)

plt.imshow(wc2.generate(pos_text))

plt.axis("off")

plt.title("Frequent Words appeared in Pos Text")
_neg_token = ' '.join([i[0] for i in nltk.pos_tag(neg_token) if "JJ" in i[1]])

plt.imshow(wc2.generate(_neg_token))

plt.axis("off")

plt.title("Frequent Adjective appeared in Pos Text")
_pos_token = ' '.join([i[0] for i in nltk.pos_tag(pos_token) if "JJ" in i[1]])

plt.imshow(wc2.generate(_pos_token))

plt.axis("off")

plt.title("Frequent Adjective appeared in Pos Text")
from sklearn.pipeline import make_pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score



classifier = make_pipeline(TfidfVectorizer(), TruncatedSVD(4500), SVC(gamma='scale', class_weight='balanced'))

result = cross_val_score(classifier, training_data['text'].values.tolist(), training_data['class'].values.tolist(), cv=10, groups=df['class'])

print(result)

print("Average:", result.mean())