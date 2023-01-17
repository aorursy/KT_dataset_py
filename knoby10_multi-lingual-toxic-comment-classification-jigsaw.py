# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
comments= pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
comments.head()
comments.isnull().sum()
import seaborn as sb



from matplotlib import pyplot as plt



sb.heatmap(comments.isnull(),cbar=True)

plt.show()
# Number of classes present in each target variable

for i,j in comments[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].items():

    print(i,j.value_counts(),'',sep='\n')
val = comments[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
plt.rcParams['figure.figsize']=[20,8]

fig =plt.figure()



plt.subplot(2,2,1)

plot = plt.hist(val.toxic)

plt.title('toxic')

plt.subplot(2,2,2)

plt.hist(val.severe_toxic,color='red')

plt.title('severe_toxic')

plt.subplot(2,2,3)

plt.hist(val.obscene,color='orange')

plt.title('obscene')

plt.subplot(2,2,4)

plt.hist(val.threat,color='green')

plt.title('threat')



plt.show()
plt.rcParams['figure.figsize']=[20,8]

fig =plt.figure()



plt.subplot(2,2,1)

plt.hist(val.insult,color='purple')

plt.title('insult')

plt.subplot(2,2,2)

plt.hist(val.identity_hate,color='black')

plt.title('identity_hate')

plt.show()
from nltk.tokenize import word_tokenize

from string import punctuation

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
stop_nltk = stopwords.words("english")



stop_updated = stop_nltk + ["...","..","\n","\t","==","=","//","'",'D',',','en wikipedia org','https en wikipedia','wikipedia org wiki'] 
contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",

                     "can't": "cannot","can't've": "cannot have",

                     "'cause": "because","could've": "could have","couldn't": "could not",

                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",

                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",

                     "hasn't": "has not","haven't": "have not","he'd": "he would",

                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",

                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",

                     "I'd": "I would", "I'd've": "I would have","I'll": "I will",

                     "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",

                     "it'd": "it would","it'd've": "it would have","it'll": "it will",

                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",

                     "mayn't": "may not","might've": "might have","mightn't": "might not", 

                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",

                     "mustn't've": "must not have", "needn't": "need not",

                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",

                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",

                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",

                     "she'll": "she will", "she'll've": "she will have","should've": "should have",

                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",

                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",

                     "there'd've": "there would have", "they'd": "they would",

                     "they'd've": "they would have","they'll": "they will",

                     "they'll've": "they will have", "they're": "they are","they've": "they have",

                     "to've": "to have","wasn't": "was not","we'd": "we would",

                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",

                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",

                     "what'll've": "what will have","what're": "what are", "what've": "what have",

                     "when've": "when have","where'd": "where did", "where've": "where have",

                     "who'll": "who will","who'll've": "who will have","who've": "who have",

                     "why've": "why have","will've": "will have","won't": "will not",

                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",

                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",

                     "y'all'd've": "you all would have","y'all're": "you all are",

                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",

                     "you'll": "you will","you'll've": "you will have", "you're": "you are",

                     "you've": "you have"}
import re



contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))
# Function for expanding contractions

def expand_contractions(text,contractions_dict=contractions_dict):

 def replace(match):

    return contractions_dict[match.group(0)]

 return contractions_re.sub(replace, text)

# Function for removing the stopwords  and punctuations

lemm = WordNetLemmatizer()

def clean_text(sent):

    tokens = word_tokenize(sent.lower())

    lemmed = [lemm.lemmatize(term) for term in tokens \

               if term not in stop_updated and \

                term not in list(punctuation) and\

               len(term) > 2] 

    res = " ".join(lemmed)

    return res
#Expanding Contractions in the comments



comments['clean_text']=comments['comment_text'].apply(lambda x:expand_contractions(x))



comments['clean_text']=comments['clean_text'].apply(lambda x: re.sub('\w*\d\w*','', x))

comments['clean_text'] = comments.clean_text.apply(clean_text)
comments.head()
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
X = comments.clean_text.values

target = comments.toxic.values
X_train, X_test, y_train, y_test = train_test_split(X, target, train_size = 0.75,random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=3500)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

X_test_tfidf = tfidf_vectorizer.transform(X_test)
classifier = MultinomialNB()

classifier.fit(X_train_tfidf, y_train)

traget_prediction = classifier.predict(X_test_tfidf)

print('Accuaracy score for toxic comment classification:',accuracy_score(y_test,traget_prediction)*100)
traget_prediction