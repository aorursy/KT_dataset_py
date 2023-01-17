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
!pip install autocorrect 
import nltk
import re
import numpy as np
import pandas as pd
from unidecode import unidecode
from autocorrect import Speller
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus.reader import CategorizedPlaintextCorpusReader
df_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
df_train.head(15)
# in our dataset we have a lot of noise that would not be good for analysis
# hashtags, mentions, urls, emojis and so on 
print(df_train.isnull().sum())

# import missingno as msno 
# msno.bar(df_train) 

# as we can see there are a lot of missing data of location so I would better better drop this col
# 'keyword' cal might be used later on
def remove_from_text(col):
    
    col = col.str.replace('åÊ', ' ', case=False) # we can replace not unicode chars using .replace
    col = col.str.replace('&lt', ' ', case=False)
    col = col.str.replace('&gt', ' ', case=False)
    col = col.str.replace('Û÷', ' ', case=False)
    col = col.str.replace('Ûª', '\'', case=False)
    col = col.str.replace('Û', '\'', case=False)
    col = col.str.replace('&amp', ' ', case=False)
    col = col.str.replace('ï', ' ', case=False)
    col = col.str.replace('ó', ' ', case=False)
    col = col.str.replace('ò', ' ', case=False)
    col = col.str.replace('\x89', '', case=False)
    
   
    col = col.str.lower()                  #everything to lower case
    col = [unidecode(x) for x in col]      #everything to unicode
    
    
    col = [re.sub(r"http\S+|www.\S+", ' ', x) for x in col]  #remove urls
    col = [re.sub(r"@(\w+)", ' ', x) for x in col]           #remove mentions
    col = [re.sub(r"#", ' ', x) for x in col]                #remove hashtags
    col = [re.sub(r"\w*\d\w*", ' ', x) for x in col]         #remove nums
    col = [re.sub(r"[^\w\d'\s]+", ' ', x) for x in col]      #remove punctuation
    col = [re.sub(r"\'s", ' is', x) for x in col]            #all 's to is
    col = [re.sub(r"\'", '', x) for x in col]                #remove ap. that are left
    col = [re.sub(r"_", ' ', x) for x in col]                #remove underscore
    col = [re.sub(r"-", ' ', x) for x in col]                #remove dash
        
    return col
# define needed functions
def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{1,}")
    word = pattern.sub(r"\1\1", text)
    return word

spell = Speller(lang='en')
def spell_check(text):
    spell_correct_text = spell(text) 
    return spell_correct_text


stop_words = stopwords.words('english')
stop_words.append('sorry') #to regular list of stop words add
stop_words.append('like')  #these two the most common words
stop_words.append('im')    #it is 'I am' and 'You' 
stop_words.append('u')     #wich shold be removed as stopword

def stop_word(text):
    without_stopw = [word for word in text if not word in stop_words]
    return without_stopw

def word_stemmer(text):
    stemmer = PorterStemmer()
    stem_text = [stemmer.stem(i) for i in text]
    return stem_text

def word_lemmatizer(text):
    lemmatizer = WordNetLemmatizer()
    lem_text = [lemmatizer.lemmatize(i) for i in text]
    return lem_text
def nlp_transform_df(df):
    
#      remove all defined noise (@, #, punctuation and so on)
    
    df['text'] = remove_from_text(df['text'])
     
    
#      SPELL CHECK CAN BE VERY SLOW PROCESS IN ORDER TO DO IT FASTER (about 5 min) WE NEED TO DO FOLLOWING STEPS
    
#      1. Lets fix all trailing letters so word would have a chance to be corrected
#      for example: goooooooal, finalllllly, looooooool
#      reduce_lengthening(arg) function is defined above

    df['text'] = [[reduce_lengthening(x) for x in y.split()] for y in df['text']]
    
#     2. Now we creat a list of all distinct(about 14K words)
#     ittirate throu each word in text and append all words that are not already in a list

    words_from_text = []
    for line in df['text']:
        for w in line:
            if w not in words_from_text:
                words_from_text.append(w)

#     3. This step is to retrieve all words that are not existing in english dictionary from list above
#     import english words from nltk corpus 
#     ittirate throu words_from_text list and check weather we have it in english dictionary
#     if not we store it in a list not_word_from_dict

    from nltk.corpus import words
    d = words.words()
    not_word_from_dict = []
    for w in words_from_text:
            if w not in d:
                not_word_from_dict.append(w) 
                
#     4. Now we apply function spell_check(arg) to every item from not_word_from_dict  
#     and store as corected_words
#     only words that are possible to correct will be corrected

    corected_words = [spell_check(x) for x in not_word_from_dict]


#     5. Here we create dictionary: key   - initial word from text, 
#                                   value - corrected word

    dict_correct_speling = dict(zip(not_word_from_dict, corected_words))

    for key in dict_correct_speling.copy():
         if key == dict_correct_speling[key]:
            dict_correct_speling.pop(key)
                  
#     6. Last step is to apply dicionary to out text
#     while performing reduce_lengthening(arg) we split our string into sep. words 
#     so lets join them back as str
#     and apply function replace_with_corrected_word(df) defined above

    df['text'] = [' '.join(x) for x in df['text']]
    
    def replace_with_corrected_word(df):
        for i in range(len(df)):   
            for w in df.loc[i, 'text'].split():
                if w in dict_correct_speling:
                    pattern = r'\b'+w+r'\b'
                    df.loc[i, 'text'] = re.sub(pattern, dict_correct_speling[w], df.loc[i, 'text'])
        return df
    
    df = replace_with_corrected_word(df)
    
#      After spell check is done we can perform basic NLP steps to transform our data

    tokenizer = RegexpTokenizer(r'\w+')
    df['text'] = df['text'].apply(lambda x: tokenizer.tokenize(x))         #tokenize
    df['text'] = df['text'].apply(lambda x: stop_word(x))                  #remove stop words
    df['text'] = df['text'].apply(lambda x: word_lemmatizer(x))            #lemmatize
    df['text'] = df['text'].apply(lambda x: word_stemmer(x))               #stemm
    df['text'] = [' '.join(x) for x in df['text']]                         #join tokens as str
    
    
    df['text'] = df['text'].fillna('')  #in case we have Null vals we replace it with empty str
    
    return df
# import dataset
df_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
df_train.head(2)
# apply transform function
%time df_train = nlp_transform_df(df_train)
df_train.drop(['keyword', 'location'], axis = 1, inplace = True)
df_train.head(2)
df_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
df_test.head(2)
# apply transform function
%time df_test = nlp_transform_df(df_test)
df_test.drop(['keyword', 'location'], axis = 1, inplace = True)
df_test.head(2)
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
# def vectorizer
vectorizer = CountVectorizer(ngram_range = (1,1), min_df = 2, analyzer='word') #checking ngrams which is better
train_vectors = vectorizer.fit_transform(df_train['text'])
test_vectors = vectorizer.transform(df_test['text'])

print(train_vectors.shape) #train
print(test_vectors.shape)  #test
# voting classifier
classifiers = [
    ('RandomForest', RandomForestClassifier(n_estimators=100)),
    ('BernoulliNB', BernoulliNB()),
    ('DecisionTree', DecisionTreeClassifier()),
    ('NuSVC', NuSVC()),
    ('LogisticRegression', LogisticRegression()),
    ('MultinomialNB', MultinomialNB()),
    ('SVC', SVC()),
]
clf = VotingClassifier(classifiers, n_jobs=-2)

clf = VotingClassifier(classifiers, n_jobs=-2)

# fit 
clf.fit(train_vectors, df_train['target'])

# make a prediction on our test
target = clf.predict(test_vectors)
# Found answers online so we can check how well our model perform
#or can find out through submitt on kaggle
answers_txt = open('/kaggle/input/answers-from-web/answers.txt', 'r')
answers = answers_txt.read().split()
df = pd.DataFrame(answers)
df_answer = pd.DataFrame(df.iloc[1::2]).reset_index()[0]

df_target = pd.DataFrame()
df_target['answer'] = df_answer.astype(int)
df_target['target'] = target

print('Wrong Predictions: ', len(df_target[df_target['target'] != df_target['answer']]), ' out of ', len(df_target))
print('Accuracy of clf: ', metrics.accuracy_score(df_target['answer'], target))
print('Report: ', classification_report(df_target['answer'], target))
df_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
df_submission['target'] = target
df_submission.to_csv('submission.csv', index=False)
df_submission
