# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import re

import string #for removing punctuations

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer



import gensim



from gensim.models import Word2Vec

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

df_test =pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
df_train.head()
df_test.head()
print(df_train.shape)

print(df_test.shape)

print('------------------------')

print(df_train.keyword.isnull().sum())

print(df_test.keyword.isnull().sum())

print('------------------------')

print(df_train.target.value_counts())
df_train.target.value_counts().plot(kind='pie',autopct='%1.0f%%')
keyword_list=df_train.keyword.unique()

print(len(keyword_list))

def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)
pd.set_option('display.max_colwidth', -1)

df_train[df_train['id']== 251]
df_train['text']=df_train['text'].apply(lambda x : remove_URL(x))

df_test['text']=df_test['text'].apply(lambda x : remove_URL(x))
pd.set_option('display.max_colwidth', -1)

df_train[df_train['id']== 251]
def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)
df_train['text']=df_train['text'].apply(lambda x : remove_html(x))

df_test['text']=df_test['text'].apply(lambda x : remove_html(x))
def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



remove_emoji("Omg another Earthquake 😔😔")
df_train['text']=df_train['text'].apply(lambda x: remove_emoji(x))

df_test['text']=df_test['text'].apply(lambda x: remove_emoji(x))
def remove_punct(text):

    table=str.maketrans('','', string.punctuation)

    return text.translate(table)



remove_punct("I am ,a #king")
df_train['text']=df_train['text'].apply(lambda x : remove_punct(x))

df_test['text']=df_test['text'].apply(lambda x : remove_punct(x))
pd.set_option('display.max_colwidth', -1)

df_train
df_train['text']=df_train['text'].apply(lambda x : re.sub('[^a-zA-Z]', ' ', x))

df_test['text']=df_test['text'].apply(lambda x : re.sub('[^a-zA-Z]', ' ', x))



df_train['text']=df_train['text'].apply(lambda x : x.lower())

df_test['text']=df_test['text'].apply(lambda x : x.lower())



#df_train['text'] = df_train['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in set(stopwords.words('english'))]))

#df_test['text'] = df_test['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in set(stopwords.words('english'))]))

df_train

!pip install pyspellchecker
from spellchecker import SpellChecker



spell = SpellChecker()

def correct_spellings(text):

    corrected_text = []

    misspelled_words = spell.unknown(text.split())

    for word in text.split():

        if word in misspelled_words:

            corrected_text.append(spell.correction(word))

        else:

            corrected_text.append(word)

    return " ".join(corrected_text)

        

text = "corect me plese"

correct_spellings(text)
#df_train['text']=df_train['text'].apply(lambda x : correct_spellings(x))

#df_test['text']=df_test['text'].apply(lambda x : correct_spellings(x))
list_of_sent_train=[]

list_of_sent_test=[]



for i in range(0,7613):

    tweet = df_train['text'][i]

    tweet = tweet.split()

    list_of_sent_train.append(tweet) 

    

for i in range(0,3263):

    tweet = df_test['text'][i]

    tweet = tweet.split()

    list_of_sent_test.append(tweet)    
w2v_model_train=gensim.models.Word2Vec(list_of_sent_train,min_count=5,size=50, workers=4)

w2v_words1 = list(w2v_model_train.wv.vocab)



model1 = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8)

tf_idf_matrix = model1.fit_transform(df_train['text'].values)

# we are converting a dictionary with word as a key, and the idf as a value

dictionary = dict(zip(model1.get_feature_names(), list(model1.idf_)))

tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list

row=0;

for sent in list_of_sent_train: # for each review/sentence

    sent_vec = np.zeros(50) # as word vectors are of zero length

    weight_sum =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        try:

            vec = w2v_model_train.wv[word]

            # obtain the tf_idfidf of a word in a sentence/review

            tf_idf = dictionary[word]*(sent.count(word)/len(sent))

            sent_vec += (vec * tf_idf)

            weight_sum += tf_idf

        except:

            pass

    sent_vec /= weight_sum

    tfidf_sent_vectors.append(sent_vec)

    row += 1  
X_train = tfidf_sent_vectors

X_train = np.nan_to_num(X_train)

y=df_train['target']
w2v_model_test=gensim.models.Word2Vec(list_of_sent_test,min_count=5,size=50, workers=4)

w2v_words2 = list(w2v_model_test.wv.vocab)



model2 = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8)

tf_idf_matrix = model2.fit_transform(df_test['text'].values)

# we are converting a dictionary with word as a key, and the idf as a value

dictionary = dict(zip(model2.get_feature_names(), list(model2.idf_)))



tfidf_sent_vectors_test = []; # the tfidf-w2v for each sentence/review is stored in this list

row=0;

for sent in list_of_sent_test: # for each review/sentence

    sent_vec = np.zeros(50) # as word vectors are of zero length

    weight_sum =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        try:

            vec = w2v_model_test.wv[word]

            # obtain the tf_idfidf of a word in a sentence/review

            tf_idf = dictionary[word]*(sent.count(word)/len(sent))

            

            sent_vec += (vec * tf_idf)

            weight_sum += tf_idf

        except:

            pass

    sent_vec /= weight_sum

    tfidf_sent_vectors_test.append(sent_vec)

    row += 1  
x_test = tfidf_sent_vectors_test

x_test = np.nan_to_num(x_test)
X_tra, X_val, y_tra, y_val = train_test_split(X_train, y,stratify=df_train['target'], test_size=0.2, random_state=42)
optimal_learners = 1000

optimal_depth = 13

optimal_childweight = 1

optimal_lossfunc = 0.1

optimal_subsampl = 0.5

optimal_lr = 0.4

clffinal = XGBClassifier(objective = "binary:logistic",alpha=1,max_delta_step=1,n_estimators= optimal_learners, max_depth=optimal_depth,

                         min_child_weight=optimal_childweight, gamma=optimal_lossfunc, subsample= optimal_subsampl,

                         learning_rate= optimal_lr,eval_metric ="auc",scale_pos_weight=1).fit(X_tra, y_tra)
y_pred=clffinal.predict(X_val)
print(confusion_matrix(y_val,y_pred))

print(classification_report(y_val,y_pred))

print(accuracy_score(y_val,y_pred))
from sklearn.metrics import roc_auc_score



print("validation score :" + str(roc_auc_score(y_val,y_pred)))
Test_Prediction = clffinal.predict(x_test)

sub_df = pd.DataFrame({"id":df_test["id"].values})

sub_df["target"] = Test_Prediction

sub_df.to_csv("submission_final.csv", index=False)