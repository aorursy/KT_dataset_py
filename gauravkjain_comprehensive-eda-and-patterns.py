# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

INPUT_DIR="/kaggle/input/nlp-getting-started/"

train=pd.read_csv(INPUT_DIR+"train.csv")

test=pd.read_csv(INPUT_DIR+"test.csv")
test.shape
train.head()
train.shape
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="whitegrid")

from collections import Counter

import re

import string

import spacy

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English

from nltk import pos_tag

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



smileys_array=[':-)', ':)', ';-)', ';)']



#print all the unique values in 

def print_col_unique_vals(df_col):

    tot=0

    for val in df_col.unique():

        print(val)

        tot=tot+1

    print(tot)

    

def print_all_cols(df):

    cols=df.columns.values

    for col in cols:

        print("****************"+col+"*******************")

        #print_col_unique_vals(df[col])

        print(df[col].value_counts())

        print("******************************************")

        

def get_words_count_from_col(df, col_name, split_rule='\s+'):

    new_col_name=col_name+'_word_count'

    df[new_col_name]=df[col_name].apply(lambda x:len(str(x).split(split_rule)))

    return df  

def print_text_df_statistical_features(df, text_col, target):

    #Average count of phrases per sentence in train/target

    #avg_count=df.groupby(target)[text_col].count().mean()

    #print("Average count of phrases per sentence in train/target is {0:.0f}".format(avg_count))

    if(target is not None):

        print('Number of Total Data Samples: {}. Number of Target Classes: {}.'.format(df.shape[0], len(df[target].unique())))

        print('\nData Distribution vs Classes:',Counter(df[target]))

    print('\nAverage word length of phrases in training dataset is {0:.0f}.'.format(np.mean(df[text_col].apply(lambda x: len(x.split('\s+'))))))



#Easy helper to search for all regular expressions in given text

def get_regexp_frequencies(regexp, text):

    return len(re.findall(regexp, text))



#Easy helper to search and replace all regular expressions in given text

def replace_regexp(regexp, text, replacement='\s'):

    return re.sub(regexp, replacement, text)



# Easy helper to get all the character counts

def get_char_count(df, text_col, search_char):

    return df[text_col].apply(lambda x: x.count(search_char))



def get_punctuation_count(df, text_col, search_chars=string.punctuation):

    return df[text_col].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))



def get_stopwords_count(df, text_col, stopwords):

    return df[text_col].apply(lambda x: len([w for w in str(x).lower().split('\s+') if w in stopwords]))



#def get_hash

    

def generate_text_numerical_features(df, text_col, target,stopwords=STOP_WORDS, verbose=True):

    text_col_df=df[text_col]

    #create total characters features in text column per text row

    df['chars_count']        = text_col_df.apply(len)

    

    #create total numer of words features in text column per text row 

    df['words_count']          = text_col_df.apply(lambda x: len(x.split(' ')))

    

    #create total numer of words features in text column per text row 

    df['CAPS_count']      = text_col_df.apply(lambda x: sum(1 for c in x if c.isupper()))

    

    #create total numer of exclmation marks in text column per text row 

    df['!_marks_count']  = get_char_count(df,text_col,'!')

    

    #create total numer of question marks in text column per text row 

    df['?_marks_count']  = get_char_count(df,text_col,'?')

    

    #create total number of hash marks in text column per text row 

    df['#_tags_count']  = text_col_df.apply(lambda x: len([x for x in x.split(' ') if x.startswith('#')]))

    #get_char_count(df,text_col,'#')

    

    #create total number of punctuations in text column per text row 

    df['punctuation_count'] = text_col_df.apply(lambda x: get_regexp_frequencies("[^\w\s]", x))

    

    df['stopwords_count'] = get_stopwords_count(df,text_col,stopwords)

    

    df['mean_word_len'] = text_col_df.apply(lambda x: np.mean([len(w) for w in str(x).split('\s+')]))

    

    df['unique_words_count'] = text_col_df.apply(

    lambda comment: len(set(w for w in comment.split('\s+'))))

    

    df['smilies_count'] = text_col_df.apply(lambda x: sum(x.count(w) for w in smileys_array))

    

    # Count number of \n

    df["\n_count"] = text_col_df.apply(lambda x: get_regexp_frequencies(r"\n", x))

    

    # Check for time stamp

    df["has_timestamp"] = text_col_df.apply(lambda x: get_regexp_frequencies(r"\d{2}|:\d{2}", x))

    

    # Check for http links

    df["has_http"] = text_col_df.apply(lambda x: get_regexp_frequencies(r"http[s]{0,1}://\S+", x))

    

    #Number of digits

    df['digit_count'] = df[text_col].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

    

    return df





def tag_part_of_speech(text):

    text_splited = text.split('\s+')

    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]

    text_splited = [s for s in text_splited if s]

    pos_list = pos_tag(text_splited)

    noun_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])

    adjective_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])

    verb_count = len([w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])

    return[noun_count, adjective_count, verb_count]



def topNFrequentWords(df, text_col, N=10):

    return pd.Series(' '.join(df[text_col]).split('\s+')).value_counts()[:N]

    



def printTopNFrequentWords(df, text_col, N=10):

    print(topNFrequentWords(df, text_col, N=N))

    

def drawWordCloud(df, text_col, clean_stopwords=True):

    

    textstr=''.join(x for x in df[text_col])

    

    if(clean_stopwords==True):

        stopwords=set(STOPWORDS)

        wordcloud = WordCloud(background_color="black").generate(textstr)

    else:

        wordcloud = WordCloud(background_color="black").generate(textstr)

    

    plt.figure(figsize=(20,10))

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.show()

    

def clean_urls(df, text_col, new_text_col):

    url_regex='http[s]?://\S+'

    #url_regex=r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?'

    rows=[]

    for row in df[text_col]:

        rows.append(re.sub(url_regex, '', row))

    df[new_text_col]= rows

    return df



def clean_digits(df, text_col, new_text_col):

    df[new_text_col] = df[new_text_col].str.replace('\d+', '')

    #df[new_text_col]=df[new_text_col].apply((filter(lambda c: not c.isdigit(), word)) for word in text_splited )

    return df



def clean_stopwords(df, text_col, stopwords, new_text_col):

    df[new_text_col]=df[text_col].apply(lambda x: ' '.join(word for word in x.split('\s+') if word not in stopwords) )

    return df



#https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a

def preprocess(ReviewText, extraPatsToRemove=[]):

    ReviewText = ReviewText.str.replace("(<br/>)", " ")

    ReviewText = ReviewText.str.replace('(<a).*(>).*(</a>)', ' ')

    ReviewText = ReviewText.str.replace('(&amp)', ' ')

    ReviewText = ReviewText.str.replace('(&gt)', ' ')

    ReviewText = ReviewText.str.replace('(&lt)', ' ')

    ReviewText = ReviewText.str.replace('(\xa0)', ' ')  

    ReviewText = ReviewText.str.replace(';', '')

    #for pattern in extraPatsToRemove:

     #   print(pattern)

      #  ReviewText = ReviewText.str.replace(str(pattern), ' ')

    return ReviewText

    

def process_dataframe_summarize_analysis(df, text_col, target_col, stopwords, extraPatsToRemove=[' '], N=10):

    new_text_col='clean_text'

    print("************START SUMMARIZATION\n\n")

    print_text_df_statistical_features(df=df, text_col=text_col, target=target_col)

    df=generate_text_numerical_features(df=df, text_col=text_col, target=target_col)

    print("\n\n")

    print("******Data head***********************")

    print(df.head(1).T)

    print("\n\n")

    print("************Top 10 Frequent words***********************")

    printTopNFrequentWords(df, text_col, N=10)

    print("\n\n")

    drawWordCloud(df, text_col)

    print("\n\n")

    print("************RAW WORD CLOUD as in DATASET*****************")

    df=clean_urls(df, text_col, new_text_col)

    print("\n\n")

    print("************WORD CLOUD after URLS Cleaning*****************")

    drawWordCloud(df, new_text_col)

    print("\n\n")

    print("************Cleaning Stopwords*****************")

    df=clean_stopwords(df, new_text_col, stopwords=stopwords,new_text_col=new_text_col)

    print("\n\n")

    print("************WORD CLOUD after STOPWORDS Cleaning*****************")

    drawWordCloud(df, new_text_col)

    print("\n\n")

    print("************Top 10 Frequent words***********************")

    printTopNFrequentWords(df, new_text_col, N=10)

    print("\n\n")

    print("************PRE PROCESSING*****************")

    df[new_text_col]=preprocess(df[new_text_col],extraPatsToRemove=extraPatsToRemove)

    print("************WORD CLOUD after PREPROCESSING*****************")

    drawWordCloud(df, new_text_col)

    print("\n\n")

    print("************Top 10 Frequent words***********************")

    printTopNFrequentWords(df, new_text_col, N=10)

    print("\n\n")

    

    print("************REMOVE DIGITS*****************")

    df=clean_digits(df, text_col, new_text_col)

    print(df.head())

    print("************WORD CLOUD after REMOVING DIGITS*****************")

    drawWordCloud(df, new_text_col)

    print("\n\n")

    print("************Top 10 Frequent words***********************")

    printTopNFrequentWords(df, new_text_col, N=10)

    print("\n\n")

    return df
def parallelize_dataframe(df, func):

    df_split = np.array_split(df, num_partitions)

    pool = Pool(num_cores)

    df = pd.concat(pool.map(func, df_split))

    pool.close()

    pool.join()

    return df



def multiply_columns(data):

    data['length_of_word'] = data['species'].apply(lambda x: len(x))

    return data

train=pd.read_csv(INPUT_DIR+"train.csv")

stopwords=set(STOPWORDS)

extraPatsToRemove=['??', '-', '_', '...']

stopwords.update(['THE', 'the', 'I', '-', 'The', ';', 'A', 'I\'m', '_','û', 'ûï','û_','Û','ÛÏ',

                 'Û–','Û_','_'])

stopwords.update(extraPatsToRemove)

#if ';' in stopwords:

print(stopwords)

train=process_dataframe_summarize_analysis(train, 'text', 'target', stopwords=stopwords, extraPatsToRemove=extraPatsToRemove)
drawWordCloud(train[train['target']==1],'clean_text')
drawWordCloud(train[train['target']==0],'clean_text')
process_dataframe_summarize_analysis(test, 'text', None, stopwords=stopwords, extraPatsToRemove=extraPatsToRemove)
results_train = set()

train['clean_text'].str.lower().str.split().apply(results_train.update)
results_test = set()

test['clean_text'].str.lower().str.split().apply(results_test.update)
diff_words=list(results_test&results_train)

textstr=' '.join(x for x in diff_words)

wordcloud = WordCloud(background_color="black").generate(textstr)

plt.figure(figsize=(20,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
diff_words=list(results_test-results_train)

textstr=' '.join(x for x in diff_words)

wordcloud = WordCloud(background_color="black").generate(textstr)

plt.figure(figsize=(20,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
diff_words=list(results_train-results_test)

textstr=' '.join(x for x in diff_words)

wordcloud = WordCloud(background_color="black").generate(textstr)

plt.figure(figsize=(20,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
import pandas as pd

import numpy as np

import gc



INPUT_DIR="/kaggle/input/nlp-getting-started/"

train=pd.read_csv(INPUT_DIR+"train.csv")

test=pd.read_csv(INPUT_DIR+"test.csv")



target_col=train['target']

train.drop(columns=['target'],inplace=True)

comb=pd.concat([train, test])

comb.reset_index(drop=True, inplace=True)

del train

del test

gc.collect()
print(comb.shape)
stopwords=set(STOPWORDS)

extraPatsToRemove=['??', '-', '_', '...']

stopwords.update(['THE', 'the', 'I', '-', 'The', ';', 'A', 'I\'m', '_','û', 'ûï','û_','Û','ÛÏ',

                 'Û–','Û_','_', '|', ':','.','To','This'])

stopwords.update(extraPatsToRemove)

#if ';' in stopwords:

print(stopwords)

process_dataframe_summarize_analysis(comb, 'text', None, stopwords=stopwords, extraPatsToRemove=extraPatsToRemove, N=200)
comb.head()
comb['entities']=np.empty((len(comb), 0)).tolist()

comb['entity_label']=np.empty((len(comb), 0)).tolist()

comb['nouns']=np.empty((len(comb), 0)).tolist()

comb['adv']=np.empty((len(comb), 0)).tolist()

comb['adj']=np.empty((len(comb), 0)).tolist()

comb['prop']=np.empty((len(comb), 0)).tolist()

comb['punct']=np.empty((len(comb), 0)).tolist()

comb['verb']=np.empty((len(comb), 0)).tolist()

comb['conj']=np.empty((len(comb), 0)).tolist()

comb['symbol']=np.empty((len(comb), 0)).tolist()

comb['sentiment']=np.empty((len(comb), 0)).tolist()

comb['tokens']=np.empty((len(comb), 0)).tolist()

comb['text_lemma']=np.empty((len(comb), 0)).tolist()
comb['clean_text']
import spacy

from spacy.lang.en import English



#create nlp object

nlp = spacy.load('en_core_web_lg')



docs=[]

#comb['doc'] = [nlp(text) for text in comb.clean_text]



for idx, row in comb.iterrows():

    doc=nlp(row['clean_text'])

    docs.append(doc)

comb['doc'] =docs

#comb.to_csv('comb_nlp.csv')
#fill the data now in columns

for idx, row in comb.iterrows():

    doc=row['doc']

    #print(type(doc))

    tokens = []

    nouns  = []

    advs   = []

    adjs   = []

    propns = []

    prop   = []

    verbs  = []

    puncts = []

    conjs  = []

    syms   = []

    lemmas = []

    for token in doc:

        tokens.append(token.text)

        if(token.pos_ == "NOUN"):

            nouns.append(token.text )

        elif(token.pos_ == "VERB"):

            verbs.append(token.text )

        elif(token.pos_ == "PROPN"):

            propns.append(token.text )

        elif(token.pos_ == "SYM"):

            syms.append(token.text )

        ####

        lemmas.append(token.lemma_ )

    

    comb.at[idx, 'tokens']=tokens

    comb.at[idx,'nouns']=nouns

    comb.at[idx,'verb']=verbs

    comb.at[idx,'prop']=propns

    comb.at[idx,'symbol']=syms

    comb.at[idx,'text_lemma']=lemmas

    
import pickle

storecomb='./storecomb_nlp.pkl'

combfile=open(storecomb, 'wb')

pickle.dump(comb, combfile)

combfile.close()
import pandas as pd

import spacy

from spacy.lang.en import English

#create nlp object

nlp = spacy.load('en_core_web_lg')
import pickle

comb=pickle.load(open('/kaggle/input/comprehensive-eda-and-patterns/storecomb_nlp.pkl', 'rb'))

#pd.read_csv('/kaggle/input/comprehensive-eda-and-patterns/comb_nlp.csv')
comb.head(5)
comb['keyword'].fillna('oth_keyword', inplace=True)

comb['location'].fillna('oth_loc', inplace=True)
comb['clean_text']
comb['text_lemma']
comb['lemma_'] = [' '.join(map(str, l)) for l in comb['text_lemma']]
comb['lemma_']
from sklearn.feature_extraction.text import CountVectorizer

def create_dataframe_matrix(df_col):

    vectorizer=CountVectorizer()

    matrix=vectorizer.fit_transform(df_col)

    features=vectorizer.get_feature_names()

    df=pd.DataFrame(matrix.toarray(), columns=features)

    return df
comb_df=create_dataframe_matrix(comb['lemma_'])
comb_df
#split train and test data from comb_df

x_train=comb_df[0:len(train)]
x_test=comb_df[len(train):]
x_train.shape
x_test.shape
y_train=train['target'].values
set(y_train)
x_train.info()
#Variable Importance

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators = 100,

                           n_jobs = -1,

                           oob_score = True,

                           bootstrap = True,

                           random_state = 99)

rf.fit(x_train, y_train)
rf.feature_importances_
#0.8077 score

predictions=rf.predict(x_test)
test_df=pd.DataFrame()

test_df['id']=test['id']

test_df['target']=predictions
test_df.to_csv('submission_rf_base.csv', index=False)
import gc

del comb

del comb_df

del test_df

gc.collect()
test_df
from xgboost import XGBClassifier

best_params = {'learning_rate': 0.16, 'n_estimators': 500, 

               'max_depth': 6, 'min_child_weight': 7,

               'subsample': 0.9, 'colsample_bytree': 0.7, 'nthread': -1, 

               'scale_pos_weight': 1, 'random_state': 42, 

               

               #next parameters are used to enable gpu for fasting fitting

               'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor', 'gpu_id': 0}

model = XGBClassifier(**best_params)
model.fit(x_train, y_train, eval_metric="error",verbose=True, early_stopping_rounds = 10)