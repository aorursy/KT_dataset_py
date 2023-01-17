# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import re
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
import nltk

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# reading train and test files
train = pd.read_csv('../input/nlp-getting-started/train.csv', dtype={'id': np.int16, 'target': np.int8})
test = pd.read_csv('../input/nlp-getting-started/test.csv', dtype={'id': np.int16})
print ('Done')
# basic info about data like shape ,type of variable, null values in different columns
print('shape of train dataset is {} columns and {} rows'.format(train.shape[0],train.shape[1]))
print('shape of test dataset is {} columns and {} rows'.format(test.shape[0],test.shape[1]))
print ('#####################################################################################')
print('columns name in train dataset')
print(train.columns.tolist())
print ('#####################################################################################')
print('target column is {}'.format(set(train.columns).difference(set(test.columns))))
print ('#####################################################################################')
print('variable type of different columns in train dataset')
print(train.dtypes)
print ('#####################################################################################')
print('total null values in train dataset')
print(train.isnull().sum().sum())
print ('#####################################################################################')
print('null values across different columns of train dataset')
print(train.isnull().sum())
print ('#####################################################################################')
print('number of unique values across different columns of train dataset')
print(train.nunique())
# Analysis of columns with null values
print('% of missing values in columns with null values:')
for col in train.columns:
    pct = []
    if train[col].isnull().sum() > 0:
        pct = ((train[col].isnull().sum())*100/len(train)).round(2)
        #print (pct)
        print ('percentage of missing value in column "{}" is {}%'.format(col,pct))

print("##############################################################################")
print('count values of different items in column with null values')      
print("##############################################################################")
print('column:keyword')
print(train['keyword'].value_counts()[:10])
print("##############################################################################")
print('column:location')
print(train['location'].value_counts()[:10])

#'looking top 5 rows of train dataset to get more insights'

print('looking top 5 rows of train dataset to get more insights')
train.head()
##### basic visualisation across different columns of train dataset 
# first we will try to check whether train dataset is balanced or unbalanced dataset
print('Distribution of target column')

tgt_count = train['target'].value_counts()
print(tgt_count)
sns.barplot(tgt_count.index, tgt_count)


# relationship between target and keywords
# credit for this plot : https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert
train['target_mean'] = train.groupby('keyword')['target'].transform('mean')

fig = plt.figure(figsize=(8, 72), dpi=100)

sns.countplot(y=train.sort_values(by='target_mean', ascending=False)['keyword'],
              hue=train.sort_values(by='target_mean', ascending=False)['target'])

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=12)
plt.legend(loc=1)
plt.title('Target Distribution in Keywords')

plt.show()

train.drop(columns=['target_mean'], inplace=True)

# cleaning text column of train dataset
import re

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def cleanhtml (sentence):
    cleantext = re.sub(r'http\S+',r'',sentence)
    return cleantext

def cleanpunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|)|\|/]',r' ',cleaned)      
    return cleaned

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

str1=' '
final_string=[]
s=''

for sent in train['text']:
    filter_sent = []
    sent = decontracted(sent)
    sent1 = remove_emoji(sent)
    rem_html = cleanhtml(sent1)
    rem_punc = cleanpunc (rem_html)
    for w in rem_punc.split():
        if ((w.isalpha())):
            if (w.lower() not in stopwords):
#               s=(ps.stem(w.lower())).encode('utf8')
                s=(w.lower()).encode('utf8')
                filter_sent.append(s)
            else:
                continue
        else:
            continue
    str1 = b" ".join(filter_sent)
    final_string.append(str1)
    
# attaching column new_col (cleaned text ) to dataframe
train['clean_text'] = np.array(final_string)
train['clean_text'].head(10)

# cleaning text column of train dataset
import re

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def cleanhtml (sentence):
    cleantext = re.sub(r'http\S+',r'',sentence)
    return cleantext

def cleanpunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|)|\|/]',r' ',cleaned)      
    return cleaned

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

str1=' '
final_string=[]
s=''

for sent in test['text']:
    filter_sent = []
    sent = decontracted(sent)
    sent1 = remove_emoji(sent)
    rem_html = cleanhtml(sent1)
    rem_punc = cleanpunc (rem_html)
    for w in rem_punc.split():
        if ((w.isalpha())):
            if (w.lower() not in stopwords):
#               s=(ps.stem(w.lower())).encode('utf8')
                s=(w.lower()).encode('utf8')
                filter_sent.append(s)
            else:
                continue
        else:
            continue
    str1 = b" ".join(filter_sent)
    final_string.append(str1)
    
# attaching column new_col (cleaned text ) to dataframe
test['clean_text'] = np.array(final_string)
test['clean_text'].head(10)

#### EDA ########
def length(text):    
    '''a function which returns the length of text'''
    return len(text)

train['tweet_len'] = train['text'].apply(length)
test['tweet_len'] = test['text'].apply(length)

plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
ax = sns.violinplot(x="target", y="tweet_len", data=train)
plt.subplot(1,2,2)
sns.distplot(train[train['target']==0]['tweet_len'][0:],label='Not Real')
sns.distplot(train[train['target']==1]['tweet_len'][0:],label='Real')

#### EDA ########

train['word_count'] = train.clean_text.apply(lambda x : len(x.split()))
test['word_count'] = test.clean_text.apply(lambda x : len(x.split()))

#train[['word_count','clean_text']].head()
plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
ax2 = sns.violinplot(x="target", y='word_count', data=train)
plt.subplot(1,2,2)
sns.distplot(train[train['target']==0]['word_count'][0:],label='Not Real')
sns.distplot(train[train['target']==1]['word_count'][0:],label='Real')


##### mean word length #########################################
train['avg_char_count'] = train.clean_text.apply(lambda x : np.mean([len(w) for w in x.split()]))
train['avg_char_count'].fillna(0,inplace=True)

test['avg_char_count'] = test.clean_text.apply(lambda x : np.mean([len(w) for w in x.split()]))
test['avg_char_count'].fillna(0,inplace=True)
#train['avg_char_count'].isnull().sum()
#train['avg_char_count'].head(10)
plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
ax2 = sns.violinplot(x="target", y='avg_char_count', data=train)
plt.subplot(1,2,2)
sns.distplot(train[train['target']==0]['avg_char_count'][0:],label='Not Real')
sns.distplot(train[train['target']==1]['avg_char_count'][0:],label='Real')

########### eda ##############
## no. of stopwords
train['num_stopwords'] = train.text.apply(lambda x :len([w for w in x.lower().split() if w in stopwords]))
test['num_stopwords'] = test.text.apply(lambda x :len([w for w in x.lower().split() if w in stopwords]))

plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
ax2 = sns.violinplot(x="target", y='num_stopwords', data=train)
plt.subplot(1,2,2)
sns.distplot(train[train['target']==0]['num_stopwords'][0:],label='Not Real')
sns.distplot(train[train['target']==1]['num_stopwords'][0:],label='Real')
##### url_count in the text
train['url_count'] = train.text.apply(lambda x : len([w for w in str(x).lower().split() if w in 'http' or w in 'https']))
test['url_count'] = test.text.apply(lambda x : len([w for w in str(x).lower().split() if w in 'http' or w in 'https']))

#dtrain['url_count'].head(10)
#dtrain['url_count'].value_counts()

x = train[['target','url_count']].groupby('url_count')['target'].value_counts()
x
######## hastag count #############################
train['hash_count'] = train.text.apply(lambda x : len([c for c in x if c in ('#')]))
test['hash_count'] = test.text.apply(lambda x : len([c for c in x if c in ('#')]))

train['hash_count'].head()
max(train['hash_count'])

x = train[['target','hash_count']].groupby('hash_count')['target'].value_counts()
x
#### retweet_count ############
#import re

retweet_dict = {}
for i in range(0,len(train)):
    txt = train['text'][i]
    txt1 = cleanhtml(txt)
    txt2 = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",txt1).split())
    retweet_dict[txt] = txt2.lower()

f_retweet_dict = {} 
for key1 in retweet_dict:
    if key1 not in f_retweet_dict:
        val = retweet_dict[key1]
        count = 0
        for key2 in retweet_dict:
            if val == retweet_dict[key2]:
                 count = count+1
        f_retweet_dict[key1]= count
print("###################### print first 10 element of dictionary####################")
print({k: f_retweet_dict[k] for k in list(f_retweet_dict)[:10] })
train['repeat_tweet_count'] = train['text'].map(f_retweet_dict)
#train[['same_tweet_count','text']].head(20)
#train['same_tweet_count'].isnull().sum()

y = train[['target','repeat_tweet_count']].groupby('repeat_tweet_count')['target'].value_counts()
y
# credit goes to : https://www.kaggle.com/saipkb86/disaster-tweets-logistic-naive
# Referenec : https://www.analyticsvidhya.com/blog/2018/07/hands-on-sentiment-analysis-dataset-python/

def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags

HT_regular = hashtag_extract(train['text'][train['target'] == 0])

# extracting hashtags from racist/sexist tweets
HT_disaster = hashtag_extract(train['text'][train['target'] == 1])

# unnesting list
HT_regular = sum(HT_regular,[])
HT_disaster = sum(HT_disaster,[])

fig,axes = plt.subplots(2,1,figsize=(18,10))

a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
sns.barplot(data=d, x= "Hashtag", y = "Count",ax=axes[0]).set_title('Normal Tweets')


a = nltk.FreqDist(HT_disaster)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
sns.barplot(data=d, x= "Hashtag", y = "Count",ax=axes[1]).set_title('Disaster Tweets')
#### retweet_count ############ for test dataset
retweet_dict_test = {}
for i in range(0,len(test)):
    txt = test['text'][i]
    txt1 = cleanhtml(txt)
    txt2 = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",txt1).split())
    retweet_dict_test[txt] = txt2.lower()

f_retweet_dict_test = {} 
for key1 in retweet_dict_test:
    if key1 not in f_retweet_dict_test:
        val = retweet_dict_test[key1]
        count = 0
        for key2 in retweet_dict_test:
            if val == retweet_dict_test[key2]:
                 count = count+1
        f_retweet_dict_test[key1]= count
print("###################### print first 10 element of dictionary####################")
print({k: f_retweet_dict_test[k] for k in list(f_retweet_dict_test)[:10] })
test['repeat_tweet_count'] = test['text'].map(f_retweet_dict_test)
### adding a column 'imputed_kw' to give information which all train['keyword'] indexes have null values
ids_train = train[train['keyword'].isnull()].index.tolist()
train['imputed_kw'] = 0
for i in ids_train:
    train['imputed_kw'].iloc[i]=1
    
ids_test = test[test['keyword'].isnull()].index.tolist()
test['imputed_kw'] = 0
for i in ids_test:
    test['imputed_kw'].iloc[i]=1
#train[['imputed_kw','keyword']].head(20)
# imputation of missing values in 'keyword' column
# kw_dict contain key = 'keyword' and value = cleaned and stemmed keyword.. for eg. key='airplane%20accident' , value = airplan accid
kw_dict = {}

for i in range(0,len(train)):
    kw = train['keyword'].iloc[i]
    if kw is not np.nan:
        kw_nonum = re.sub(r'[0-9]+',' ',kw) # removing any digits 0-9
        kw_nopunc = re.sub(r'[%]','',kw_nonum)# removing punctuation '%'
        if kw not in kw_dict:       # if keyword is present in kw_dict, then stop
            if len(kw_nopunc.split()) > 1:  # if cleaned 'keyword' contains more than 1 word
                split_kw = kw_nopunc.split()# spli the leyword and put it in 'split_kw'
                sdf = []
                s=''
                for w in split_kw:          # w is word in split_kw
                    if w not in stopwords:
                        w1 = ps.stem(w)     # stemming of word
                        sdf.append(w1.lower())   # appending stemmed word in sdf
                s =' '.join(sdf)         # after stemming , joining all words in sdf
                kw_dict[kw] = s          # putting s in kw_dict with original keyword as key
            else :
                w2 = ps.stem(kw)
                kw_dict[kw] = w2.lower()# if cleaned 'keyword' contains has only 1 word

first5pairs = {k: kw_dict[k] for k in list(kw_dict)[:5]}
print("############ First five key-value pairs of kw_dict ##################")
print(first5pairs)
c = train[['keyword','target']].groupby(['keyword'])['target'].value_counts()
cpct = c / c.groupby(level=0).sum()


pct_dict = {}
for kw in kw_dict:
    if (kw,1) in cpct.index:
        pct_dict[kw] = cpct[(kw,1)]
    else:
        pct_dict[kw] = 0
        
pct_dict5 = {k:pct_dict[k] for k in list(pct_dict)[:5]}
print(pct_dict5)
# sorting keyword in dictionary by respective value and then reversing the order, 
#so list having element lined up according the probailitues of keyword signifying target='1'

a = list(reversed(sorted(pct_dict,key=pct_dict.get)))
my_list = [str(l) for l in a]
#my_list

#kw_dict['wild%20fires']
# ids contain index value for which train['keyword'] is missing
# ids_sorted is not needed , but it is list ids sorted by index values
# final_kw_ids  contain key = index and value = 'Keyword' which is searched by using train['text'] data by matching with different keywords
ids = train[train['keyword'].isnull()].index.tolist()
ids_sorted = sorted(ids)
final_kw_ids = {}
for x in my_list:
    #kw1 = kw
    pr_kw = kw_dict[x]
    for i in ids_sorted:
            txt = train['clean_text'][i].decode('utf_8')
            if len(pr_kw.split()) > 1:
                pr_kw_list = pr_kw.split()
                if all(w in txt for w in pr_kw_list):
                    final_kw_ids[i] = x

for key in final_kw_ids:
    ids_sorted.remove(key)
    
for x in my_list:
        pr_kw = kw_dict[x]
        for i in ids_sorted:
            txt = train['clean_text'][i].decode('utf_8')
            if len(pr_kw.split()) == 1:
                if pr_kw in txt :
                    final_kw_ids[i] = x


                
final_kw_ids5 = {k:final_kw_ids[k] for k in list(final_kw_ids)[:5]}
print(final_kw_ids5)                

# ids_test contain index value for which train['keyword'] is missing
# ids_test_sorted is not needed , but it is list ids sorted by index values
# final_kw_ids_test  contain key = index and value = 'Keyword' which is searched by using test['text'] data by matching with different keywords
ids_test = test[test['keyword'].isnull()].index.tolist()
ids_test_sorted = sorted(ids_test)
final_kw_ids_test = {}
for x in my_list:
    #kw1 = kw
    pr_kw = kw_dict[x]
    for i in ids_test_sorted:
            txt = test['clean_text'][i].decode('utf_8')
            if len(pr_kw.split()) > 1:
                pr_kw_list = pr_kw.split()
                if all(w in txt for w in pr_kw_list):
                    final_kw_ids_test[i] = x

for key in final_kw_ids_test:
    ids_test_sorted.remove(key)
    
for x in my_list:
        pr_kw = kw_dict[x]
        for i in ids_test_sorted:
            txt = test['clean_text'][i].decode('utf_8')
            if len(pr_kw.split()) == 1:
                if pr_kw in txt :
                    final_kw_ids_test[i] = x


                
final_kw_ids_test5 = {k:final_kw_ids_test[k] for k in list(final_kw_ids_test)[:5]}
print(final_kw_ids_test5)
            
    
##### imputing value from final_kw_ids into train data :

#for key in final_kw_ids:
#   kw = final_kw_ids[key]
#   train['keyword'].iloc[key] = kw


train['keyword'].update(pd.Series(final_kw_ids))
test['keyword'].update(pd.Series(final_kw_ids_test))

#train.isnull().sum()
test.isnull().sum()

##### checking the train['keyword'], for which we are not able to find 'keywords'
null_ids = train[train['keyword'].isnull()].index.tolist()
train['text'][null_ids]
##### checking the train['keyword'], for which we are not able to find 'keywords'
null_ids = test[test['keyword'].isnull()].index.tolist()
test['text'][null_ids]
# tweets can be labeled 'nodisaster'
train['keyword'].fillna('nodisaster',inplace=True)
train['keyword'].value_counts()
# checking top 20 train['text'] and train['keyword'] combination
#train[['text','keyword']].head(20)
# tweets can be labeled 'nodisaster'
test['keyword'].fillna('nodisaster',inplace=True)
test['keyword'].value_counts()
pct_dict['nodisaster'] = 0

train['kw_dist_pct'] = train['keyword'].map(pct_dict)
test['kw_dist_pct'] = test['keyword'].map(pct_dict)
#test['kw_dist_pct'].isnull().sum()
#train['kw_dist_pct'].isnull().sum()
# since there are lot of null values in train['location'].. it may take lot of effort to impute. 
# it would be rather easier to merge location with text 
# first we should save info about which all indexes of column 'location' is null
### adding a column 'imputed_kw' to give information which all train['keyword'] indexes have null values
ids = train[train['location'].isnull()].index.tolist()
train['imputed_loc'] = 1
for i in ids:
    train['imputed_loc'].iloc[i]=0

#train[['imputed_loc','location']].head(20)

# since there are lot of null values in train['location'].. it may take lot of effort to impute. 
# it would be rather easier to merge location with text 
# first we should save info about which all indexes of column 'location' is null
### adding a column 'imputed_kw' to give information which all train['keyword'] indexes have null values
ids = test[test['location'].isnull()].index.tolist()
test['imputed_loc'] = 1
for i in ids:
    test['imputed_loc'].iloc[i]=0
# joining text and location column into column "full_text"

train["full_text"] = train["text"].fillna('').map(str) + " " + train["location"].fillna('').map(str) #+train["keyword"]
test["full_text"] = test["text"].fillna('').map(str) + " " + test["location"].fillna('').map(str) #+ test["keyword"]
# cleaning the column full text

import re

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def cleanhtml (sentence):
    cleantext = re.sub(r'http\S+',r'',sentence)
    return cleantext

def cleanpunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|)|\|/]',r' ',cleaned)      
    return cleaned

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

str1=' '
final_string=[]
s=''

for sent in train['full_text']:
    filter_sent = []
    sent = decontracted(sent)
    sent1 = remove_emoji(sent)
    rem_html = cleanhtml(sent1)
    rem_punc = cleanpunc (rem_html)
    for w in rem_punc.split():
        if ((w.isalpha()) & (len(w)>2)):
            if (w.lower() not in stopwords):
                s=(ps.stem(w.lower())).encode('utf8')
                #s=(w.lower()).encode('utf8')
                filter_sent.append(s)
            else:
                continue
        else:
            continue
    str1 = b" ".join(filter_sent)
    final_string.append(str1)
    
# attaching column new_col (cleaned text ) to dataframe
train['clean_fulltext'] = np.array(final_string)
train['clean_fulltext'].head(10)

# cleaning the column full text

import re

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def cleanhtml (sentence):
    cleantext = re.sub(r'http\S+',r'',sentence)
    return cleantext

def cleanpunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|)|\|/]',r' ',cleaned)      
    return cleaned

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

str1=' '
final_string=[]
s=''

for sent in test['full_text']:
    filter_sent = []
    sent = decontracted(sent)
    sent1 = remove_emoji(sent)
    rem_html = cleanhtml(sent1)
    rem_punc = cleanpunc (rem_html)
    for w in rem_punc.split():
        if ((w.isalpha()) & (len(w)>2)):
            if (w.lower() not in stopwords):
                s=(ps.stem(w.lower())).encode('utf8')
                #s=(w.lower()).encode('utf8')
                filter_sent.append(s)
            else:
                continue
        else:
            continue
    str1 = b" ".join(filter_sent)
    final_string.append(str1)
    
# attaching column new_col (cleaned text ) to dataframe
test['clean_fulltext'] = np.array(final_string)
test['clean_fulltext'].head(10)

############## wordcloud #####################
# we need to create corpus of word from train['text']. we will create seperate corpus of word for text with target=1 and target=0
ids_1 = train[train['target']==1].index.tolist()
all_words1=[]

for i in ids_1:
    txt = train['clean_fulltext'][i].decode('utf_8')
    for w in txt.split():
        all_words1.append(w)
text_1 = ' '.join(all_words1)    
    
    
wordcloud1 = WordCloud(width=800, height=400).generate(text_1)
#wordcloud2.generate_from_frequencies
plt.figure( figsize=(20,10) )

plt.imshow(wordcloud1)
plt.axis("off")
plt.show()


############## wordcloud #####################
# we need to create corpus of word from train['text']. we will create seperate corpus of word for text with target=1 and target=0
ids_0 = train[train['target']==0].index.tolist()
all_words0=[]

for i in ids_0:
    txt = train['clean_fulltext'][i].decode('utf_8')
    for w in txt.split():
        all_words0.append(w)
text_0 = ' '.join(all_words0)    
    
    
wordcloud0 = WordCloud(width=800, height=400).generate(text_0)
#wordcloud2.generate_from_frequencies
plt.figure( figsize=(20,10) )
plt.imshow(wordcloud0)
plt.axis("off")
plt.show()
print(train.columns)
print(test.columns)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from sklearn.preprocessing import normalize
import scipy as sp

# Create vectorizer for function to use
vectorizer = TfidfVectorizer(ngram_range=(1,3),min_df=2,max_df=0.5,max_features=1000)
y = train["target"].values 

X = sp.sparse.hstack((vectorizer.fit_transform(train.clean_fulltext),sc.fit_transform(train[['tweet_len','num_stopwords','hash_count','repeat_tweet_count', 'imputed_kw', 'kw_dist_pct',
       'imputed_loc']].values)),format='csr')

X_columns=vectorizer.get_feature_names()+train[['tweet_len','num_stopwords','hash_count','repeat_tweet_count', 'imputed_kw', 'kw_dist_pct',
       'imputed_loc']].columns.tolist()
print(X.shape)
test_sp = sp.sparse.hstack((vectorizer.transform(test.clean_fulltext),sc.transform(test[['tweet_len','num_stopwords','hash_count','repeat_tweet_count', 'imputed_kw', 'kw_dist_pct',
       'imputed_loc']].values)),format='csr')
test_columns=vectorizer.get_feature_names()+test[['tweet_len','num_stopwords','hash_count','repeat_tweet_count', 'imputed_kw', 'kw_dist_pct',
       'imputed_loc']].columns.tolist()
print(test_sp.shape)
import sklearn
from sklearn import linear_model
from sklearn.model_selection import learning_curve, GridSearchCV

# Create logistic regression object
logistic = linear_model.LogisticRegression(max_iter=500)
# Create a list of all of the different penalty values that you want to test and save them to a variable called 'penalty'
penalty = ['l2']
# Create a list of all of the different C values that you want to test and save them to a variable called 'C'
C = [0.0001, 0.001, 0.01, 1,10, 100]
# Now that you have two lists each holding the different values that you want test, use the dict() function to combine them into a dictionary. 
# Save your new dictionary to the variable 'hyperparameters'
hyperparameters = dict(C=C, penalty=penalty)
# Fit your model using gridsearch
clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=1,scoring='f1')
best_model = clf.fit(X, y)
#Print all the Parameters that gave the best results:
print('Best Parameters',clf.best_params_)
# You can also print the best penalty and C value individually from best_model.best_estimator_.get_params()
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

kfold = model_selection.KFold(n_splits=5)
model = LogisticRegression(penalty='l2',dual=False,max_iter=1000,C=1)
model.fit(X,y)
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print("Accuracy: Final mean:%.3f%%, Final standard deviation:(%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
print('Accuracies from each of the 5 folds using kfold:',results)
print("Variance of kfold accuracies:",results.var())
sub = model.predict(test_sp)
test['target'] = sub

Final_submission= test[['id','target']]
Final_submission.to_csv('submission1.csv',index=False)
feature_importance = abs(model.coef_[0])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
featfig = plt.figure(figsize=(8,64), dpi=100)
featax = featfig.add_subplot(1, 1, 1)
featax.barh(pos, feature_importance[sorted_idx], align='center')
featax.set_yticks(pos)
featax.set_yticklabels(np.array(X_columns)[sorted_idx], fontsize=8)
featax.set_xlabel('Relative Feature Importance')

plt.tight_layout()   
plt.show()
# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.71, kernel='linear', degree=3, gamma='auto')
kfold = model_selection.KFold(n_splits=5)
SVM.fit(X,y)
results = model_selection.cross_val_score(SVM, X, y, cv=kfold)
print("Accuracy: Final mean:%.3f%%, Final standard deviation:(%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
print('Accuracies from each of the 5 folds using kfold:',results)
print("Variance of kfold accuracies:",results.var())
# predict the labels on validation dataset
predictions_SVM = SVM.predict(test_sp)
test['target'] = predictions_SVM
Final_submission2= test[['id','target']]
Final_submission2.to_csv('submission2.csv',index=False)
# Use accuracy_score function to get the accuracy
#print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)