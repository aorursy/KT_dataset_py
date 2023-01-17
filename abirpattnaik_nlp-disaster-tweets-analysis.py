# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import regex

import nltk

from nltk.tokenize import RegexpTokenizer

from nltk.stem import WordNetLemmatizer,PorterStemmer

from nltk.corpus import stopwords

import re

from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_data=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

train_data.head()
print('The train set contains {0} rows and {1} columns '.format(train_data.shape[0],train_data.shape[1]))
ax=sns.countplot(data=train_data,x=train_data['target'])

plt.xlabel('Target Variable- Disaster or not disaster tweet')

plt.ylabel('Count of tweets')

plt.title('Count of disaster and non-disaster tweets')

total = len(train_data)

for p in ax.patches:

        ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.1, p.get_height()+5))



#https://stackoverflow.com/questions/33179122/seaborn-countplot-with-frequencies

# Study the document for matplotlib
train_data['keyword']=train_data.keyword.str.replace('%20','_')

train_data['keyword'] = train_data['keyword'].replace(np.nan, '', regex=True)
#plt.figure(figsize=(100,50))



#sns.catplot(data=train_data,y='keyword',col='target',height=200,aspect=0.5,kind='count')
#pd.set_option('display.max_rows' ,None)

keyword_count=pd.DataFrame(train_data.groupby(['keyword','target']).agg(['count']).sort_values(by=('id', 'count'),ascending=False)[('id', 'count')])
keyword_count.columns


#keyword_count['keyword_value']=keyword_count.keyword[0][0]

'''

keyword_count['keyword']=keyword_count.index

keyword_count.columns=keyword_count.columns.droplevel()

keyword_count.columns=['count','keyword']



for value in range(0,len(keyword_count)):

    #print(keyword_count.keyword[value][0])

    if 'keyword_value' not in keyword_count.columns:

        keyword_count['keyword_value']=keyword_count.keyword[0][0]

    else:

        keyword_count['keyword_value'][value]=keyword_count.keyword[value][0]

    #print(keyword_count.keyword[value][1])

    if 'target_value' not in keyword_count.columns:

        keyword_count['target_value']=keyword_count.keyword[0][1]

    else:

        keyword_count['target_value'][value]=keyword_count.keyword[value][1]



if 'keyword' in keyword_count.columns:

    keyword_count=keyword_count.drop(['keyword'],axis=1)

#Index(['count', 'keyword', 'keyword_value', 'target_value'], dtype='object')

'''
keyword_count
wordcloud = WordCloud(

                          background_color='white',

                          max_words=100,

                          max_font_size=80, 

                          random_state=42,

    collocations=False,

    colormap="Oranges_r"

                         ).generate(' '.join(train_data[train_data['target']==1]['keyword']))

#.join(text2['Crime Type']))



plt.figure(figsize=(10,10))

plt.title('Major keywords for disaster tweets', fontsize=30)

plt.imshow(wordcloud)



plt.axis('off')

plt.show()
wordcloud = WordCloud(

                          background_color='white',

                          max_words=100,

                          max_font_size=40, 

                            collocations=False,

    colormap="PuOr"

                         ).generate(' '.join(train_data[train_data['target']==0]['keyword']))

#.join(text2['Crime Type']))



print(wordcloud)

plt.figure(figsize=(10,25))

plt.imshow(wordcloud)

plt.title('Major keywords for non-disaster tweets', fontsize=30)

plt.axis('off')

plt.show()
train_data['location'].value_counts()
#pd.set_option('display.max_rows' ,None)

train_data['location']
lemmatizer = WordNetLemmatizer()

stemmer = PorterStemmer() 

def preprocess(sentence):

    sentence=str(sentence)

    sentence = sentence.lower()

    sentence=sentence.replace('{html}',"") 

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, '', sentence)

    rem_url=re.sub(r'http\S+', '',cleantext)

    rem_num = re.sub('[0-9]+', '', rem_url)

    tokenizer = RegexpTokenizer(r'\w+')

    tokens = tokenizer.tokenize(rem_num)  

    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]

    stem_words=[stemmer.stem(w) for w in filtered_words]

    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]

    return " ".join(filtered_words)
train_data['location_cleaned']=train_data['location'].map(lambda s:preprocess(s))
#train_data['location_cleaned']
train_data["location_cleaned"].replace({"united states": "usa", 

                                        "world": "worldwide",

                                        "nyc":"new york",

                                       "california usa":"california",

                                        "new york city":"new york",

                                        "california united states":"california",

                                        "mumbai":"india"

                                       }, inplace=True)
train_data['location_cleaned'].value_counts().nlargest(20)
#Preprocessing text

train_data['text_cleaned']=train_data['text'].map(lambda s:preprocess(s)) 

test_data['text_cleaned']=test_data['text'].map(lambda s:preprocess(s))
#train_data

train_text = train_data['text_cleaned']

test_text = test_data['text_cleaned']

train_target = train_data['target']

all_text = train_text.append(test_text)
tfidf_vectorizer = TfidfVectorizer()

tfidf_vectorizer.fit(all_text)



count_vectorizer = CountVectorizer()

count_vectorizer.fit(all_text)



train_text_features_cv = count_vectorizer.transform(train_text)

test_text_features_cv = count_vectorizer.transform(test_text)



train_text_features_tf = tfidf_vectorizer.transform(train_text)

test_text_features_tf = tfidf_vectorizer.transform(test_text)
train_text.head()
kfold = KFold(n_splits = 5, shuffle = True, random_state = 2018)

test_preds = 0

oof_preds = np.zeros([train_data.shape[0],])



for i, (train_idx,valid_idx) in enumerate(kfold.split(train_data)):

    x_train, x_valid = train_text_features_tf[train_idx,:], train_text_features_tf[valid_idx,:]

    y_train, y_valid = train_target[train_idx], train_target[valid_idx]

    classifier = LogisticRegression()

    print('fitting.......')

    classifier.fit(x_train,y_train)

    print('predicting......')

    print('\n')

    oof_preds[valid_idx] = classifier.predict_proba(x_valid)[:,1]

    test_preds += 0.2*classifier.predict_proba(test_text_features_tf)[:,1]
pred_train = (oof_preds > .25).astype(np.int)

f1_score(train_target, pred_train)
#submission1 = pd.DataFrame.from_dict({'id': test['id']})

#submission1['prediction'] = (test_preds>0.25).astype(np.int)

#submission1.to_csv('submission.csv', index=False)

#submission1['prediction'] = (test_preds>0.25)
submission1=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

submission1['target'] = (test_preds>0.25).astype(np.int)

submission1.to_csv('submission.csv', index=False)

submission1['target'] = (test_preds>0.25)

submission1['target']=submission1['target'].map(lambda x:int(x==True))
submission1.head()