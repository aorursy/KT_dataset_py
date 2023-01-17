### Libraries to be imported

import pandas as pd 

import numpy as np

import nltk,re,string

import matplotlib.pyplot as plt



from nltk.corpus import stopwords

from sklearn.svm import SVC

from sklearn import model_selection

from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from nltk.tokenize import word_tokenize,sent_tokenize,RegexpTokenizer

from wordcloud import WordCloud

from sklearn.linear_model import LogisticRegression
data_frame = pd.read_csv("../input/nlp-getting-started/train.csv")
data_test = pd.read_csv("../input/nlp-getting-started/test.csv")
data_test.head(20)
data_frame.head(20)
data_test.isnull().sum()
data_frame.isnull().sum()
def show_word_distrib(target=1, field="text"):

    txt = data_frame[data_frame['target']==target][field].str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')

    words = nltk.tokenize.word_tokenize(txt)

    words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stop) 

    

    rslt = pd.DataFrame(words_except_stop_dist.most_common(top_N),

                        columns=['Word', 'Frequency']).set_index('Word')

    print(rslt)

    

# score_df = pd.DataFrame(columns={'Model Description','Score'})
data_frame['keyword'] = data_frame['keyword'].map(lambda s: s.replace('%20', ' ') if isinstance(s, str) else s)

un_KW  = {kw for kw in data_frame['keyword' ].values if isinstance(kw, str)}

tot_KW = len(data_frame) - len(data_frame[data_frame["keyword" ].isna()])

print(len(un_KW))

print("total",tot_KW)

print("Samples with no KW",len(data_frame[data_frame['keyword'].isna()]))

data_test['keyword'] = data_test['keyword'].map(lambda s: s.replace('%20', ' ') if isinstance(s, str) else s)

un_KW  = {kw for kw in data_test['keyword' ].values if isinstance(kw, str)}

tot_KW = len(data_test) - len(data_test[data_test["keyword" ].isna()])

print(len(un_KW))

print("total",tot_KW)

print("Samples with no KW",len(data_test[data_test['keyword'].isna()]))

un_LOC  = {kw for kw in data_frame['location' ].values if isinstance(kw, str)}

tot_LOC = len(data_frame) - len(data_frame[data_frame["location" ].isna()])

print(len(un_LOC))

print("total",tot_LOC)

print("Samples with no location",len(data_frame[data_frame['location'].isna()]))
#KW and location analysis

#remove space between keywords

data_frame['keyword'] = data_frame['keyword'].map(lambda s: s.replace('%20', ' ') if isinstance(s, str) else s)

total_keyword = {kw for kw in data_frame['keyword'].values if isinstance(kw,str)}

print(total_keyword)



#for location

locations = {loc for loc in data_frame['location'].values if isinstance(loc,str)}

# print(locations)
#KW and location analysis

#remove space between keywords

data_test['keyword'] = data_test['keyword'].map(lambda s: s.replace('%20', ' ') if isinstance(s, str) else s)

total_keyword = {kw for kw in data_test['keyword'].values if isinstance(kw,str)}

# print(total_keyword)



#for location

locations = {loc for loc in data_test['location'].values if isinstance(loc,str)}

print(locations)
#disaster keywords and regular keywords

#disaster 



disaster_keyword = [kw for kw in data_frame.loc[data_frame.target == 1].keyword]

disaster_keywords_counts = dict(pd.DataFrame(data={'x': disaster_keyword}).x.value_counts())

# print(disaster_keywords_counts)



#regualar

regular_kw = [kw for kw in data_frame.loc[data_frame.target == 0].keyword]

regular_keywords_counts = dict(pd.DataFrame(data={'x': regular_kw}).x.value_counts())

print(regular_keywords_counts)
#exploring regular tweets 

regular_tweets = data_frame[data_frame['target']==0]['text']

print(regular_tweets.values[11])



#exploring disaster tweets

disaster_tweets = data_frame[data_frame['target'] == 1]['text']

print(disaster_tweets.values[69])
#word count

data_frame['word_count'] = data_frame['text'].apply(lambda x: len(str(x).split()))

print(data_frame['word_count'])



#unique_word_count

data_frame['unique_word_count'] = data_frame['text'].apply(lambda x: len(set(str(x).split())))

print(data_frame['unique_word_count'])



#url count

data_frame['url_count'] = data_frame['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))

print(data_frame['url_count'])



#character count

data_frame['char_count'] = data_frame['text'].apply(lambda x: len(str(x)))

print(data_frame['char_count'])



#mention count

data_frame['mention_count'] = data_frame['text'].apply(lambda x: len([c for c in str(x) if c == '@']))

print(data_frame['mention_count'])



#hashtag count

data_frame['hashtag_count'] = data_frame['text'].apply(lambda x: len([c for c in str(x) if c == '#']))

# print(data_frame['text'])

# print(data_frame['hashtag_count'])

hashtag_exists = data_frame[(data_frame.hashtag_count != 0)]

print(hashtag_exists['text'])



#mean word length

data_frame['mean_word_length'] = data_frame['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

print(data_frame['mean_word_length'])
#word count

data_test['word_count'] = data_test['text'].apply(lambda x: len(str(x).split()))

print(data_test['word_count'])



#unique_word_count

data_test['unique_word_count'] = data_test['text'].apply(lambda x: len(set(str(x).split())))

print(data_test['unique_word_count'])



#url count

data_test['url_count'] = data_test['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))

print(data_test['url_count'])



#character count

data_test['char_count'] = data_test['text'].apply(lambda x: len(str(x)))

print(data_test['char_count'])



#mention count

data_test['mention_count'] = data_test['text'].apply(lambda x: len([c for c in str(x) if c == '@']))

print(data_test['mention_count'])



#hashtag count

data_test['hashtag_count'] = data_test['text'].apply(lambda x: len([c for c in str(x) if c == '#']))

# print(data_frame['text'])

# print(data_frame['hashtag_count'])

hashtag_exists = data_test[(data_test.hashtag_count != 0)]

print(hashtag_exists['text'])



#mean word length

data_test['mean_word_length'] = data_test['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

print(data_test['mean_word_length'])
#removing noises : removing urls,html punctuations, numerical values

def text_clean(text):

    #to lower case

    test =  text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    #remove urls

    text = re.sub('https?://\S+|www\.\S+', '', text)

    #remove html

    text = re.sub('<.*?>+', '', text)

    #remove punctuations

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    #remove noises

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text

data_frame['text'] = data_frame['text'].apply(lambda x: text_clean(x))

print(data_frame['text'])

print("test data")

data_test['text'] = data_test['text'].apply(lambda x: text_clean(x))

print(data_test['text'])
#tokenization

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

# for x in data_frame['text']:

#     print(x)

data_frame['text'] = data_frame['text'].apply(lambda x: tokenizer.tokenize(x))

print(data_frame['text'])



data_test['text'] = data_test['text'].apply(lambda x: tokenizer.tokenize(x))

print(data_test['text'])

#remove stopwords

stop = stopwords.words('english')

def remove_stopwords(text):

    words = [w for w in text if w not in stop]

    return(words)



data_frame['text'] = data_frame['text'].apply(lambda x: remove_stopwords(x))

print(data_frame['text'])



data_test['text'] = data_test['text'].apply(lambda x: remove_stopwords(x))

print(data_test['text'])
#return data to original format

def combine_text(list_of_text):

    combined_text = ' '.join(list_of_text)

    return combined_text

data_frame['text'] = data_frame['text'].apply(lambda x: combine_text(x))

data_frame.head()



data_test['text'] = data_test['text'].apply(lambda x: combine_text(x))

data_test.head()
count_vectorizer = CountVectorizer()

data_vectors = count_vectorizer.fit_transform(data_frame['text'])

data_test_vectors = count_vectorizer.fit_transform(data_test['text'])
print(data_vectors[0].todense())
print(data_test_vectors[0].todense())
tfidf = TfidfVectorizer(min_df=4, max_df=0.5, ngram_range=(1, 2))

train_tfidf = tfidf.fit_transform(data_frame['text'])
# Fitting a simple Logistic Regression on Counts

classifier = LogisticRegression(C=1.0)

scores = model_selection.cross_val_score(classifier, data_vectors, data_frame["target"], cv=5, scoring="f1")

scores
classifier.fit(data_vectors, data_frame["target"])
# Fitting a simple Logistic Regression on TFIDF

clf_tfidf = LogisticRegression(C=1.0)

scores = model_selection.cross_val_score(clf_tfidf, train_tfidf, data_frame["target"], cv=5, scoring="f1")

scores
# X_train, X_test, y_train, y_test = \

#         train_test_split(data_frame['text'], data_frame['target'], random_state=20)

# ## Apply Tfidf tranformation

# vector = TfidfVectorizer().fit(X_train)

# X_train_vector = vector.transform(X_train)

# X_test_vector  = vector.transform(X_test)

# df_test_vector = vector.transform(data_test['text'])



# gb_model= GaussianNB().fit(X_train_vector.todense(),y_train)

# predict = gb_model.predict(X_test_vector.todense())



# print('Roc AUC score - %3f'%(roc_auc_score(y_test,predict)))

# score_df = score_df.append({'Model Description':'Naive Bayes',

#                            'Score':roc_auc_score(y_test,predict)}

#                            ,ignore_index=True)
vector = TfidfVectorizer().fit(data_frame['text'])

df_train_vector = vector.transform(data_frame['text'])

df_test_vector = vector.transform(data_test['text'])



svc_model = SVC()

grid_values={'kernel':['linear', 'poly', 'rbf'],'C':[0.001,0.01,1,10]}

grid_search_model= GridSearchCV(svc_model,param_grid=grid_values,cv=3)

grid_search_model.fit(df_train_vector,data_frame['target'])



print(grid_search_model.best_estimator_)

print(grid_search_model.best_score_)

print(grid_search_model.best_params_)



# score_df = score_df.append({'Model Description':'SVC - with Grid Search',

#                            'Score':grid_search_model.best_score_}

#                            ,ignore_index=True)



predict = grid_search_model.predict(df_test_vector)

predict_df = pd.DataFrame()

predict_df['id'] = data_test['id']

predict_df['target'] = predict
predict_df.to_csv('sample_submission_1.csv', index=False)