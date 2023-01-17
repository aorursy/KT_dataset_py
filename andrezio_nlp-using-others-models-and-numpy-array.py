import numpy as np 

import pandas as pd 
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sample_submission=pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

train_df.head(2)
import string

import re



from nltk.corpus import stopwords

stop = stopwords.words('english')

from sklearn.preprocessing import LabelEncoder



def remove_id(df):

    df.drop(['id'], axis=1, inplace=True)

    columns = list(df.columns)

    df = df.filter(items=columns)

    df.fillna('0', inplace=True)

    return df



def remove_stop_words(df):

    df['text2']=df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    return df



def find_hashtags(tweet):

    '''This function will extract hashtags'''

    return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet) 



def novas_colunas(trainDF):

    trainDF['char_count'] = trainDF['text'].apply(len)

    trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))

    trainDF['punctuation_count'] = trainDF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 

    trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))

    trainDF['upper_case_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

    trainDF['hashtag'] = trainDF['text'].apply(find_hashtags) 

    trainDF['counthash'] = trainDF['hashtag'].apply(len)

    return trainDF
train_df = remove_id(train_df)

test_df = remove_id(test_df)



train_df = novas_colunas(train_df)

test_df = novas_colunas(test_df)



train_df = remove_stop_words(train_df)

test_df = remove_stop_words(test_df)





train_df['keyword_id'] = LabelEncoder().fit_transform(train_df.keyword)

test_df['keyword_id'] = LabelEncoder().fit_transform(test_df.keyword)



train_df['location_id'] = LabelEncoder().fit_transform(train_df.location)

test_df['location_id'] = LabelEncoder().fit_transform(test_df.location)



df_y =train_df.target

train_df.drop(['target'], axis=1, inplace=True)
train_df.head(2)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))

X = vectorizer.fit_transform(train_df.text2).toarray()

X1 = vectorizer.transform(test_df.text2).toarray()





tfidfconverter = TfidfTransformer(use_idf=True)

X = tfidfconverter.fit_transform(X).toarray()

X1 = tfidfconverter.transform(X1).toarray()





n_df = train_df[[ 'char_count', 'word_count',

       'punctuation_count', 'title_word_count',

       'upper_case_word_count', 'counthash', 'keyword_id',

       'location_id']]



df_x = np.concatenate((n_df.values,X),axis=1)





test_n_df = test_df[[ 'char_count', 'word_count',

       'punctuation_count', 'title_word_count',

       'upper_case_word_count', 'counthash', 'keyword_id',

       'location_id']]



test_df_x = np.concatenate((test_n_df,X1),axis=1)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score





X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.3, random_state=0)



params={'n_estimators':1000, 'random_state':0}



classifier = RandomForestClassifier(**params)

classifier.fit(X_train, y_train) 





y_pred = classifier.predict(X_test)

print('score: {}'.format(accuracy_score(y_test, y_pred)))

# classifier.fit(df_x, df_y)



sample_submission["target"] = classifier.predict(test_df_x)

sample_submission.to_csv("submission.csv", index=False)