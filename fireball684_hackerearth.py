# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import nltk 

from nltk.tokenize import sent_tokenize, word_tokenize, WhitespaceTokenizer

from nltk.stem import PorterStemmer, WordNetLemmatizer

from nltk.probability import FreqDist

from nltk.corpus import stopwords, wordnet

from nltk import pos_tag

import string

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.tokenize import RegexpTokenizer

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from xgboost import XGBClassifier

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score



from wordcloud import WordCloud

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')
# nltk.corpus.wordnet.fileids()
# lemma = WordNetLemmatizer()

# pstem = PorterStemmer()

# lemma.lemmatize('goodness'), pstem.stem('goodness')
# for word in nltk.corpus.wordnet.words():

#     print(word)
df_train=pd.read_csv('../input/train.csv')

df_test=pd.read_csv('../input/test.csv')

df_subm=pd.read_csv('../input/sample_submission.csv')
df_train.info()
df_train.sample(5)
df_train.shape, df_test.shape
# for i in range(len(df_train.columns)):

#     if i in [0,2,3,6,7,8,9]:

#         pass

#     else:

#         print(df_train.iloc[:,i].value_counts())
df_train.dtypes
df_train.isnull().sum()
col = ['score_1', 'score_2', 'score_3', 'score_4', 'score_5', 'score_6']

for c in col:

    df_train[c].fillna(df_train[c].dropna().median(), inplace=True)

    df_test[c].fillna(df_train[c].dropna().median(), inplace=True)



df_train['advice_to_mgmt'].fillna('', inplace=True)

df_test['advice_to_mgmt'].fillna('', inplace=True)



df_train.dropna(subset=['negatives','summary'], inplace=True)

df_test.dropna(subset=['negatives','summary'], inplace=True)
df_train.shape, df_test.shape
df_train.isnull().sum()
drop_col = ['ID', 'location', 'date']

df_train.drop(columns=drop_col, inplace=True)

df_test.drop(columns=drop_col, inplace=True)
df_train.shape, df_test.shape
df_train.sample(5)
df_train['Place'].shape
OEncoder = OrdinalEncoder()

Encoded = OEncoder.fit_transform(df_train[['Place', 'status']])

Encoded_test = OEncoder.transform(df_test[['Place', 'status']])
Encoded
Encoded.shape, df_train.shape
def Create_ENC(df, Enc):

#   Create empty arrays with random elements with dimensions of the encoded column

    Place_enc = np.empty((len(Enc),))  

    Status_enc = np.empty((len(Enc),))

    for i in range(len(Enc)):

        Place_enc[i] = Enc[i][0]

        Status_enc[i] = Enc[i][1]

    df['place_enc'] = Place_enc

    df['status_enc'] = Status_enc
Create_ENC(df_train, Encoded)

Create_ENC(df_test, Encoded_test)
df_test.isnull().sum()
df_train.sample(5)
df_train.groupby('overall').Place.count()
df_train.groupby('Place').overall.count()
# df_train.groupby('job_title').overall.count()

# no information from this
def Review_len(df):

    df['len_pos'] = df['positives'].str.len()

    df['len_neg'] = df['negatives'].str.len()

    df['len_sum'] = df['summary'].str.len()
Review_len(df_train)

Review_len(df_test)
df_train.sample(10)
df_train.dtypes
def ChangeToInt(df,col):

    df[col]=df[col].astype('int')
label='overall'

ChangeToInt(df_train,label)
def show_wordcloud(data, title = None):

    V_wordcloud = WordCloud(

        background_color = 'white',

        max_words = 200,

        max_font_size = 40, 

        scale = 3,

        random_state = 7

    ).generate(str(data))



    fig = plt.figure(1, figsize = (20, 20))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize = 20)

        fig.subplots_adjust(top = 2.3)



    plt.imshow(V_wordcloud)

    plt.show()
# print positive wordcloud

show_wordcloud(df_train["positives"])
# print negatives wordcloud

show_wordcloud(df_train["negatives"])
# print summary wordcloud

show_wordcloud(df_train["summary"])
# def Reviews(df):

#     df['Reviews']=df['positives']+' '+df['negatives']+' '+df['summary']

# #     +' '+df['advice_to_mgmt']
# Reviews(df_train)

# Reviews(df_test)
# df_train.Reviews[0]
def get_wordnet_pos(pos_tag):

    if pos_tag.startswith('J'):

        return wordnet.ADJ

    elif pos_tag.startswith('V'):

        return wordnet.VERB

    elif pos_tag.startswith('N'):

        return wordnet.NOUN

    elif pos_tag.startswith('R'):

        return wordnet.ADV

    else:

        return wordnet.NOUN
def clean_text(text):

    # lower text

    text = text.lower()

    # tokenize text and remove puncutation

    text = [word.strip(string.punctuation) for word in text.split(" ")]

    # remove words that contain numbers

    text = [word for word in text if not any(c.isdigit() for c in word)]

    # remove stop words

    stop = stopwords.words('english')

    text = [x for x in text if x not in stop]

    # remove empty and less than 3 length tokens

    text = [t for t in text if len(t) >= 3]

    # pos tag text

    pos_tags = pos_tag(text)

    # lemmatize text

    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]

    # remove words with less than 3 letters

    text = [t for t in text if len(t) >= 3]

    # join all

    text = " ".join(text)

    return(text)
# # clean text data

# df_train["Clean_reviews"] = df_train["Reviews"].apply(lambda x: clean_text(x))

# df_test["Clean_reviews"] = df_test["Reviews"].apply(lambda x: clean_text(x))
# df_train.drop(columns='Reviews', inplace=True)

# df_test.drop(columns='Reviews', inplace=True)
df_train["Clean_positives"] = df_train["positives"].apply(lambda x: clean_text(x))

df_train["Clean_negatives"] = df_train["negatives"].apply(lambda x: clean_text(x))
df_test["Clean_positives"] = df_test["positives"].apply(lambda x: clean_text(x))

df_test["Clean_negatives"] = df_test["negatives"].apply(lambda x: clean_text(x))
df_train["Clean_summary"] = df_test["summary"].apply(lambda x: clean_text(x))

df_test["Clean_summary"] = df_test["summary"].apply(lambda x: clean_text(x))
df_train.shape, df_test.shape
df_train.isnull().sum()
df_test.sample(2)
df_train["Clean_reviews"] = df_train["Clean_positives"]+' '+df_train["Clean_negatives"]+' '+df_train["Clean_summary"]

df_test["Clean_reviews"] = df_test["Clean_positives"]+' '+df_test["Clean_negatives"]+' '+df_test["Clean_summary"]
df_train.head(2)
# scores = ['score_1', 'score_2', 'score_3', 'score_4', 'score_5', 'score_6']

# for col in scores:

#     print(df_train[col].value_counts())

# score_6 column doesn't have uniform values. Ignore this column in analysis
# df_train.isnull().sum()  

# 1115

# df_train[df_train['Clean_reviews'].isnull() == True]
# df_train.drop(columns=['num_words_pos', 'num_words_neg'], inplace=True)

# df_test.drop(columns=['num_words_pos', 'num_words_neg'], inplace=True)
df_train.shape, df_test.shape
df_train.isnull().sum()
df_train.dropna(inplace=True)

df_test.dropna(inplace=True)
df_train.shape, df_test.shape
len(df_train.loc[0,'Clean_positives'].split())
def num_words(df):

    df['num_words_pos'] = df['positives'].apply(lambda x: len(x.split()))

    df['num_words_neg'] = df['negatives'].apply(lambda x: len(x.split()))

    df['num_words_sum'] = df['summary'].apply(lambda x: len(x.split()))

#     df['num_words_neg'] = len(df['negatives'].str.split())
num_words(df_train)

num_words(df_test)
df_train.columns
# df_test.isnull().sum()
# token = RegexpTokenizer(r'[a-zA-Z0-9]+')

# CVec = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)

CVec = CountVectorizer(stop_words='english', min_df=3)

feat_col=['place_enc', 'status_enc', 'len_pos', 'len_neg', 'num_words_pos', 'num_words_neg', 'num_words_sum', 

          'len_sum', 'score_1', 'score_2', 'score_3', 'score_4', 'score_5', 'Clean_reviews']

X=df_train[feat_col]

Y=df_train['overall']

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=7)
X_train_vect=CVec.fit_transform(X_train['Clean_reviews'])

X_test_vect=CVec.transform(X_test['Clean_reviews'])
X_train_vect.shape, X_test_vect.shape
len(CVec.get_feature_names())
reviews_train = pd.DataFrame(X_train_vect.todense(), columns=CVec.get_feature_names())

reviews_test = pd.DataFrame(X_test_vect.todense(), columns=CVec.get_feature_names())
reviews_train.shape, reviews_test.shape
y_train.shape, y_test.shape
X_train.drop(columns='Clean_reviews', inplace=True)

X_test.drop(columns='Clean_reviews', inplace=True)
X_train.shape , X_test.shape
# X_train_withvect = pd.concat([X_train, reviews_train], axis=1)

# X_test_withvect = pd.concat([X_test, reviews_test], axis=1)



# creating a new df is causing RAM to be exhausted.
X_train.reset_index(drop=True, inplace=True)

# X_train.drop(columns='index', inplace=True)

X_test.reset_index(drop=True, inplace=True)

# X_test.drop(columns='index', inplace=True)
X_train.head()

reviews_train.head()
# X_train_ = pd.concat([X_train, reviews_train], axis=1)

# X_test_ = pd.concat([X_test, reviews_test], axis=1)
# X_train_.shape, X_test_.shape
clean_text('aa'), clean_text('abused'), clean_text('abusive')
# X_train_.isnull().sum()
# X_train_.dropna(inplace=True)

# X_test_.dropna(inplace=True)
X_train.columns
dummy_cols=['place_enc','status_enc']

train_dummies=pd.get_dummies(data=X_train, columns=dummy_cols)

test_dummies=pd.get_dummies(data=X_test, columns=dummy_cols)
t_dummy_cols = [ col for col in X_train.columns.tolist() if col not in dummy_cols]
train_dummies.drop(columns=t_dummy_cols, inplace=True)

test_dummies.drop(columns=t_dummy_cols, inplace=True)
train_dummies.dtypes
train_dummies.shape, test_dummies.shape
train_cat = pd.concat([train_dummies, reviews_train], axis=1)

test_cat = pd.concat([test_dummies, reviews_test], axis=1)
train_cat.shape, test_cat.shape
train_cat.head()
X_train.drop(columns=['place_enc', 'status_enc'], inplace=True)

X_test.drop(columns=['place_enc', 'status_enc'], inplace=True)
X_train.shape, X_test.shape
X_train.head()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train.columns[:10]
# col_to_scale = ['len_pos', 'len_neg', 'num_words_pos', 'num_words_neg', 'num_words_sum', 'len_sum']

col_to_scale = X_train.columns.tolist()

scaled_tr_values = scaler.fit_transform(X_train[col_to_scale])

scaled_ts_values = scaler.transform(X_test[col_to_scale])
X_train.loc[:,col_to_scale] = scaled_tr_values

X_test.loc[:,col_to_scale] = scaled_ts_values
X_test.head()
model_XGB = XGBClassifier(random_state=5, n_jobs=-1)

model_XGB.fit(train_cat, y_train)
# pred_XGB = model_XGB.predict(test_cat)

# print('XGB_Train_Accuracy: ', model_XGB.score(train_cat, y_train))

# print('XGB_Test_Accuracy: ', accuracy_score(y_test, pred_XGB))

# print('XGB_F1_score: ', f1_score(y_test, pred_XGB, average='weighted'))
pred_test_XGB = model_XGB.predict_proba(test_cat)
pred_test_XGB
pb_col=['pb1', 'pb2', 'pb3', 'pb4', 'pb5']
prob_sc = pd.DataFrame(data=pred_test_XGB, columns=pb_col)
prob_sc.shape
X_test.shape
# from random import sample

data_tr = pd.concat([prob_sc, X_test], axis = 1)

# index_trn = sample(list(data_tr.index),round(len(data_tr)*0.8))
y_test.reset_index(drop=True, inplace=True)
Xf_train, Xf_test, yf_train, yf_test = train_test_split(data_tr, y_test, random_state=4)
Xf_train.shape, yf_train.shape
logi1 = LogisticRegression('l2',1,.01,.05,1,solver='liblinear',max_iter=500)

logi1.fit(Xf_train,yf_train)

prediction_logi1 = logi1.predict(Xf_test)
print('Train_Accuracy: ', logi1.score(Xf_train,yf_train))

print('Test_Accuracy: ', accuracy_score(yf_test, prediction_logi1))

print('F1_score: ', f1_score(yf_test, prediction_logi1, average='weighted'))
# model_NB = MultinomialNB()

# model_NB.fit(train_cat, y_train)



# prediction_NB = model_NB.predict_proba(test_cat)
# # print('AUC: ', roc_auc_score(y_test, prediction))

# print('NB_Train_Accuracy: ', model_NB.score(X_train_, y_train))

# print('NB_Test_Accuracy: ', accuracy_score(y_test, prediction_NB))

# print('NB_F1_score: ', f1_score(y_test, prediction_NB, average='weighted'))
# X_prob_NB = model_NB.predict_proba(X_test_)
# X_prob_NB[:5]
# y_test[:10]
# prediction_NB[:10]
# model_DTC = DecisionTreeClassifier(random_state=7)

# model_DTC.fit(X_train_, y_train)



# prediction_DTC = model_DTC.predict(X_test_)
# print('DTC_Train_Accuracy: ', model_DTC.score(X_train_, y_train))

# print('DTC_Test_Accuracy: ', accuracy_score(y_test, prediction_DTC))

# print('DTC_F1_score: ', f1_score(y_test, prediction_DTC, average='weighted'))
# # # num = [5 , 7 , 9, 11]

# # num = [5]

# # for k in num:

# k=5

# model_KNN = KNeighborsClassifier(n_neighbors=k)

# model_KNN.fit(X_train_, y_train)



# prediction_KNN = model_KNN.predict(X_test_)
# print('Num_Neighbors: ', k)

# print('KNN_Train_Accuracy: ', model_KNN.score(X_train_, y_train))

# print('KNN_Test_Accuracy: ', accuracy_score(y_test, prediction_KNN))

# print('KNN_F1_score: ', f1_score(y_test, prediction_KNN, average='weighted'))
# model_RDF = RandomForestClassifier(n_estimators=300, random_state=5, n_jobs=-1)

# model_RDF.fit(X_train_, y_train)



# prediction_RDF = model_RDF.predict(X_test_)
# # print('AUC: ', roc_auc_score(y_test, prediction))

# print('RDF_Accuracy: ', accuracy_score(y_test, prediction_RDF))

# print('RDF_F1_score: ', f1_score(y_test, prediction_RDF, average='weighted'))
# model_SGD = SGDClassifier(random_state=5, n_jobs=-1)

# model_SGD.fit(X_train_, y_train)



# prediction_SGD = model_SGD.predict(X_test_)
# # print('AUC: ', roc_auc_score(y_test, prediction))

# print('SGD_Train_Accuracy: ', model_SGD.score(X_train_, y_train))

# print('SGD_Test_Accuracy: ', accuracy_score(y_test, prediction_SGD))

# print('SGD_F1_score: ', f1_score(y_test, prediction_SGD, average='weighted'))
# model_LGR = LogisticRegression(random_state=5, n_jobs=-1)

# model_LGR.fit(X_train_, y_train)



# prediction_LGR = model_LGR.predict(X_test_)
# print('LGR_Train_Accuracy: ', model_LGR.score(X_train_, y_train))

# print('LGR_Test_Accuracy: ', accuracy_score(y_test, prediction_LGR))

# print('LGR_F1_score: ', f1_score(y_test, prediction_LGR, average='weighted'))
# model_ET = ExtraTreeClassifier(random_state=5)

# model_ET.fit(X_train_, y_train)



# prediction_ET = model_ET.predict(X_test_)
# print('ET_Train_Accuracy: ', model_ET.score(X_train_, y_train))

# print('ET_Test_Accuracy: ', accuracy_score(y_test, prediction_ET))

# print('ET_F1_score: ', f1_score(y_test, prediction_ET, average='weighted'))