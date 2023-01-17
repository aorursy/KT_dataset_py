# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import warnings

warnings.filterwarnings('ignore')

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import lightgbm  as lgb



import os

import json

import string

import numpy as np

import pandas as pd

from pandas.io.json import json_normalize

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline



from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train['set'] = 'train'

test['set'] = 'test'
traintest = pd.concat([train,test])

traintest['texts'] = traintest['Complaint-reason'] + ' ' + traintest['Consumer-complaint-summary']
import re

import string

import nltk

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words(['english','french','spanish']))

def convert_lower(x):

    return x.lower().strip()

def remove_numbers(x):

    x = re.sub(r'\d+[$]','##d',x)

    return re.sub(r'\d+|[x]+|[/]+','',x )

def remove_punctuation(x):

    translator = str.maketrans('', '', string.punctuation)

    return x.translate(translator)

def stop_word_removal(x):

    tokens = word_tokenize(x)

    result = [i for i in tokens if not i in stop_words]

    return ' '.join(result)



def stem_words(x):

    ans = []

    stemmer= PorterStemmer()

    input_str=word_tokenize(x)

    for word in input_str:

        ans.append(stemmer.stem(word))

    return ' '.join(ans)

def preprocess_text(x):

    x = convert_lower(x)

    x = remove_numbers(x)

    x = remove_punctuation(x)

    x = stop_word_removal(x)

    x = stem_words(x)

    return x

# preprocess_text(traintest['texts'].iloc[2243])

# from nltk.tokenize import word_tokenize

# tokens = word_tokenize(input_str)

# result = [i for i in tokens if not i in stop_words]

traintest['texts'] = traintest['texts'].apply(lambda x: preprocess_text(x))
from wordcloud import WordCloud, STOPWORDS

STOPWORDS = stop_words = set(stopwords.words(['english','french','spanish']))

# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##

def plot_wordcloud(text, mask=None, max_words=400, max_font_size=100, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color='black',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=800, 

                    height=400,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  

    

plot_wordcloud(traintest["texts"], title="Word Cloud of Complaint-Summary")

train['Complaint-Status'].unique()
from collections import defaultdict

from nltk.corpus import stopwords

STOPWORDS  = set(stopwords.words(['english','french','spanish']))

train0_df = traintest[train['Complaint-Status']=='Closed with explanation']

train1_df = traintest[train['Complaint-Status']=='Closed with non-monetary relief']

train2_df = traintest[train['Complaint-Status']=='Closed']

train3_df = traintest[train['Complaint-Status']=='Closed with monetary relief']

train4_df = traintest[train['Complaint-Status']=='Untimely response']

## custom function for ngram generation ##

def generate_ngrams(text, n_gram=1):

    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]



## custom function for horizontal bar chart ##

def horizontal_bar_chart(df, color):

    trace = go.Bar(

        y=df["word"].values[::-1],

        x=df["wordcount"].values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace



## Get the bar chart from target ##

freq_dict = defaultdict(int)

for sent in train0_df["texts"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



## Get the bar chart from insincere questions ##

freq_dict = defaultdict(int)

for sent in train1_df["texts"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



freq_dict = defaultdict(int)

for sent in train2_df["texts"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace2 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

#class 3

freq_dict = defaultdict(int)

for sent in train3_df["texts"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace3 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

#class 4

freq_dict = defaultdict(int)

for sent in train1_df["texts"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace4 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



# Creating two subplots

fig = tools.make_subplots(rows=3, cols=2, vertical_spacing=0.04,

                          subplot_titles=["Frequent words of closed with explanation", 

                                          "Frequent words of closed with non-monatory relief",

                                          "Frequent words of closed","Frequent Words of closed with monetory relief",

                                          "Untimely response", ""])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 2, 1)

fig.append_trace(trace3, 2, 2)

fig.append_trace(trace4, 3, 1)

fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")

py.iplot(fig, filename='word-plots')



#plt.figure(figsize=(10,16))

#sns.barplot(x="ngram_count", y="ngram", data=fd_sorted.loc[:50,:], color="b")

#plt.title("Frequent words for Insincere Questions", fontsize=16)

#plt.show()
## Get the bar chart from target ##

freq_dict = defaultdict(int)

for sent in train0_df["texts"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



## Get the bar chart from insincere questions ##

freq_dict = defaultdict(int)

for sent in train1_df["texts"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



freq_dict = defaultdict(int)

for sent in train2_df["texts"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace2 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

#class 3

freq_dict = defaultdict(int)

for sent in train3_df["texts"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace3 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

#class 4

freq_dict = defaultdict(int)

for sent in train1_df["texts"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace4 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



# Creating two subplots

fig = tools.make_subplots(rows=3, cols=2, vertical_spacing=0.04,

                          subplot_titles=["Frequent bigram words of closed with explanation", 

                                          "Frequent bigram words of closed with non-monatory relief",

                                          "Frequentbigram words of closed","Frequent bigrams Words of closed with monetory relief",

                                          "bigrams words of Untimely response", ""])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 2, 1)

fig.append_trace(trace3, 2, 2)

fig.append_trace(trace4, 3, 1)

fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")

py.iplot(fig, filename='word-plots')

## Get the bar chart from target ##

freq_dict = defaultdict(int)

for sent in train0_df["texts"]:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



## Get the bar chart from insincere questions ##

freq_dict = defaultdict(int)

for sent in train1_df["texts"]:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



freq_dict = defaultdict(int)

for sent in train2_df["texts"]:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace2 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

#class 3

freq_dict = defaultdict(int)

for sent in train3_df["texts"]:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace3 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

#class 4

freq_dict = defaultdict(int)

for sent in train1_df["texts"]:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace4 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



# Creating two subplots

fig = tools.make_subplots(rows=5, cols=1, vertical_spacing=0.04,

                          subplot_titles=["Frequent bigram words of closed with explanation", 

                                          "Frequent bigram words of closed with non-monatory relief",

                                          "Frequentbigram words of closed","Frequent bigrams Words of closed with monetory relief",

                                          "bigrams words of Untimely response", ""])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 2, 1)

fig.append_trace(trace2, 3, 1)

fig.append_trace(trace3, 4, 1)

fig.append_trace(trace4, 5, 1)

fig['layout'].update(height=1200, width=1000, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")

py.iplot(fig, filename='word-plots')
traintest.to_csv('bowTFIDF.csv',index = False)
ls
from sklearn.feature_extraction.text import TfidfVectorizer
TfidfVectorizer()


# import catboost as cb

# from catboost import Pool



# cb_clf=cb.CatBoostClassifier()
# X
# word_vectorizer = TfidfVectorizer(

#     sublinear_tf=True,

#     strip_accents='unicode',

#     analyzer='word',

#     token_pattern=r'\w{1,}',

#     stop_words='english',

#     ngram_range=(1, 1),

#     max_features=10000)

# word_vectorizer.fit(traintest['texts'])

# traintest_word_features = word_vectorizer.transform(traintest['texts'])

# # test_word_features = word_vectorizer.transform(test_text)



# char_vectorizer = TfidfVectorizer(

#     sublinear_tf=True,

#     strip_accents='unicode',

#     analyzer='char',

#     stop_words='english',

#     ngram_range=(2, 6),

#     max_features=20000)



# char_vectorizer.fit(traintest['texts'])

# traintest_char_features = char_vectorizer.transform(traintest['texts'])


# vectorizer = TfidfVectorizer(ngram_range=(2,3),min_df = 5,max_df= 10000)

vectorizer = TfidfVectorizer(ngram_range=(1,3),

               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,

               smooth_idf=1, sublinear_tf=1 )

X = vectorizer.fit_transform(traintest['texts'])

traintestc = vectorizer.transform(traintest['texts'].values.tolist())
# from scipy import sparse

# traintestc = sparse.hstack([traintest_char_features, traintest_word_features]).tocsr()

# train_tfidf = traintestc[:43266]

# test_tfidf = traintestc[43266:]

from sklearn.feature_selection import chi2
traintestc
# features_chi2 = chi2(train_tfidf,train['Complaint-Status'].values)
# features_ind = np.argsort(features_chi2[0])

# feature_names = np.array(vectorizer.get_feature_names())[features_ind]
# selected = feature_names[:10000]

# traintestc=X[:,features_ind[:390]]
# N = 3

# for Product, category_id in sorted(category_to_id.items()):

#   features_chi2 = chi2(features, labels == category_id)

#   indices = np.argsort(features_chi2[0])

#   feature_names = np.array(tfidf.get_feature_names())[indices]

#   unigrams = [v for v in feature_names if len(v.split(' ')) == 1]

#   bigrams = [v for v in feature_names if len(v.split(' ')) == 2]

#   print("# '{}':".format(Product))

#   print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))

#   print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
gp = train[['Complaint-ID','Complaint-Status']].groupby('Complaint-Status')

gp['Complaint-Status'].apply(lambda x: x.count())
for ttype in  train['Transaction-Type'].unique():

    tempdf = train[train['Transaction-Type']==ttype][['Transaction-Type','Complaint-Status']]

    tempdf = tempdf.groupby('Complaint-Status')['Complaint-Status'].apply(lambda x : x.count())

    plt.figure(figsize=(15,5))

    sns.barplot(tempdf.index, tempdf.values, alpha=0.8)

    plt.title(ttype+' transaction with target')

    plt.show()

    

    

    
train['Date-received']=pd.to_datetime(train['Date-received'])

train['Date-sent-to-company'] = pd.to_datetime(train['Date-sent-to-company'])

test['Date-received']=pd.to_datetime(test['Date-received'])

test['Date-sent-to-company'] = pd.to_datetime(test['Date-sent-to-company'])
train['Diff_days'] = train['Date-sent-to-company'] - train['Date-received']

test['Diff_days'] = test['Date-sent-to-company'] - test['Date-received']

train['Diff_days'] = train.Diff_days.apply(lambda x: x.days)

test['Diff_days'] = test.Diff_days.apply(lambda x: x.days)
train.groupby('Complaint-Status')['Diff_days'].apply(lambda x: x.sum()/x.count())
train['month'] = train['Date-received'].dt.month
for mnth in train.month.unique():

    tdf = train[train.month==mnth][['Complaint-Status','month']]

    tdf = tdf.groupby('Complaint-Status')['Complaint-Status'].apply(lambda x : x.count())

    print(tdf.head())

    plt.figure(figsize=(12,5))

    sns.barplot(tdf.index,tdf.values)

    plt.title('month '+str(mnth))

    plt.show()

    

    
train.groupby('month')['Complaint-ID'].apply(lambda x: x.count())
for cstatus in train['Complaint-Status'].unique():

    tdf = train[train['Complaint-Status']==cstatus][['Complaint-Status','Consumer-disputes']]

    tdf = tdf.groupby(['Consumer-disputes'])['Consumer-disputes'].apply(lambda x: x.count())

    print(tdf.head())

    

    sns.barplot(tdf.index,tdf.values)

    plt.title('target '+ cstatus)

    plt.show()

    

    
from sklearn.preprocessing import LabelEncoder as le

from sklearn.naive_bayes import MultinomialNB

# clf = MultinomialNB().fit(train_tfidf, twenty_train.target)
le1 = le()

le2 = le()

le3 = le()

le4 = le()

print('Null counts')

for col in train.columns:

    t = pd.isnull(train[col]).sum()

    print(col,' : ',t)
train['Company-response'].fillna(value='New CAT',inplace =True)

train['Consumer-disputes'].fillna(value = "Not Known",inplace =True)

X = train.copy(deep=True)

test['Company-response'].fillna(value='New CAT',inplace =True)

test['Consumer-disputes'].fillna(value = "Not Known",inplace =True)
train['Transaction-Type'] = pd.Series(le1.fit_transform(train['Transaction-Type']))

train['Complaint-reason'] = pd.Series(le2.fit_transform(train['Complaint-reason']))

train['Company-response'] = pd.Series(le3.fit_transform(train['Company-response']))

train['Consumer-disputes'] = pd.Series(le3.fit_transform(train['Consumer-disputes']))

train['Complaint-Status'] = pd.Series(le4.fit_transform(train['Complaint-Status']))
test['Transaction-Type'] = pd.Series(le1.fit_transform(test['Transaction-Type']))

test['Complaint-reason'] = pd.Series(le2.fit_transform(test['Complaint-reason']))

test['Company-response'] = pd.Series(le3.fit_transform(test['Company-response']))

test['Consumer-disputes'] = pd.Series(le3.fit_transform(test['Consumer-disputes']))
use_col = ['Transaction-Type','Company-response','Consumer-disputes', 'Diff_days']

target = 'Complaint-Status'
import scipy

# trainAll = scipy.sparse.hstack([train_tfidf, train[use_col]])
from sklearn.neural_network import MLPClassifier

from sklearn.decomposition import TruncatedSVD

from sklearn.feature_selection import chi2,SelectFpr, SelectFromModel,SelectKBest,f_classif

from sklearn.preprocessing import MinMaxScaler

ss = MinMaxScaler()

ss.fit(train.Diff_days.values.reshape((-1,1)))

train['Diff_days'] = pd.DataFrame(ss.transform(train.Diff_days.values.reshape((-1,1))))

test['Diff_days'] = pd.DataFrame(ss.transform(test.Diff_days.values.reshape((-1,1))))

traincpy = train[use_col].copy()

traincpy = pd.concat([traincpy,pd.get_dummies(traincpy['Transaction-Type'],prefix = 'TT')],axis=1)

traincpy = pd.concat([traincpy,pd.get_dummies(traincpy['Consumer-disputes'],prefix= 'Consumer_Dispute')],axis = 1)

traincpy = pd.concat([traincpy,pd.get_dummies(traincpy['Company-response'],prefix= 'company_response')],axis = 1)

traincpy.drop(columns=['Transaction-Type','Consumer-disputes','Company-response'],axis = 1,inplace=True)

traincpy.head()



testcpy = test[use_col].copy()

testcpy = pd.concat([testcpy,pd.get_dummies(testcpy['Transaction-Type'],prefix = 'TT')],axis=1)

testcpy = pd.concat([testcpy,pd.get_dummies(testcpy['Consumer-disputes'],prefix= 'Consumer_Dispute')],axis = 1)

testcpy = pd.concat([testcpy,pd.get_dummies(testcpy['Company-response'],prefix= 'company_response')],axis = 1)

testcpy.drop(columns=['Transaction-Type','Consumer-disputes','Company-response'],axis = 1,inplace=True)

testcpy.head()
from scipy import sparse

train_tfidf = traintestc[:43266]

test_tfidf = traintestc[43266:]



train_tfidf = sparse.hstack([train_tfidf,sparse.csr_matrix(traincpy)]).tocsr()

test_tfidf = sparse.hstack([test_tfidf,sparse.csr_matrix(testcpy)]).tocsr()
print(train_tfidf.shape,test_tfidf.shape)
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from sklearn.svm import LinearSVC

from sklearn import model_selection

import xgboost as xgb



# >>> iris = datasets.load_iris()

# >>> X, y = iris.data, iris.target

from sklearn.metrics import f1_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

# from sklearn.feature_extraction.text import CountVectorizer

# from sklearn.feature_extraction.text import TfidfTransformer



def f1_eval(y_pred, dtrain):



    y_true = dtrain.get_label()

#     print('y_true',y_pred)

    err = 1-f1_score(y_true,y_pred,average='weighted')

    print( 'F1 score : {}'.format(f1_score(y_true,y_pred,average='weighted')))

    return 'f1_err', err



    
# model = xgb.XGBClassifier(objective='multi:softmax',num_class= 5,eval_metric = f1_eval)
# model = xgb.XGBClassifie
model = OneVsRestClassifier(LinearSVC(verbose=1,class_weight='balanced',C = 5.0 ,random_state=0,max_iter=4000))

# model = LogisticRegression(C=0.1, solver='sag',multi_class='multinomial')

# # model2 = KNeighborsClassifier(n_neighbors=20)

# # model2 = RandomForestClassifier()



fs = SelectFromModel(model)

# # fs = SelectKBest(f_classif,k = 200000)

# # fs = SelectFpr()

train_tfidf = fs.fit_transform(train_tfidf,train['Complaint-Status'])

test_tfidf = fs.transform(test_tfidf)



                            
# # from sklearn.feature_selection import RFE

# model = OneVsrestClassifier(LinearSVC(C = 5.0 , random_state=0))

# model2 = MultinomialNB()

# # selector = RFE(model, 390, step=20000)

# # selector = selector.fit(train_tfidf, train['Complaint-Status'].values)

# from sklearn.pipeline import Pipeline

# text_clf = Pipeline([('tfidf', vectorizer),('clf', model)],)

# xgb.train
# train_y = train['Complaint-Status'].values

# params = {'max_depth': 2,"min_child_weight" : 30,'objective' :'multi:softmax','num_class' :5,'silent': 1 ,'gamma' : 10,'eval_metric':'auc','early_stopping_rounds':100}

# model = xgb.train(params,train_tfidf,1000)

# # #     model = KNeighborsClassifier(n_neighbors=20) 

# # #     model = RandomForestClassifier()

# print('training_started')

# model.fit(train_X, train_y)

# #     print(model.predict(test_X))

# pred_test_y = model.predict(train_tfidf)

# pred_test_y2 = model.predict(test_tfidf)

train_tfidf.shape
train_y = train['Complaint-Status'].values

# train_tfidf = train_tfidf.toarray()

# test_tfidf = test_tfidf.toarray()

# test_tfidf = traintestc[:43266,features_ind[:232000]]



def runModel(train_X, train_y, test_X, test_y, test_X2):

#     model = linear_model.LogisticRegression(C=5., solver='sag')

#     model = OneVsOneClassifier(LinearSVC(C = 1.0 , random_state=0))

#     model = LogisticRegression(C=0.1, solver='sag',multi_class='multinomial')

    model = OneVsRestClassifier(LinearSVC(verbose=1,class_weight='balanced',C = 5.0 ,random_state=0))

#     params = {'eta': 0.03, 'max_depth': 6,"min_child_weight" : 30,'objective' :'multi:softmax','num_class' :5,'silent': 1 ,'gamma' : 10}

#     train_X = xgb.DMatrix(train_X,label= train_y)

#     test_X = xgb.DMatrix(test_X,label=test_y)

#     model = KNeighborsClassifier(n_neighbors=20) 

#     model = RandomForestClassifier()

#     test_X2 = xgb.DMatrix(test_X2)

    print('training_started')

#     evallist  = [(test_X, 'eval')]

#     model = xgb.train(params,train_X,2000,evallist,feval=f1_eval)

    model.fit(train_X,train_y)

#     print(model.predict(test_X))

    pred_test_y = model.predict(test_X)

    pred_test_y2 = model.predict(test_X2)

    return pred_test_y, pred_test_y2, model



print("Building model.")

cv_scores = []

pred_full_test = 0

pred_train = pd.Series(np.zeros([train.shape[0]]))

pred_full_test  = pd.Series(np.zeros([test.shape[0]]))

kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train):

    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_val_y, pred_test_y, model = runModel(dev_X, dev_y, val_X, val_y, test_tfidf)

    pred_full_test = pd.concat([pred_full_test,pd.Series(pred_test_y)],axis=1)

    pred_train[val_index] = pred_val_y

    cv_scores.append(f1_score(val_y, pred_val_y,average='weighted'))

    print(cv_scores[-1])

    break
test['Complaint-Status'] = pd.Series(le4.inverse_transform(pred_full_test[1].apply(lambda x: int(x))))

test[['Complaint-ID','Complaint-Status']].to_csv('tfidf2metakbestc01.csv',index = False)
train_tfidf = traintestc[:43266]

test_tfidf = traintestc[43266:]



train_tfidf = sparse.hstack([train_tfidf,sparse.csr_matrix(traincpy)]).tocsr()

test_tfidf = sparse.hstack([test_tfidf,sparse.csr_matrix(testcpy)]).tocsr()
from sklearn.linear_model import LogisticRegression
# # model = LogisticRegression(C=5.0, solver='sag',multi_class='multinomial')

# fs = SelectKBest(f_classif,k = 20000)



# train_tfidf = fs.fit_transform(train_tfidf,train['Complaint-Status'])

# test_tfidf = fs.transform(test_tfidf)
def runModel(train_X, train_y, test_X, test_y, test_X2):

#     model = linear_model.LogisticRegression(C=5., solver='sag')

#     model = OneVsOneClassifier(LinearSVC(C = 1.0 , random_state=0))

    model = LogisticRegression(C=1.0, solver='sag',multi_class='multinomial')

#     model = OneVsRestClassifier(LinearSVC(verbose=1,class_weight='balanced',C = 5.0 ,random_state=0))

#     params = {'eta': 0.03, 'max_depth': 6,"min_child_weight" : 30,'objective' :'multi:softmax','num_class' :5,'silent': 1 ,'gamma' : 10}

#     train_X = xgb.DMatrix(train_X,label= train_y)

#     test_X = xgb.DMatrix(test_X,label=test_y)

#     model = KNeighborsClassifier(n_neighbors=20) 

#     model = RandomForestClassifier()

#     test_X2 = xgb.DMatrix(test_X2)

    print('training_started')

#     evallist  = [(test_X, 'eval')]

#     model = xgb.train(params,train_X,2000,evallist,feval=f1_eval)

    model.fit(train_X,train_y)

#     print(model.predict(test_X))

    pred_test_y = model.predict(test_X)

    pred_test_y2 = model.predict(test_X2)

    return pred_test_y, pred_test_y2, model



print("Building model.")

cv_scores = []

pred_full_test2 = 0

pred_train = pd.Series(np.zeros([train.shape[0]]))

pred_full_test2  = pd.Series(np.zeros([test.shape[0]]))

pred_full_test3  = pd.Series(np.zeros([test.shape[0]]))

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train):

    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_val_y, pred_test_y, model = runModel(dev_X, dev_y, val_X, val_y, test_tfidf)

    pred_full_test2 = pd.concat([pred_full_test2,pd.Series(pred_test_y)],axis=1)

    pred_train[val_index] = pred_val_y

    cv_scores.append(f1_score(val_y, pred_val_y,average='weighted'))

    print(cv_scores[-1])

    break
ans = pd.concat([pred_full_test[1],pred_full_test2[1]],axis=1)

ans.columns = ['SVM','LR']
ans['probLR'] = pd.Series(np.max(model.predict_proba(test_tfidf),axis=1))
# ans[(ans.SVM !=ans.LR )& (ans.probLR > 0.75)]

p = ans.apply( lambda x : x.LR if  (x.SVM != x.LR) and x.probLR >=0.75 else x.SVM,axis=1)
test['Complaint-Status'] = pd.Series(le4.inverse_transform(p.apply(lambda x: int(x))))

test[['Complaint-ID','Complaint-Status']].to_csv('ensembleLRSVM.csv',index = False)
model.pre