import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

import re

import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer

import string

from sklearn.utils import resample

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.preprocessing import StandardScaler

from collections import Counter





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

trainData = pd.read_csv("/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv")
trainData.head(5)
trainData.shape
def calMissingData(df):

    total = df.isnull().sum().sort_values(ascending=False)

    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data



def fillNa(df):

    for col in df.columns:

        df[col] = df[col].fillna("")

    return df  
print(calMissingData(trainData).head())
trainData['Category'] = trainData['Category'].map(dict(spam=1, ham=0))
trainData[trainData['Category']==1].shape
sns.distplot(trainData['Category'])
down_sampled_0 = resample(trainData[trainData.Category==0],replace=True,n_samples=1000,random_state=27)
trainData_sampled =  pd.concat([down_sampled_0,trainData[trainData.Category!=0]])
sns.distplot(trainData_sampled['Category'])
stopwords = nltk.corpus.stopwords.words('english')

ps = nltk.PorterStemmer()

SIA = SentimentIntensityAnalyzer()



def count_words(text):

    return len(str(text).split())



def body_len(text):

    return len(str(text))



def avg_word_len(text):

    if(len(str(text)) - text.count(" ")>0):

        return len(str(text))/len(str(text).split())

    return 0



def count_stop_words(text):

    return len([w for w in str(text).lower().split() if w in stopwords])



def count_punct(text):

    count = sum([1 for char in text if char in string.punctuation])

    if len(text)>0:

        return round(count/(len(text) - text.count(" ")), 3)*100

    else:

        return 0

    

def clean_text(text):

    text = "".join([word.lower() for word in text if word not in string.punctuation])

    tokens = re.split('\W+', text)

    text = [ps.stem(word) for word in tokens if word not in stopwords]

    return text



def count_title_words(text):

    return len([w for w in str(text).replace("I",'i').replace("A","a").split() if w.istitle() == True])



def sentimentScore(text):

    return SIA.polarity_scores(str(text))['compound']



def count_nouns(text):

    nouns = ['NN','NNS','NNP','NNPS']

    c = Counter(tup[1] for tup in nltk.pos_tag(i for i in clean_text(text) if i))

    return sum(v for k, v in c.items() if k in nouns)



def count_pronouns(text):

    pronouns = ['PRP','PRP$','WP','WP$']

    c = Counter(tup[1] for tup in nltk.pos_tag(i for i in clean_text(text) if i))

    return sum(v for k, v in c.items() if k in pronouns)

    

def count_verbs(text):

    verbs = ['VB','VBD','VBG','VBN','VBP','VBZ']

    c = Counter(tup[1] for tup in nltk.pos_tag(i for i in clean_text(text) if i))

    return sum(v for k, v in c.items() if k in verbs)



def count_adverbs(text):

    adverbs = ['RB','RBR','RBS','WRB']

    c = Counter(tup[1] for tup in nltk.pos_tag(i for i in clean_text(text) if i))

    return sum(v for k, v in c.items() if k in adverbs)



def count_adj(text):

    adjectives = ['JJ','JJR','JJS']

    c = Counter(tup[1] for tup in nltk.pos_tag(i for i in clean_text(text) if i))

    return sum(v for k, v in c.items() if k in adjectives)    
trainData_sampled['count_words'] = trainData_sampled['Message'].apply(lambda x: count_words(x))

trainData_sampled['body_len'] = trainData_sampled['Message'].apply(lambda x: body_len(x))

trainData_sampled['avg_word_len'] = trainData_sampled['Message'].apply(lambda x: avg_word_len(x))

trainData_sampled['count_stop_words'] = trainData_sampled['Message'].apply(lambda x: count_stop_words(x))

trainData_sampled['punct%'] = trainData_sampled['Message'].apply(lambda x: count_punct(x))

trainData_sampled['count_title_words'] = trainData_sampled['Message'].apply(lambda x: count_title_words(x))

trainData_sampled['text_sentiment'] = trainData_sampled['Message'].apply(lambda x:sentimentScore(x))

trainData_sampled['count_nouns'] = trainData_sampled['Message'].apply(lambda x: count_nouns(x))

trainData_sampled['count_pronouns'] = trainData_sampled['Message'].apply(lambda x: count_pronouns(x))

trainData_sampled['count_verbs'] = trainData_sampled['Message'].apply(lambda x: count_verbs(x))

trainData_sampled['count_adverbs'] = trainData_sampled['Message'].apply(lambda x: count_adverbs(x))

trainData_sampled['count_adj'] = trainData_sampled['Message'].apply(lambda x: count_adj(x))
def plot_bar_chart_from_dataframe(dataframe1,key_column,columns_to_be_plotted):

    import pandas as pd

    test_df1 = dataframe1.groupby(key_column).sum()

    test_df2 = pd.DataFrame()

    for column in columns_to_be_plotted:

        test_df2[column] = round(test_df1[column]/ test_df1[column].sum()*100,2)

    test_df2 = test_df2.T 

    ax = test_df2.plot(kind='bar', stacked=True, figsize =(10,5),legend = 'reverse',title = '% usage for each target')

    for p in ax.patches:

        a = p.get_x()+0.4

        ax.annotate(str(p.get_height()), (a, p.get_y()), xytext=(5, 10), textcoords='offset points')
plot_bar_chart_from_dataframe(trainData_sampled,'Category',['count_words','body_len','avg_word_len','count_stop_words','punct%','count_title_words','text_sentiment'])
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(trainData_sampled.drop(['Category'],axis=1), trainData_sampled['Category'], test_size=0.2)
'''skiping ngram first case'''

tfidf_vect = TfidfVectorizer(analyzer=clean_text)

tfidf_vect_fit = tfidf_vect.fit(X_train['Message'])



tfidf_train = tfidf_vect_fit.transform(X_train['Message'])

tfidf_test = tfidf_vect_fit.transform(X_test['Message'])



X_train_vect = pd.concat([X_train.drop(['Message'],axis=1).reset_index(drop=True), 

           pd.DataFrame(tfidf_train.toarray())], axis=1)

X_test_vect = pd.concat([X_test.drop(['Message'],axis=1).reset_index(drop=True), 

           pd.DataFrame(tfidf_test.toarray())], axis=1)



X_train_vect.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train_vect)

X_train_vect = scaler.transform(X_train_vect)

X_test_vect = scaler.transform(X_test_vect)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import precision_recall_fscore_support as score

import time

import lightgbm as lgb
lgb_train = lgb.Dataset(X_train_vect, y_train)

lgb_eval = lgb.Dataset(X_test_vect, y_test, reference=lgb_train)
# specify your configurations as a dict

params = {

    'boosting_type': 'gbdt',

    'objective': 'multiclass',

    'num_class':3,

    'metric': 'multi_logloss',

    'num_leaves': 10,

    'learning_rate': 0.05,

    'feature_fraction': 0.5,

    'bagging_fraction': 0.5,

    'bagging_freq': 5,

    'verbose': 0,

    'lambda_l2': 0.5

}
start = time.time()

gbm = lgb.train(params,

                lgb_train,

                num_boost_round=300,

                valid_sets=lgb_eval,

                early_stopping_rounds=10)

end = time.time()

fit_time = (end - start)
from sklearn.metrics import accuracy_score



y_pred = gbm.predict(X_test_vect, num_iteration=gbm.best_iteration)



predictions = []

for x in y_pred:

    predictions.append(np.argmax(x))



accuracy_score(y_test, predictions)