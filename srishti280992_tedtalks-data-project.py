# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import scipy.stats as stats

import datetime

import re

import itertools

from plotly.offline import init_notebook_mode, iplot

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt

import spacy 

from collections import Counter

import re

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfTransformer

from scipy.sparse import coo_matrix

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from sklearn.metrics import classification_report,accuracy_score

import sklearn.metrics as metrics

from keras.preprocessing.text import Tokenizer

from nltk.tokenize import word_tokenize

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D

from keras.initializers import Constant

from keras.optimizers import Adam

from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc

sns.set_style("whitegrid")





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from warnings import filterwarnings

filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df= pd.read_csv('../input/ted-current-views-full/ted_full_v2.csv')

data=df.copy()

df.shape
# taking tags that have occurred more than 180 times to create columns

count_vector = CountVectorizer(stop_words='english',min_df=180/len(data)) 

tag_array = count_vector.fit_transform(data.tags).toarray()

tag_matrix = pd.DataFrame(tag_array, columns = count_vector.get_feature_names())

tag_matrix = tag_matrix.add_prefix('tags_')



# append the columns obtained to the base data

data = pd.concat([data,tag_matrix], axis=1)

data=data.drop(['tags'], axis = 1) # drop tags column

#list(data)
data.head()
# all date operations

data['film_date'] = data['film_date'].apply(lambda x: datetime.date.fromtimestamp(int(x)))

data['published_date'] = data['published_date'].apply(lambda x: datetime.date.fromtimestamp(int(x)))

data['film_month'] = data['film_date'].apply(lambda x: x.month)

data['pub_month'] = data['published_date'].apply(lambda x: x.month)

data['film_weekday'] = data['film_date'].apply(lambda x: x.weekday()) # Monday: 0, Sunday: 6

data['pub_weekday'] = data['published_date'].apply(lambda x: x.weekday())

data[['film_date','published_date']].head()
# pairplots between numerical variables to check for evident patterns and correlations

nums = ['comments', 'duration', 'num_speaker', 'views']

sns.pairplot(data, vars=nums,   size=3);
sns.jointplot(x=data['languages'], y=data['views'], kind='reg').annotate(stats.pearsonr)
sns.jointplot(x=data['views'], y=data['comments'], kind='reg').annotate(stats.pearsonr)
# check relation between duration, comments and views

data_sorted=data.sort_values(by='views',ascending=True)

df2=data_sorted.iloc[:20,:]

df2.index=range(0,len(df2))

#visualization

data_viz = [

    {

        'y': df2.views,

        'x': df2.index,

        'mode': 'markers',

        'marker': {

            'color': df2.duration,

            'size': df2.comments,

            'showscale': True

        },

        "text" :  df2.main_speaker    

    }

]

iplot(data_viz)
data['event'].unique()

    

data['event_category'] = data.event.apply(lambda x: "TEDx" if "TEDx" in x else ("TED" if "TED" in x else "Other"))
data['event_category'].value_counts()
data['duration']= data['duration']/60 # per minute

data['transcript'] = data['transcript'].fillna('')

data['wc_per_min'] = data['transcript'].apply(lambda x: len(x.split()))/data['duration']
data.head()
data.shape
nlp = spacy.load('en')



feats = ['char_count', 'word_count', 'word_count_cln',

       'stopword_count', '_NOUN', '_VERB', '_ADP', '_ADJ', '_DET', '_PROPN',

       '_INTJ', '_PUNCT', '_NUM', '_PRON', '_ADV', '_PART', '_amod', '_advmod', '_acl', '_relcl', '_advcl',

       '_neg','_PERSON','_NORP','_FAC','_ORG','_GPE','_LOC','_PRODUCT','_EVENT','_WORK_OF_ART','_LANGUAGE']



class text_features:

    def __init__(self, df, textcol):

        self.df = df

        self.textcol = textcol

        self.c = "spacy_" + textcol

        self.df[self.c] = self.df[self.textcol].apply( lambda x : nlp(x))

        

        self.pos_tags = ['NOUN', 'VERB', 'ADP', 'ADJ', 'DET', 'PROPN', 'INTJ', 'PUNCT',\

                         'NUM', 'PRON', 'ADV', 'PART']

        self.dep_tags = ['amod', 'advmod', 'acl', 'relcl', 'advcl','neg']

        self.ner_tags = ['PERSON','NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT','WORK_OF_ART','LANGUAGE']

        

    def _spacy_cleaning(self, doc):

        tokens = [token for token in doc if (token.is_stop == False)\

                  and (token.is_punct == False)]

        words = [token.lemma_ for token in tokens]

        return " ".join(words)

        

    def _spacy_features(self):

        self.df["clean_text"] = self.df[self.c].apply(lambda x : self._spacy_cleaning(x))

        self.df["char_count"] = self.df[self.textcol].apply(len)

        self.df["word_count"] = self.df[self.c].apply(lambda x : len([_ for _ in x]))

        self.df["word_count_cln"] = self.df["clean_text"].apply(lambda x : len(x.split()))

        

        self.df["stopword_count"] = self.df[self.c].apply(lambda x : 

                                                          len([_ for _ in x if _.is_stop]))

        self.df["pos_tags"] = self.df[self.c].apply(lambda x :

                                                    dict(Counter([_.head.pos_ for _ in x])))

        self.df["dep_tags"] = self.df[self.c].apply(lambda x :

                                                    dict(Counter([_.dep_ for _ in x])))

        self.df["ner_tags"] = self.df[self.c].apply(lambda x :

                                                    dict(Counter([_.ent_type_ for _ in x])))

        

    def _flatten_features(self):

        for key in self.pos_tags:

            self.df["_" + key] = self.df["pos_tags"].apply(lambda x : \

                                                           x[key] if key in x else 0)

        

        for key in self.dep_tags:

            self.df["_" + key] = self.df["dep_tags"].apply(lambda x : \

                                                           x[key] if key in x else 0)

            

        for key in self.ner_tags:

            self.df["_" + key] = self.df["ner_tags"].apply(lambda x : \

                                                           x[key] if key in x else 0)

                

    def generate_features(self):

        self._spacy_features()

        self._flatten_features()

        self.df = self.df.drop([self.c, "pos_tags", "dep_tags", 'ner_tags',"clean_text"], axis=1)

        return self.df

    

    

def spacy_features(df, tc):

    fe = text_features(df, tc)

    return fe.generate_features()
textcol = "transcript"

transcript_features = spacy_features(data, textcol)

transcript_features[[textcol] + feats].head()
# data['transcript'].str.count("(Laughter)")

# data['transcript'].str.count("(Applause)")
data['laughter_count']=data['transcript'].str.count("(Laughter)")

data['applaud_count']=data['transcript'].str.count("(Applause)")

data.head()
data['clean_text']= data['clean_text'].str.replace('(Laughter)',"laughter")

data['clean_text']= data['clean_text'].str.replace('Applause',"applause")





data['laughter_count']=data['clean_text'].str.count("(laughter)")

data['applaud_count']=data['clean_text'].str.count("applause")

data.head()
data['clean_text']= data['clean_text'].str.replace('(laughter)',"")

data['clean_text']= data['clean_text'].str.replace('applause',"")

data['clean_text']= data['clean_text'].str.replace('(',"")

data['clean_text']= data['clean_text'].str.replace(')',"")

#list(data['clean_text'])
data['views_data_pull_date']= pd.to_datetime("'2020-04-10'".replace("'",""),format='%Y-%m-%d')



data['days_since_publish'] = (data['views_data_pull_date'] - pd.to_datetime(data['published_date'],format='%Y-%m-%d'))/np.timedelta64(1,'D')

#data.views_data_pull_date
data.head()
# data.loc[data['speaker_occupation'].str.contains('Artist|artist') == True, 'speaker_occupation'] = "Artist"

# data.loc[data['speaker_occupation'].str.contains('Author|author|Writer|writer') == True, 'speaker_occupation'] = "Writer"

# data.loc[data['speaker_occupation'].isin(small_occupations) == True, 'speaker_occupation'] = "Other"



data.loc[:, 'super_popular'] = 0

data.loc[data['view_2020'] > 4000000, 'super_popular'] = 1


##Creating a list of stop words and adding custom stopwords

stop_words = set(stopwords.words("english"))

corpus= data['clean_text']

cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))

X=cv.fit_transform(corpus)





 

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)

tfidf_transformer.fit(X)

# get feature names

feature_names=cv.get_feature_names()

 

# fetch document for which keywords needs to be extracted

doc=corpus[1]

 

#generate tf-idf for the given document

tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))



#Function for sorting tf_idf in descending order



def sort_coo(coo_matrix):

    tuples = zip(coo_matrix.col, coo_matrix.data)

    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

 

def extract_topn_from_vector(feature_names, sorted_items, topn=10):

    """get the feature names and tf-idf score of top n items"""

    

    #use only topn items from vector

    sorted_items = sorted_items[:topn]

 

    score_vals = []

    feature_vals = []

    

    # word index and corresponding tf-idf score

    for idx, score in sorted_items:

        

        #keep track of feature name and its corresponding score

        score_vals.append(round(score, 3))

        feature_vals.append(feature_names[idx])

 

    #create a tuples of feature,score

    #results = zip(feature_vals,score_vals)

    results= {}

    for idx in range(len(feature_vals)):

        results[feature_vals[idx]]=score_vals[idx]

    

    return results

#sort the tf-idf vectors by descending order of scores

sorted_items=sort_coo(tf_idf_vector.tocoo())

#extract only the top n; n here is 10

keywords=extract_topn_from_vector(feature_names,sorted_items,5)

 

# now print the results

print("\nAbstract:")

print(doc)

print("\nKeywords:")

for k in keywords:

    print(k,keywords[k])

keywords_arr=[]

for i in range(len(data['clean_text'])):

# fetch document for which keywords needs to be extracted

    doc=corpus[i]



    #generate tf-idf for the given document

    tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

    

    #sort the tf-idf vectors by descending order of scores

    sorted_items=sort_coo(tf_idf_vector.tocoo())

    #extract only the top n; n here is 10

    keywords=extract_topn_from_vector(feature_names,sorted_items,5)

    

    top_5= list(keywords.keys())

    #print(i,"keywords:",top_5)

    keywords_arr.append(top_5)

    

#keywords_arr
pd.DataFrame(keywords_arr)
feature_columns_mod1=['wc_per_min','stopword_count',

                      '_NOUN',

 '_VERB',

 '_ADP',

 '_ADJ',

 '_DET',

 '_PROPN',

 '_INTJ',

 '_PUNCT',

 '_NUM',

 '_PRON',

 '_ADV',

 '_PART',

 '_amod',

 '_advmod',

 '_acl',

 '_relcl',

 '_advcl',

 '_neg',

 '_PERSON',

 '_NORP',

 '_FAC',

 '_ORG',

 '_GPE',

 '_LOC',

 '_PRODUCT',

 '_EVENT',

 '_WORK_OF_ART',

 '_LANGUAGE',

 'laughter_count',

 'applaud_count']



target= ['super_popular']



#list(data[feature_columns_mod1])
data_mod1= data[feature_columns_mod1+target]

X_train, X_test,y_train,y_test =train_test_split(

     data_mod1.drop(['super_popular'], axis = 1), data_mod1[['super_popular']], 

     test_size = 0.2, random_state = 100)



#X_train=X_train.drop(['days_since_publish'], axis = 1)

#X_test=X_test.drop(['days_since_publish'], axis = 1)



# X_train, X_valid, y_train, y_valid = train_test_split(

#     train.drop(['super_popular'], axis = 1), train[['super_popular']], 

#     test_size = 0.2, random_state = 42)
def cv_performance_assessment(X,y,k,clf):

    '''Cross validated performance assessment

    

    X   = training data

    y   = training labels

    k   = number of folds for cross validation

    clf = classifier to use

    

    Divide the training data into k folds of training and validation data. 

    For each fold the classifier will be trained on the training data and

    tested on the validation data. The classifier prediction scores are 

    aggregated and output

    '''

    # Establish the k folds

    prediction_scores = np.empty(y.shape[0],dtype='object')

    kf = StratifiedKFold(n_splits=k, shuffle=True)

    i=1

    for train_index, val_index in kf.split(X, y):

        # Extract the training and validation data for this fold

        print('iteration:',i,",length:",len(train_index),"Train indices selected:",train_index)

        X_train, X_val   = X.iloc[train_index], X.iloc[val_index]

        y_train          = y.iloc[train_index]

        

        # Train the classifier

        clf              = clf.fit(X_train,y_train)

        

        # Test the classifier on the validation data for this fold

        cpred            = clf.predict_proba(X_val)

        

        # Save the predictions for this fold

        prediction_scores[val_index] = cpred[:,1]

        i=i+1

    return prediction_scores
#create LGBM classifier model

gbm_model = lgb.LGBMClassifier(

        boosting_type= "dart",

        n_estimators=1850,

        learning_rate=0.12,

        num_leaves=35,

        colsample_bytree=.8,

        subsample=.9,

        max_depth=9,

        reg_alpha=.1,

        reg_lambda=.1,

        min_split_gain=.01

)
num_training_folds = 10

# get cross validated scores



cv_prediction_scores = cv_performance_assessment(X_train,y_train,num_training_folds,gbm_model)
def plot_roc(labels, prediction_scores):

    '''Obtain the true positive rate, false positive rate and plot ROC curve for a classifier

    

    Input for the function is the ground truth and the predicted probability scores

    AUC will computed using these arguments

    Chance curve is also shown to compare the performance of your classifier

    '''

    fpr, tpr, _ = metrics.roc_curve(labels, prediction_scores, pos_label=1)

    auc = metrics.roc_auc_score(labels, prediction_scores)

    legend_string = 'AUC = {:0.3f}'.format(auc)

   

    plt.plot([0,1],[0,1],'--', color='gray', label='Chance')

    plt.plot(fpr, tpr, label=legend_string)

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.grid(True)

    plt.axis('square')

    plt.legend()

    plt.tight_layout()

    

def plot_prec_recall(labels, prediction_scores):

    '''Plot Precision Recall curve for a classifier

    

    Input for the function is the ground truth and the predicted probability scores

    The majority class predictor is also shown to compare the performance of your classifier to baseline

    '''

    # for your classifier

    precision, recall, _ = metrics.precision_recall_curve(labels, prediction_scores, pos_label=1)

    plt.step(recall,precision,label="classifier")



    # for a classifier that predicts all 0-- majority class

    zero_preds= np.zeros(labels.shape)

    precision2, recall2, _ = metrics.precision_recall_curve(labels, zero_preds, pos_label=1)

    plt.plot(recall2,precision2,'--o',color='grey',label="zero class predictor")

    

    plt.xlabel('recall')

    plt.ylabel('precision')

    plt.grid('on')

    plt.axis('square')

    plt.legend()

    plt.title('Precision Recall curve')

    plt.tight_layout()
# Compute and plot the ROC curves

plot_roc(y_train, cv_prediction_scores)

plt.title("ROC curve from 10-fold cross validation")
# predict probabilities on test set

y_pred_gbm = gbm_model.predict_proba(X_test)

plot_roc(y_test, y_pred_gbm[:,1])

plt.title("ROC curve on 20% test split")
predictions_lgbm_bin = np.where(y_pred_gbm[:,1] > 0.5, 1, 0) #Turn probability to 0-1 binary output
print("Test accuracy:",accuracy_score(y_test,predictions_lgbm_bin))
# fit model on complete data

gbm_model=gbm_model.fit(data[feature_columns_mod1],data[target],verbose=0)





# predict probabilities on complete set

y_pred_total_gbm0 = gbm_model.predict_proba(data[feature_columns_mod1])

predictions_lgbm_bin_mod1 = np.where(y_pred_total_gbm0[:,1] > 0.5, 1, 0) #Turn probability to 0-1 binary out
lgb.plot_importance(gbm_model,max_num_features=15)
# saving both predictions to CSV files

submission_file1 = pd.DataFrame({'id':    data['url'],

                                   'proba':  y_pred_total_gbm0[:,1],

                                   'preds': predictions_lgbm_bin_mod1,

                                   'actuals': data['super_popular']})

submission_file1.to_csv('submission_feats.csv',

                           columns=['id','proba','preds','actuals'],

                           index=False)
feature_columns_mod2=['wc_per_min','stopword_count',

                      'clean_text']



data_mod2= data[feature_columns_mod2+target]
def create_corpus(df):

    corpus=[]

    for transcript in tqdm(df['clean_text']):

        words=[word.lower() for word in word_tokenize(transcript) if((word.isalpha()==1) & (word not in stop_words))]

        corpus.append(words)

    return corpus
corpus=create_corpus(data_mod2)
embedding_dict={}

with open('../input/glove100dtedtranscriptsv1/glove.6B.100d.txt','r') as f:

    for line in f:

        values=line.split()

        word = values[0]

        vectors=np.asarray(values[1:],'float32')

        embedding_dict[word]=vectors

f.close()
MAX_LEN=32

tokenizer_obj=Tokenizer()

tokenizer_obj.fit_on_texts(corpus)

sequences=tokenizer_obj.texts_to_sequences(corpus)



transcript_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')
word_index=tokenizer_obj.word_index

print('Number of unique words:',len(word_index))
num_words=len(word_index)+1

embedding_matrix=np.zeros((num_words,100))



for word,i in tqdm(word_index.items()):

    if i < num_words:

        emb_vec=embedding_dict.get(word)

        if emb_vec is not None:

            embedding_matrix[i]=emb_vec 
model=Sequential()



embedding=Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),

                   input_length=MAX_LEN,trainable=False)



model.add(embedding)

model.add(SpatialDropout1D(0.2))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))





optimzer=Adam(learning_rate=3e-4)



model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])
model.summary()
model_glove=model.fit(X_train,y_train,batch_size=4,epochs=10,validation_data=(X_test,y_test),verbose=2)
train_pred_GloVe = model.predict(X_train)

train_pred_GloVe_int = train_pred_GloVe.round().astype('int')
sentiment= pd.read_csv('../input/ted-talks-transcripts-sentiment-score/url_and_positivity.csv')

sentiment= sentiment[['url', 'positive scores']]

sentiment['url']= sentiment['url'].map(lambda x: x.lstrip('+-').rstrip('\n'))
data_v2 = pd.merge(data,

                 sentiment,

                 on='url', 

                 how='left')

data_v2['positive scores'] = data_v2['positive scores'].fillna(0)

data_v2.head()
feature_columns_mod3=['wc_per_min','stopword_count',

                      '_NOUN',

 '_VERB',

 '_ADP',

 '_ADJ',

 '_DET',

 '_PROPN',

 '_INTJ',

 '_PUNCT',

 '_NUM',

 '_PRON',

 '_ADV',

 '_PART',

 '_amod',

 '_advmod',

 '_acl',

 '_relcl',

 '_advcl',

 '_neg',

 '_PERSON',

 '_NORP',

 '_FAC',

 '_ORG',

 '_GPE',

 '_LOC',

 '_PRODUCT',

 '_EVENT',

 '_WORK_OF_ART',

 '_LANGUAGE',

 'laughter_count',

 'applaud_count',

 'positive scores']



target= ['super_popular']



#list(data_v2[feature_columns_mod3])
data_mod3= data_v2[feature_columns_mod3+target]

X_train, X_test,y_train,y_test =train_test_split(

     data_mod3.drop(['super_popular'], axis = 1), data_mod3[['super_popular']], 

     test_size = 0.2, random_state = 100)
#create LGBM classifier model

gbm_model = lgb.LGBMClassifier(

        boosting_type= "dart",

        n_estimators=1850,

        learning_rate=0.12,

        num_leaves=35,

        colsample_bytree=.8,

        subsample=.9,

        max_depth=9,

        reg_alpha=.1,

        reg_lambda=.1,

        min_split_gain=.01

)


num_training_folds = 10

# get cross validated scores



cv_prediction_scores = cv_performance_assessment(X_train,y_train,num_training_folds,gbm_model)
# Compute and plot the ROC curves

plot_roc(y_train, cv_prediction_scores)

plt.title("ROC curve from 10-fold cross validation")
# predict probabilities on test set

y_pred_gbm = gbm_model.predict_proba(X_test)

plot_roc(y_test, y_pred_gbm[:,1])

plt.title("ROC curve on 20% test split")
predictions_lgbm_bin_mod3_test = np.where(y_pred_gbm[:,1] > 0.5, 1, 0) #Turn probability to 0-1 binary out

print("Test accuracy:",accuracy_score(y_test,predictions_lgbm_bin_mod3_test))
# fit model on complete data

gbm_model=gbm_model.fit(data_v2[feature_columns_mod3],data_v2[target],verbose=0)





# predict probabilities on complete set

y_pred_total_gbm1 = gbm_model.predict_proba(data_v2[feature_columns_mod3])

predictions_lgbm_bin_mod3 = np.where(y_pred_total_gbm1[:,1] > 0.5, 1, 0) #Turn probability to 0-1 binary out
lgb.plot_importance(gbm_model,max_num_features=15)
# saving both predictions to CSV files

submission_file1 = pd.DataFrame({'id':    data_v2['url'],

                                   'proba':  y_pred_total_gbm1[:,1],

                                   'preds': predictions_lgbm_bin_mod3,

                                   'actuals': data_v2['super_popular']})

submission_file1.to_csv('submission_sentiments_feats.csv',

                           columns=['id','proba','preds','actuals'],

                           index=False)
topic_models=pd.read_csv('../input/topic-models/transcript_topic.csv')

topic_models['url']= topic_models['url'].map(lambda x: x.lstrip('+-').rstrip('\n'))

topic_models
data_v3 = pd.merge(data_v2,

                 topic_models,

                 on='url', 

                 how='left')

data_v3['Dominant_Topic'] = data_v3['Dominant_Topic'].fillna(0)

data_v3['Dominant_Topic'] = data_v3['Dominant_Topic'].astype(int).astype(str)

data_v3.head()
# create a list of categorical columns for one hot encoding

cat_variables= ['Dominant_Topic']



# One-Hot encoding to convert categorical columns to numeric

print('start one-hot encoding')



data_v3 = pd.get_dummies(data_v3, prefix = cat_variables,

                         columns = cat_variables)



print('one-hot encoding done')

feature_columns_mod4=['url','wc_per_min','stopword_count',

                      '_NOUN',

 '_VERB',

 '_ADP',

 '_ADJ',

 '_DET',

 '_PROPN',

 '_INTJ',

 '_PUNCT',

 '_NUM',

 '_PRON',

 '_ADV',

 '_PART',

 '_amod',

 '_advmod',

 '_acl',

 '_relcl',

 '_advcl',

 '_neg',

 '_PERSON',

 '_NORP',

 '_FAC',

 '_ORG',

 '_GPE',

 '_LOC',

 '_PRODUCT',

 '_EVENT',

 '_WORK_OF_ART',

 '_LANGUAGE',

 'laughter_count',

 'applaud_count',

 'Dominant_Topic_0',

 'Dominant_Topic_1',

 'Dominant_Topic_2',

 'Dominant_Topic_3',

 'Dominant_Topic_4',

 'Dominant_Topic_5']

target= ['super_popular']





data_mod4= data_v3[feature_columns_mod4+target]

X_train, X_test,y_train,y_test =train_test_split(

     data_mod4.drop(['super_popular'], axis = 1), data_mod4[['super_popular']], 

     test_size = 0.2, random_state = 100)
X_train=X_train.drop(['url'], axis = 1)

X_test1=X_test.drop(['url'], axis = 1)
feature_columns_mod4=['wc_per_min','stopword_count',

                      '_NOUN',

 '_VERB',

 '_ADP',

 '_ADJ',

 '_DET',

 '_PROPN',

 '_INTJ',

 '_PUNCT',

 '_NUM',

 '_PRON',

 '_ADV',

 '_PART',

 '_amod',

 '_advmod',

 '_acl',

 '_relcl',

 '_advcl',

 '_neg',

 '_PERSON',

 '_NORP',

 '_FAC',

 '_ORG',

 '_GPE',

 '_LOC',

 '_PRODUCT',

 '_EVENT',

 '_WORK_OF_ART',

 '_LANGUAGE',

 'laughter_count',

 'applaud_count',

 'Dominant_Topic_0',

 'Dominant_Topic_1',

 'Dominant_Topic_2',

 'Dominant_Topic_3',

 'Dominant_Topic_4',

 'Dominant_Topic_5']
#create LGBM classifier model

gbm_model = lgb.LGBMClassifier(

        boosting_type= "dart",

        n_estimators=1850,

        learning_rate=0.12,

        num_leaves=35,

        colsample_bytree=.8,

        subsample=.9,

        max_depth=9,

        reg_alpha=.1,

        reg_lambda=.1,

        min_split_gain=.01

)



num_training_folds = 10

# get cross validated scores



cv_prediction_scores = cv_performance_assessment(X_train,y_train,num_training_folds,gbm_model)
# Compute and plot the ROC curves

plot_roc(y_train, cv_prediction_scores)

plt.title("ROC curve from 10-fold cross validation")
# predict probabilities on test set

y_pred_gbm = gbm_model.predict_proba(X_test1)

plot_roc(y_test, y_pred_gbm[:,1])

plt.title("ROC curve on 20% test split")
predictions_lgbm_bin2 = np.where(y_pred_gbm[:,1] > 0.5, 1, 0) #Turn probability to 0-1 binary out

print("Test accuracy:",accuracy_score(y_test,predictions_lgbm_bin2))
#Print Confusion Matrix

plt.figure()

cm = confusion_matrix(y_test,predictions_lgbm_bin2)

labels = ['Not Very Popular', 'Very Popular']

plt.figure(figsize=(8,6))

sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, fmt='d', cmap="Blues", vmin = 0.2);

plt.title('Confusion Matrix')

plt.ylabel('True Class')

plt.xlabel('Predicted Class')

plt.show()
prediction_file = pd.DataFrame({'id':    X_test['url'],

                                   'proba':  y_pred_gbm[:,1],

                                   'preds': predictions_lgbm_bin2,

                                   'actuals': y_test['super_popular']})

prediction_file.to_csv('prediction_topics_feats.csv',

                           columns=['id','proba','preds','actuals'],

                           index=False)
prediction_file
# fit model on complete data

gbm_model=gbm_model.fit(data_v3[feature_columns_mod4],data_v3[target],verbose=0)





# predict probabilities on complete set

y_pred_total_gbm2 = gbm_model.predict_proba(data_v3[feature_columns_mod4])

predictions_lgbm_bin_mod4 = np.where(y_pred_total_gbm2[:,1] > 0.5, 1, 0) #Turn probability to 0-1 binary out
lgb.plot_importance(gbm_model,max_num_features=15)
feature_imp_lgb=pd.DataFrame(list(zip(feature_columns_mod4,gbm_model.feature_importances_)))

column_names_lgb= ['features','LGB_imp']

feature_imp_lgb.columns= column_names_lgb



feature_imp_lgb= feature_imp_lgb.sort_values('LGB_imp',ascending=False)

feature_imp_lgb[:15]
ax = sns.barplot(x="LGB_imp", y="features", data=feature_imp_lgb[:15])

ax.set_ylabel("Features for training")

ax.set_xlabel("Feature Importance from LightGBM Model")

ax.set_title("Feature importance of top 15 important features")
# saving both predictions to CSV files

submission_file2 = pd.DataFrame({'id':    data_v3['url'],

                                   'proba':  y_pred_total_gbm2[:,1],

                                   'preds': predictions_lgbm_bin_mod4,

                                   'actuals': data_v3['super_popular']})

submission_file2.to_csv('submission_topics_feats.csv',

                           columns=['id','proba','preds','actuals'],

                           index=False)