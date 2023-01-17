import pandas as pd

from pandas import Series,DataFrame

import numpy as np
#data

#KimKardashianTweets data

df_kk=pd.read_csv('../input/KimKardashianTweets.csv')

len(df_kk)
#HillaryClintonTweets data

df_hc=pd.read_csv('../input/HillaryClintonTweets.csv')

len(df_hc)
author1='KimKardashian'

author2='HillaryClinton'
import random

#2000 random sample rows for KK

rows = random.sample(list(df_kk.index), 2000)

df_kk = df_kk.ix[rows]

#2000 random sample rows for HC

rows = random.sample(list(df_hc.index), 2000)

df_hc = df_hc.ix[rows]

#join back together

df=df_kk.append(df_hc,ignore_index=True)

len(df)
#data pre-processing

df.drop(df[df.retweet==True].index, inplace=True)

df['num_of_words'] = df["text"].str.split().apply(len)

df.drop(df[df.num_of_words<4].index, inplace=True)

df["text"].replace(r"http\S+", "URL", regex=True,inplace=True)

df["text"].replace(r"@\S+", "REF", regex=True ,inplace=True)

df["text"].replace(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})+", "DATE", regex=True,inplace=True)

df["text"].replace(r"(\d{1,2})[/:](\d{2})[/:](\d{2})?(am|pm)+", "TIME", regex=True,inplace=True)

df["text"].replace(r"(\d{1,2})[/:](\d{2})?(am|pm)+", "TIME", regex=True,inplace=True)

df["text"].replace(r"\d+", "NUM", regex=True,inplace=True)

len(df)
#POS tag 2-3 chars abbrivation mapping to 1 char abbrevations

#http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

pos_code_map={'CC':'A','CD':'B','DT':'C','EX':'D','FW':'E','IN':'F','JJ':'G','JJR':'H','JJS':'I','LS':'J','MD':'K','NN':'L','NNS':'M',

'NNP':'N','NNPS':'O','PDT':'P','POS':'Q','PRP':'R','PRP$':'S','RB':'T','RBR':'U','RBS':'V','RP':'W','SYM':'X','TO':'Y','UH':'Z',

'VB':'1','VBD':'2','VBG':'3','VBN':'4','VBP':'5','VBZ':'6','WDT':'7','WP':'8','WP$':'9','WRB':'@'}

#Python 2 code_pos_map={v: k for k, v in pos_code_map.iteritems()}

code_pos_map = {v: k for k, v in  pos_code_map.items()}
#abbrivation converters

def convert(tag):

    try:

        code=pos_code_map[tag]

    except:

        code='?'

    return code

def inv_convert(code):

    try:

        tag=code_pos_map[code]

    except:

        tag='?'

    return tag
#POS tag converting

import nltk

from nltk.tokenize import RegexpTokenizer

from nltk import pos_tag, word_tokenize

def pos_tags(text):

    tokenizer = RegexpTokenizer(r'\w+')

    text_processed=tokenizer.tokenize(text)

    return "".join(convert(tag) for (word, tag) in nltk.pos_tag(text_processed))

def text_pos_inv_convert(text):

    return "-".join(inv_convert(c.upper()) for c in text)
#a new column for pos tags

df['text_pos']=df.apply(lambda x: pos_tags(x['text']), axis=1)
df.ix[:,['author','text','text_pos']].head()
df_features=pd.DataFrame()
from sklearn.feature_extraction.text import CountVectorizer

for a in df.author.unique():

    v = CountVectorizer(analyzer='char',ngram_range=(3, 3))

    ngrams = v.fit_transform(df[df['author'] == a]['text_pos'])

    df_t=pd.DataFrame(

    {'Feature': v.get_feature_names(),

     'Count': list(ngrams.sum(axis=0).flat),

     'Author': a

    })

    #

    df_features=df_features.append(df_t,ignore_index=True)
df_features['Feature_POS']=df_features.apply(lambda x: text_pos_inv_convert(x['Feature']), axis=1)
df_features[~df_features.Feature.isin(df_features[df_features['Author'] != author2].Feature)].sort_values('Count', ascending=False).ix[:,['Author','Count','Feature','Feature_POS']].head()
df_features[~df_features.Feature.isin(df_features[df_features['Author'] != author1].Feature)].sort_values('Count', ascending=False).ix[:,['Author','Count','Feature','Feature_POS']].head()
from sklearn.model_selection import train_test_split

twt_train, twt_test, author_train, author_test = train_test_split(df.ix[:,['text','text_pos']], df['author'], test_size=0.4, random_state=42)
from nltk.tokenize import RegexpTokenizer

from nltk.stem.porter import PorterStemmer

def text_process(text):

    """

    Takes in a string of text, then performs the following:

    1. Tokenizes and removes punctuation

    3. Stems

    4. Returns a list of the cleaned text

    """



    # tokenizing

    tokenizer = RegexpTokenizer(r'\w+')

    text_processed=tokenizer.tokenize(text)

    

    

    # steming

    porter_stemmer = PorterStemmer()

    

    text_processed = [porter_stemmer.stem(word) for word in text_processed]

    



    return text_processed
ScoreSummaryByModelParams = list()
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import FeatureUnion

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):

        self.key = key



    def fit(self, x, y=None):

        return self



    def transform(self, data_dict):

        return data_dict[self.key]
class TextAndTextCodedExtractor(BaseEstimator, TransformerMixin):

    """Extract the text & text_pos from a tweet in a single pass.

    """

    def fit(self, x, y=None):

        return self



    def transform(self, tweets):

        features=tweets.ix[:,['text_pos','text']].to_records(index=False)



        return features
def ModelParamsEvaluation (f_union,model,params,comment):

    pipeline = Pipeline([

    # Extract the text & text_coded

    ('textandtextcoded', TextAndTextCodedExtractor()),



    # Use FeatureUnion to combine the features from text and text_coded

    ('union', f_union, ),



    # Use a  classifier on the combined features

    ('clf', model),

    ])

    grid_search = GridSearchCV(estimator=pipeline, param_grid=params, verbose=1, cv=5)

    grid_search.fit(twt_train, author_train)

    #best score

    print("Best score: %0.3f" % grid_search.best_score_)

    print("Best parameters set:")

    best_parameters = grid_search.best_estimator_.get_params()

    for param_name in sorted(params.keys()):

        print("\t%s: %r" % (param_name, best_parameters[param_name]))

        ScoreSummaryByModelParams.append([comment,grid_search.best_score_,"\t%s: %r" % (param_name, best_parameters[param_name])])    

 
f1_union=FeatureUnion(

        transformer_list=[

              # Pipeline for pulling char features  from the text

            ('char', Pipeline([

                ('selector', ItemSelector(key='text')),

                ('tfidf',     TfidfVectorizer(analyzer='char')),

            ])),               



        ],

    )
from sklearn.naive_bayes import BernoulliNB

p = {

    'union__char__tfidf__max_df': (0.5, 0.75, 1.0),

    'union__char__tfidf__ngram_range': ((2, 2), (3, 3)), 

    'clf__alpha': (1,0.1,0.01,0.001,0.0001,0)}



ModelParamsEvaluation(f1_union,BernoulliNB(),p,'Bernoulli Naive Bayes, char')
f1_union=FeatureUnion(

        transformer_list=[

            # Pipeline for pulling word features from the text

            ('word', Pipeline([

            ('selector', ItemSelector(key='text')),

            ('tfidf',    TfidfVectorizer(analyzer='word')),

            ])),              



        ],

    )
p = {

    'union__word__tfidf__max_df': (0.5, 0.75, 1.0),

    'union__word__tfidf__ngram_range': ((1, 1),(2, 2), (3, 3)), 

    'union__word__tfidf__stop_words': (None, 'english'),

    'clf__alpha': (1,0.1,0.01,0.001,0.0001,0)}



ModelParamsEvaluation(f1_union,BernoulliNB(),p,'Bernoulli Naive Bayes, word')
f1_union=FeatureUnion(

        transformer_list=[

            # Pipeline for pulling word features from the text

            ('text', Pipeline([

            ('selector', ItemSelector(key='text')),

            ('tfidf',    TfidfVectorizer(analyzer='word',tokenizer= text_process)),

            ])),              



        ],

    )
p = {

    'union__text__tfidf__max_df': (0.5, 0.75, 1.0),

    'union__text__tfidf__ngram_range': ((1, 1),(2, 2), (3, 3)), 

    'clf__alpha': (1,0.1,0.01,0.001,0.0001,0)}



ModelParamsEvaluation(f1_union,BernoulliNB(),p,'Bernoulli Naive Bayes, stemmed words, no stop words')
f1_union=FeatureUnion(

        transformer_list=[

            # Pipeline for pulling pos tag features  from the text_pos

            ('text_pos', Pipeline([

            ('selector', ItemSelector(key='text_pos')),

            ('tfidf',    TfidfVectorizer(analyzer='char',ngram_range=(3, 3),max_df=0.5,max_features=None)),

            ])),                  



        ],

    )
p = {

    'clf__alpha': (1,0.1,0.01,0.001,0.0001,0)}



ModelParamsEvaluation(f1_union,BernoulliNB(),p,'Bernoulli Naive Bayes, POS tags')
df_ScoreSummaryByModelParams=DataFrame(ScoreSummaryByModelParams,columns=['Method','BestScore','BestParameter'])

df_ScoreSummaryByModelParams.sort_values(['BestScore'],ascending=False,inplace=True)

df_ScoreSummaryByModelParams
f3_union=FeatureUnion(

        transformer_list=[

             # Pipeline for pulling word stemmed features from the text

            ('text', Pipeline([

                ('selector', ItemSelector(key='text')),

                ('tfidf',     TfidfVectorizer(analyzer='word',tokenizer= text_process,ngram_range=(1, 1),max_df=0.5,max_features=None)),

            ])),

                    

            # Pipeline for pulling char features  from the text

            ('char', Pipeline([

                ('selector', ItemSelector(key='text')),

                ('tfidf',     TfidfVectorizer(analyzer='char',ngram_range=(3, 3),max_df=0.5,max_features=5000)),

            ])),

                    

            # Pipeline for pulling flexible pattern features  from the text_coded with POS tags

            ('text_pos', Pipeline([

                ('selector', ItemSelector(key='text_pos')),

                ('tfidf',    TfidfVectorizer(analyzer='char',ngram_range=(3, 3),max_df=0.5,max_features=None)),

            ])),                  



        ],



    )
p = {'clf__alpha': (1,0.1,0.01,0.001,0.0001,0)}

ModelParamsEvaluation(f3_union,BernoulliNB(),p,'Bernoulli Naive Bayes, char + stemmed word + POS tags')
df_ScoreSummaryByModelParams=DataFrame(ScoreSummaryByModelParams,columns=['Method','BestScore','BestParameter'])

df_ScoreSummaryByModelParams.sort_values(['BestScore'],ascending=False,inplace=True)

df_ScoreSummaryByModelParams
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve, auc,precision_score, accuracy_score, recall_score, f1_score

from scipy import interp
ScoreSummaryByVector = list()
def PredictionEvaluation(author_test_b,author_predictions_b,comment):

    Precision=precision_score(author_test_b,author_predictions_b)

    print ('Precision: %0.3f' % (Precision))

    Accuracy=accuracy_score(author_test_b,author_predictions_b)

    print ('Accuracy: %0.3f' % (Accuracy))

    Recall=recall_score(author_test_b,author_predictions_b)

    print ('Recall: %0.3f' % (Recall))

    F1=f1_score(author_test_b,author_predictions_b)

    print ('F1: %0.3f' % (F1))

    print ('Confussion matrix:')

    print (confusion_matrix(author_test_b,author_predictions_b))

    ROC_AUC=roc_auc_score(author_test_b,author_predictions_b)

    print ('ROC-AUC: %0.3f' % (ROC_AUC))

    ScoreSummaryByVector.append([Precision,Accuracy,Recall,F1,ROC_AUC,comment])
def ModelRun (f_union,model):

    pipeline = Pipeline([

    # Extract the text & text_coded

    ('textandtextcoded', TextAndTextCodedExtractor()),



    # Use FeatureUnion to combine the features from text and text_coded

    ('union', f_union, ),



    # Use a  classifier on the combined features

    ('clf', model),

    ])

    pipeline.fit(twt_train, author_train)

    author_predicted = pipeline.predict(twt_test)

    

    feature_names=list()

    for p in (pipeline.get_params()['union'].transformer_list):

        fn=(p[0],pipeline.get_params()['union'].get_params()[p[0]].get_params()['tfidf'].get_feature_names())

        feature_names.append(fn)

    df_fn=pd.DataFrame()

    for fn in feature_names:

        df_fn= df_fn.append(pd.DataFrame(

        {'FeatureType': fn[0],

         'Feature': fn[1]

        }),

        ignore_index=True)    

    

    from sklearn.preprocessing import LabelBinarizer

    lb = LabelBinarizer()

    author_test_b = lb.fit_transform(author_test.values)

    author_predicted_b  = lb.fit_transform(author_predicted)

    return (df_fn,pipeline.get_params()['clf'],author_predicted,author_predicted_b, author_test_b)
def most_informative_feature_for_binary_classification(feature_names, classifier):

    class_labels = classifier.classes_



    topnvalues_class0 = sorted(zip(classifier.coef_[0], feature_names['Feature'].values, feature_names['FeatureType'].values))

    topnvalues_class1 = sorted(zip(classifier.coef_[0], feature_names['Feature'].values, feature_names['FeatureType'].values), reverse=True)



    topn_df_class0=pd.DataFrame(topnvalues_class0, columns=['Coef','Feature','FeatureType'])

    topn_df_class0['Author']=class_labels[0]

    

    topn_df_class1=pd.DataFrame(topnvalues_class1, columns=['Coef','Feature','FeatureType'])

    topn_df_class1['Author']=class_labels[1]    

    

    topn_df=topn_df_class0.append(topn_df_class1)

    

        

    return topn_df
f2_union=FeatureUnion(

        transformer_list=[

            # Pipeline for pulling char features  from the text

            ('char', Pipeline([

                ('selector', ItemSelector(key='text')),

                ('tfidf',     TfidfVectorizer(analyzer='char',ngram_range=(3, 3),max_df=0.5,max_features=5000)),

            ])),

            # Pipeline for pulling stememd word features from the text

            ('text', Pipeline([

                ('selector', ItemSelector(key='text')),

                ('tfidf',    TfidfVectorizer(analyzer='word',tokenizer= text_process,ngram_range=(1, 1),max_df=0.5,max_features=None)),

            ])),        



        ],



    )
(feature_names,clf,author_predicted,author_predicted_b, author_test_b)=ModelRun(f2_union,BernoulliNB(alpha=0.0001))
PredictionEvaluation(author_predicted_b, author_test_b,'char+stemmed word')
f3_union=FeatureUnion(

        transformer_list=[

             # Pipeline for pulling word stemmed features from the text

            ('text', Pipeline([

                ('selector', ItemSelector(key='text')),

                ('tfidf',     TfidfVectorizer(analyzer='word',tokenizer= text_process,ngram_range=(1, 1),max_df=0.5,max_features=None)),

            ])),

                    

            # Pipeline for pulling char features  from the text

            ('char', Pipeline([

                ('selector', ItemSelector(key='text')),

                ('tfidf',     TfidfVectorizer(analyzer='char',ngram_range=(3, 3),max_df=0.5,max_features=5000)),

            ])),

                    

            # Pipeline for pulling flexible pattern features  from the text_coded with POS tags

            ('text_pos', Pipeline([

                ('selector', ItemSelector(key='text_pos')),

                ('tfidf',    TfidfVectorizer(analyzer='char',ngram_range=(3, 3),max_df=0.5,max_features=None)),

            ])),                  



        ],



    )
(feature_names,clf,author_predicted,author_predicted_b, author_test_b)=ModelRun(f3_union,BernoulliNB(alpha=0.0001))
PredictionEvaluation(author_predicted_b, author_test_b,'char+stemmed word+POS tag')
f4_union=FeatureUnion(

        transformer_list=[



            # Pipeline for pulling word features from the text

            ('word', Pipeline([

                ('selector', ItemSelector(key='text')),

                ('tfidf',    TfidfVectorizer(analyzer='word',stop_words=None,ngram_range=(1, 1),max_df=0.5,max_features=None)),

            ])),

                    

             # Pipeline for pulling word features after word_processing from the text

            ('text', Pipeline([

                ('selector', ItemSelector(key='text')),

                ('tfidf',     TfidfVectorizer(analyzer='word',tokenizer= text_process,ngram_range=(1, 1),max_df=0.5,max_features=None)),

            ])),

                    

            # Pipeline for pulling char features  from the text

            ('char', Pipeline([

                ('selector', ItemSelector(key='text')),

                ('tfidf',     TfidfVectorizer(analyzer='char',ngram_range=(3, 3),max_df=0.5,max_features=5000)),

            ])),

                    

            # Pipeline for pulling flexible pattern features  from the text_coded

            ('text_pos', Pipeline([

                ('selector', ItemSelector(key='text_pos')),

                ('tfidf',    TfidfVectorizer(analyzer='char',ngram_range=(3, 3),max_df=0.5,max_features=None)),

            ])),                  



        ],



    )
(feature_names,clf,author_predicted,author_predicted_b, author_test_b)=ModelRun(f4_union,BernoulliNB(alpha=0.0001))
PredictionEvaluation(author_predicted_b, author_test_b,'char+word+stemmed word+POS tag')
df_ScoreSummaryByVector=DataFrame(ScoreSummaryByVector,columns=['Precision','Accuracy','Recall','F1','ROC-AUC','Vector'])

df_ScoreSummaryByVector.sort_values(['F1'],ascending=False,inplace=True)

df_ScoreSummaryByVector
TopFeatures_df=most_informative_feature_for_binary_classification(feature_names, clf)
df1=TopFeatures_df.loc[((TopFeatures_df['Author']==author2) & (TopFeatures_df['FeatureType']=='char')),['Author','Coef','Feature']].head(10)

df1.rename(columns={'Coef':'CoefChar','Feature':'Char'}, inplace=True)

df1.reset_index(inplace=True)

df2=TopFeatures_df.loc[((TopFeatures_df['Author']==author2) & (TopFeatures_df['FeatureType']=='word')),['Coef','Feature']].head(10)

df2.rename(columns={'Coef':'CoefWord','Feature':'Word'}, inplace=True)

df2.reset_index(inplace=True)

df3=TopFeatures_df.loc[((TopFeatures_df['Author']==author2) & (TopFeatures_df['FeatureType']=='text')),['Coef','Feature']].head(10)

df3.rename(columns={'Coef':'CoefText','Feature':'Text'}, inplace=True)

df3.reset_index(inplace=True)

df4=TopFeatures_df.loc[((TopFeatures_df['Author']==author2) & (TopFeatures_df['FeatureType']=='text_pos')),['Coef','Feature']].head(10)

df4.rename(columns={'Coef':'CoefTextPOS','Feature':'TextPOS'}, inplace=True)

df4['TextPOS']=df4.apply(lambda x: text_pos_inv_convert(x['TextPOS']), axis=1)

df4.reset_index(inplace=True)

df_kk_top_features = pd.concat([df1,df2,df3,df4],axis=1)

df_kk_top_features.drop('index', axis=1, inplace=True)

df_kk_top_features
df1=TopFeatures_df.loc[((TopFeatures_df['Author']==author1) & (TopFeatures_df['FeatureType']=='char')),['Author','Coef','Feature']].head(10)

df1.rename(columns={'Coef':'CoefChar','Feature':'Char'}, inplace=True)

df1.reset_index(inplace=True)

df2=TopFeatures_df.loc[((TopFeatures_df['Author']==author1) & (TopFeatures_df['FeatureType']=='word')),['Coef','Feature']].head(10)

df2.rename(columns={'Coef':'CoefWord','Feature':'Word'}, inplace=True)

df2.reset_index(inplace=True)

df3=TopFeatures_df.loc[((TopFeatures_df['Author']==author1) & (TopFeatures_df['FeatureType']=='text')),['Coef','Feature']].head(10)

df3.rename(columns={'Coef':'CoefText','Feature':'Text'}, inplace=True)

df3.reset_index(inplace=True)

df4=TopFeatures_df.loc[((TopFeatures_df['Author']==author1) & (TopFeatures_df['FeatureType']=='text_pos')),['Coef','Feature']].head(10)

df4.rename(columns={'Coef':'CoefTextPOS','Feature':'TextPOS'}, inplace=True)

df4['TextPOS']=df4.apply(lambda x: text_pos_inv_convert(x['TextPOS']), axis=1)

df4.reset_index(inplace=True)

df_kk_top_features = pd.concat([df1,df2,df3,df4],axis=1)

df_kk_top_features.drop('index', axis=1, inplace=True)

df_kk_top_features
author_predicted=pd.DataFrame(author_predicted,columns=['predicted'])

df_wrong_result = pd.concat([twt_test.reset_index(),author_test.reset_index(),author_predicted], axis=1)

df_wrong_result.drop('index', axis=1, inplace=True)

df_wrong_result.drop('text_pos', axis=1, inplace=True)

df_wrong_result=df_wrong_result[df_wrong_result['author']!=df_wrong_result['predicted']]

df_wrong_result.head(10)