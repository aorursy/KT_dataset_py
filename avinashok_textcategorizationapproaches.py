from __future__ import unicode_literals



%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('bmh')



import warnings

warnings.filterwarnings('ignore')



import pandas as pd

from collections import Counter

from textblob import TextBlob

import re

import nltk

import xgboost as xgb

import numpy as np

import collections

import operator

import itertools

import os





from nltk.corpus import stopwords

import nltk.stem.snowball

st = nltk.stem.snowball.SnowballStemmer('english')

from nltk.stem import WordNetLemmatizer, SnowballStemmer

stemmer = SnowballStemmer('english')



from sklearn.model_selection import train_test_split, ShuffleSplit, learning_curve

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from sklearn.metrics import classification_report, accuracy_score ,confusion_matrix

from sklearn.linear_model import SGDClassifier

from sklearn import preprocessing

from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from sklearn.svm import SVC

from sklearn import utils



from yellowbrick.text import FreqDistVisualizer



import gensim

from gensim.models import Word2Vec

from gensim.models import Doc2Vec

from gensim.models import Phrases

from gensim.models.phrases import Phraser

from gensim import models

from gensim.models.doc2vec import TaggedDocument

import pyLDAvis

import pyLDAvis.gensim as gensimvis



from itertools import islice



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation, Dropout

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing import text, sequence

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras import utils



from tqdm import tqdm

tqdm.pandas(desc="progress-bar")



import spacy

from spacy_langdetect import LanguageDetector

nlp = spacy.load('en')



from wordcloud import WordCloud 



import chart_studio.plotly.plotly as py

import plotly.graph_objs as go

import chart_studio



## Offline mode

from plotly.offline import init_notebook_mode, iplot

chart_studio.tools.set_credentials_file(username='avinashok', api_key='SZBL9GagG9yPYCwfSuDc')

init_notebook_mode(connected=True)
#Directories

dataDirectory = "../input/"

# reading data file  

df = pd.read_csv(dataDirectory+'telecomagentcustomerinteractiontext/'+'CustomerInteractionData.csv')



#Randomly shuffling data

df = df.sample(len(df))



#Copying the Raw comment column for Future Use

df['CustomerInteractionText'] = df['CustomerInteractionRawText']



#Checking the initial 5 rows

df.head(5)
#Objects Used

commentTextColumn = 'CustomerInteractionText'

agentAssignedColumn = 'AgentAssignedTopic'

locationID = 'LocationID'

callDuration = 'CallDurationSeconds' #In Seconds

agentID = 'AgentID'

customerID = 'CustomerID'

rawText = 'CustomerInteractionRawText'
# Identifying Primary Key



print("Number of Columns in the Dataset = " + str(len(df.columns)))



uniqueColumns=[]

for col in df.columns.to_list():

    if len(df[str(col)].unique())==df.shape[0]:

        uniqueColumns.append(col)

if len(uniqueColumns)==1:

    primaryKeyColumn = str(uniqueColumns[0])

    print("Primary Key = "+primaryKeyColumn)
# Checking Datatypes of each columns in the dataset

print(df.dtypes)



# Specifying datatypes we want for each column

dataTypeDictionary = {

    primaryKeyColumn: 'int64',

    commentTextColumn: 'object',

    rawText:'object',

    agentAssignedColumn: 'object',             

    locationID: 'int64',

    callDuration: 'int64',

    agentID: 'object',

    customerID: 'object'

    

 }
duplicatesCount = {}

for col in df.columns.to_list():

    duplicatesCount[col] = [((df.duplicated(col).sum()/len(df))*100),100-((df.duplicated(col).sum()/len(df))*100)]



nullCounter = {}

for col in df.columns.to_list():

    count = 0

    for cell in df[str(col)]:

        if cell=='?' or cell=="":   # or len(str(cell))==1

            count= count +1

    nullCounter[col]=[float(count/len(df))*100,100-float(count/len(df))*100]
def dataQualityCheck(checkName, columnName):

    if checkName == "Null Values":

        # create data

        names='Null Values', 'Non Null Values',

        size=np.array(nullCounter[columnName])

        print("Null Values Data Quality Check for "+str(columnName))

        def absolute_value(val):

            a  = np.round(val/100.*size.sum(), 0)

            return a

        # Create a circle for the center of the plot

        my_circle=plt.Circle( (0,0), 0.7, color='white')



        # Custom colors --> colors will cycle

        plt.pie(size, labels=names, colors=['red','green'],autopct=absolute_value)

        p=plt.gcf()

        p.gca().add_artist(my_circle)

        plt.show();

    elif checkName =="Duplicates": 

                # create data

        names='Duplicate Values', 'Unique Values Values',

        size=np.array(duplicatesCount[columnName])

        print("Duplicate Value Data Quality check for "+str(columnName))

        def absolute_value(val):

            a  = np.round(val/100.*size.sum(), 0)

            return a

        # Create a circle for the center of the plot

        my_circle=plt.Circle( (0,0), 0.7, color='white')



        # Custom colors --> colors will cycle

        plt.pie(size, labels=names, colors=['red','green'],autopct=absolute_value)

        p=plt.gcf()

        p.gca().add_artist(my_circle)

        plt.show();

    elif checkName == "Details":

        print("Details of the Column: \n ")

        print("Original Datatype should be "+dataTypeDictionary[columnName]+"\n")

        print("Datatype in the data is "+str(df[str(columnName)].dtypes)+"\n")

    elif checkName == "Range":

        if str(df[str(columnName)].dtypes)=='int64' or str(df[str(columnName)].dtypes)=='datetime64[ns]':

            print("Maximum Value is "+str(df[str(columnName)].max())+" \n ")

            print("Minimum Value is "+str(df[str(columnName)].min()))

        else:

            print("Since the Datatype of column "+str(columnName)+" is not numeric in the given data, Range cannot be calculated.")

    

    

def dQexecute(columnName):

    print("\n Name of the Column "+str(columnName)+"\n \n")

    dataQualityCheck("Details",columnName)

    dataQualityCheck("Null Values",columnName)

    dataQualityCheck("Duplicates",columnName)

    dataQualityCheck("Range",columnName)

    print("*****************")
for col in df.columns.to_list():

    dQexecute(col)
z = {}

uniqueTopics = list(df[agentAssignedColumn].unique())

for i in uniqueTopics:

    z[i]=i



data = [go.Bar(x = df[agentAssignedColumn].map(z).unique(),y = df[agentAssignedColumn].value_counts().values,

        marker= dict(colorscale='Jet',color = df[agentAssignedColumn].value_counts().values),text='Number of Calls for this reason')]



layout = go.Layout(title='Reasonwise Call Distribution')



fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename='TelecomCallDistribution')
df['totalwords'] = df[commentTextColumn].str.split().str.len()

def reasonCodeLevelWordCount(reasonCode, parameter):

    dfReasonCodeSubset = df[df[agentAssignedColumn]==reasonCode]

    if parameter == 'mean':

        return float(dfReasonCodeSubset.describe()['totalwords'][1])

    elif parameter == 'median':

        return float(dfReasonCodeSubset.describe()['totalwords'][5])
# Mean Word Count

reasonCodeDict = {}

for topic in uniqueTopics:

    reasonCodeDict[str(topic)]=float(reasonCodeLevelWordCount(topic, 'mean'))

plt.figure(figsize=(20,20))

plt.title("Mean Word Frequency for each Topic", fontdict=None, loc='center')

plt.bar(reasonCodeDict.keys(), reasonCodeDict.values(), width = 0.1  , color='g')

plt.show()



print("\n\n ******************** \n\n ")



#Median Word Count (Optional)

reasonCodeDict = {}

for topic in uniqueTopics:

    reasonCodeDict[str(topic)]=float(reasonCodeLevelWordCount(topic, 'median'))

plt.figure(figsize=(20,20))

plt.title("Median Word Frequency for each Topic", fontdict=None, loc='center')

plt.bar(reasonCodeDict.keys(), reasonCodeDict.values(), width = 0.1  , color='g')

plt.show()
vectorizer = CountVectorizer()

docs       = vectorizer.fit_transform(df[commentTextColumn])

features   = vectorizer.get_feature_names()

plt.figure(figsize=(12,8))

visualizer = FreqDistVisualizer(features=features)

visualizer.fit(docs)

for label in visualizer.ax.texts:

    label.set_size(20)

visualizer.poof()
%%time



abbrevationDictionary = {'Cus': 'customer', 'cus': 'customer',

                        'Xferred':'transferred', 'xferred': 'transferred'} 



#Function to Standardize Text

def objectStandardization(input_text):

    words = str(input_text).split() 

    new_words = [] 

    for word in words:

        word = re.sub('[^A-Za-z0-9\s]+', ' ', word) #remove special characters

        if word.lower() in abbrevationDictionary:

            word = abbrevationDictionary [word.lower()]

        new_words.append(word) 

    new_text = " ".join(new_words) 

    return new_text



df[commentTextColumn] = df[commentTextColumn].apply(objectStandardization)



print(df[commentTextColumn].head(5))
%%time



# Function to extract Names of persons, organizations, locations, products etc. from the dataset

def entityCollector(df):

    listOfNames = []

    for index, row in df.iterrows():

        doc = nlp(row[str(commentTextColumn)])

        fil = [(i.label_.lower(), i) for i in doc.ents if i.label_.lower() in ["person", "gpe", "product"]] # Extracts Person Names, Organization Names, Location, Product names

        if fil:

            listOfNames.append(fil)

        else:

            continue

    flat_list = [item for sublist in listOfNames for item in sublist]

    entityDict = {}

    for a, b in list(set(flat_list)): 

        entityDict.setdefault(a, []).append(b)

    return entityDict



entityDict = entityCollector(df)



print("\n Types of entities present in the data are: "+", ".join(list(entityDict.keys()))+" \n")

%%time

for entity in list(entityDict.keys()):

    entityDict[entity] = [str(i) for i in entityDict[entity]]



ignoreWords = []

for key in entityDict.keys():

    ignoreWords.append(entityDict[key])

ignoreWords = [item for sublist in ignoreWords for item in sublist]



print("Number of words in Custom Stopword list = "+str(len(ignoreWords)))
%%time

def languageDistribution(df):

    nlp = spacy.load("en")

    nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)

    df['language']=''

    language = []

    for index, row in df.iterrows():

        text = row[str(commentTextColumn)]

        doc = nlp(text)

        language.append(str(doc._.language['language']))

    df['language'] = language

    return df



df = languageDistribution(df)

langDict = df.groupby('language')[str(primaryKeyColumn)].nunique().to_dict()



otherLanguagesList = list(langDict.keys()).remove('en')

print("Some sample other language texts: \n")

for lang in list(langDict.keys()):

    print(str(df[df['language']==str(lang)].values.tolist()[0]))



##Plot a Pie Distribution of Language Distribution

bin_percent = pd.DataFrame(df['language'].value_counts(normalize=True) * 100)

plot = bin_percent.plot.pie(y='language', figsize=(10, 10), autopct='%1.1f%%')



#Dropping only the row with Spanish text

df = df.drop(df[df['language'] == 'es'].index);
%%time

#Function to extract Alpha numeric words

def alphanumericExtractor(input_text):

    words = str(input_text).split()

    alphanumericWordlist = []

    for word in words:

        word = re.sub('[^A-Za-z0-9\s]+', '', word.lower()) #remove special characters

        word = re.sub(r'[^\x00-\x7F]+',' ', word) # remove ascii

        if not word.isdigit() and any(ch.isdigit() for ch in word):

            alphanumericWordlist.append(word)

        else:

            continue

    return alphanumericWordlist



#Function to get the frequency of Alphanumeric words in the data

def alphanumericFrequency(df, commentTextColumnName):

    alphanumericWordsList = []

    for index, row in df.iterrows():

        if alphanumericExtractor(row[str(commentTextColumnName)]):

            alphanumericWordsList.append(alphanumericExtractor(row[str(commentTextColumnName)]))

        else:

            continue

    flat_list = [item for sublist in alphanumericWordsList for item in sublist]

    counts = Counter(flat_list)

    countsdict = dict(counts)

    return countsdict



# Final list of alphanumeric words

alphanumericWordFreqDict = alphanumericFrequency(df, commentTextColumn)

    

# To plot the distribution

totalWordcount = len(alphanumericWordFreqDict)



topWordCount = input('How many top words do you want? maximum= '+str(totalWordcount)+ ' \n ')

# topWordCount = totalWordcount



alphanumericWordFreqDictTop = dict(sorted(alphanumericWordFreqDict.items(), key=operator.itemgetter(1), reverse=True)[:int(topWordCount)])

print(alphanumericWordFreqDictTop)



plt.figure(figsize=(20,20))

plt.title('Frequency of AlphaNumeric Words in the Dataset', fontdict=None, loc='center')

plt.bar(alphanumericWordFreqDictTop.keys(), alphanumericWordFreqDictTop.values(), width = 0.1  , color='b');

plt.show();



#Updating Custom stopword list with Alphanumeric words

ignoreWords = ignoreWords + list(alphanumericWordFreqDict.keys())
def clean_text(newDesc):

    newDesc = re.sub('[^A-Za-z\s]+', '', newDesc) #remove special characters

    newDesc = re.sub(r'[^\x00-\x7F]+','', newDesc) # remove ascii

    newDesc = ' '.join( [w for w in newDesc.split() if len(w)>1] )  

    newDesc = newDesc.split()

    cleanDesc = [str(w) for w in newDesc if w not in ignoreWords] #remove entity names, alphanumeric words

    return ' '.join(cleanDesc)



df[commentTextColumn] = df[commentTextColumn].apply(clean_text)

df.head()
%%time



def textAutocorrect(df, columnName):

    df[str(columnName)] = df[str(columnName)].apply(lambda txt: ''.join(TextBlob(txt).correct()))

    return True



textAutocorrect(df, commentTextColumn)
stops = nlp.Defaults.stop_words

default_stopwords = stopwords.words('english')

customStopWords = {'PRON', 'pron'}

stops.update(set(default_stopwords))

stops.update(set(customStopWords))



def normalize(comment, lowercase, remove_stopwords):

    if lowercase:

        comment = comment.lower()

    comment = nlp(comment)

    lemmatized = list()

    for word in comment:

        lemma = word.lemma_.strip()

        if lemma:

            if not remove_stopwords or (remove_stopwords and lemma not in stops):

                lemmatized.append(lemma)

    normalizedSentence = " ".join(lemmatized)

    normalizedSentence = re.sub('[^A-Za-z\s]+', '', normalizedSentence)  # remove special characters

    normalizedSentence = normalizedSentence.split()

    cleanDesc = [str(w) for w in normalizedSentence if w not in stops] #remove PRON

    return " ".join(cleanDesc)



df[commentTextColumn] = df[commentTextColumn].apply(normalize, lowercase=True, remove_stopwords=True)

df.head()
# Removing Null Comments

def removeNullValueCommentText(df, columnName):

    initialLength = len(df)

    df = df[pd.notnull(df[columnName])]

    finalLength = len(df)

    print("\n Number of rows with Null Value in the column '"+str(columnName)+"' are: "+str(initialLength-finalLength))

    return df

df = removeNullValueCommentText(df, commentTextColumn)

print(len(df))
# Removing duplicate comments keeping the first one

def removeDuplicateComments(df, columnName, agentAssignedColumn):

    initialDf = df.copy()

    initialLength = len(initialDf)

    finalDf = df.drop_duplicates(subset=[columnName], keep='first')

    finalLength = len(finalDf)

    print("\n Number of rows with duplicate comments in the column '"+str(columnName)+"' are: "+str(initialLength-finalLength))

    print("\n The Level 3 Reason Codes for the dropped rows are given below: \n")

    droppedDF = initialDf[~initialDf.apply(tuple,1).isin(finalDf.apply(tuple,1))]

    print(droppedDF[agentAssignedColumn].value_counts())

    return finalDf,droppedDF



df,droppedDF = removeDuplicateComments(df, commentTextColumn, agentAssignedColumn)

print(len(df))
# Removing comments with just one word. (Like #CALL?)

def removingShortComments(df, columnName, agentAssignedColumn, numberOfWords = 1):

    initialDf = df.copy()

    initialLength = len(initialDf)

    finalDf = df[~(df[str(columnName)].str.split().str.len()<(int(numberOfWords)+1))]

    finalLength = len(finalDf)

    print("\n Number of rows with short comments in the column '"+str(columnName)+"' are: "+str(initialLength-finalLength))

    print("\n The Level 3 Reason Codes for the dropped rows are given below: \n")

    droppedDF = initialDf[~initialDf.apply(tuple,1).isin(finalDf.apply(tuple,1))]

    print(droppedDF[agentAssignedColumn].value_counts())

    return finalDf,droppedDF



df,droppedDF = removingShortComments(df, commentTextColumn, agentAssignedColumn)

print(len(df))
vectorizer = CountVectorizer()

docs       = vectorizer.fit_transform(df[commentTextColumn])

features   = vectorizer.get_feature_names()

plt.figure(figsize=(12,8))

visualizer = FreqDistVisualizer(features=features)

visualizer.fit(docs)

for label in visualizer.ax.texts:

    label.set_size(20)

visualizer.poof()
def wordFrequency(reasonCode):

    return (df[df[agentAssignedColumn]==str(reasonCode)][commentTextColumn].str.split(expand=True).stack().value_counts())





def wordFrequencyListPlot(reasonCode, plot = False):

    wordFreqDict = df[df[agentAssignedColumn]==str(reasonCode)][commentTextColumn].str.split(expand=True).stack().value_counts().to_dict()

    wordFreqDictMostCommon = dict(collections.Counter(wordFreqDict).most_common(10)) #Considering only Top 10 words

    print(list(wordFreqDictMostCommon.keys()))

    if plot == True:

        plt.title(str(reasonCode), fontdict=None, loc='center')

        plt.bar(wordFreqDictMostCommon.keys(), wordFreqDictMostCommon.values(), width = 0.1  , color='b');

        plt.figure(figsize=(10,10))

        plt.show();

    return list(wordFreqDictMostCommon.keys())
for reasoncode in uniqueTopics:

    print(reasoncode)

    wordFrequencyListPlot(reasoncode, plot = True)
def wordCloudGenerator(df, reasonCode, save = False):

    dfReasonCodeSubset = df[df[agentAssignedColumn]==reasonCode]

    wordcloud = WordCloud(max_words=50,background_color='white',max_font_size = 50,width=100, height=100).generate(' '.join(dfReasonCodeSubset[commentTextColumn]))

    plt.imshow(wordcloud)

    plt.title(str(reasonCode), fontdict=None, loc='center')

    plt.figure(figsize=(50,50))

    plt.axis("off")

    plt.show();

    if save:

        plt.savefig('wordCloud'+str(reasonCode)+'.png', bbox_inches='tight')

    
for topic in uniqueTopics:

    wordCloudGenerator(df, topic) #,save = True , if you want to save the Word Clouds
lbl_enc = preprocessing.LabelEncoder()

y = lbl_enc.fit_transform(df[agentAssignedColumn].values)

X_train, X_test, y_train, y_test = train_test_split(df[commentTextColumn].values, y,stratify=y,random_state=42, test_size=0.1)
tfidf = TfidfVectorizer(strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 3),use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = 'english')

# Fit and transform Tf-idf to both training and test sets

tfidf.fit(list(X_train) + list(X_test))

X_train_tfidf =  tfidf.transform(X_train) 

X_test_tfidf = tfidf.transform(X_test)
countvec = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 3), stop_words = 'english', binary=True)

# Fit and transform CountVectorizer to both training and test sets

countvec.fit(list(X_train) + list(X_test))

X_train_countvec =  countvec.transform(X_train) 

X_test_countvec = countvec.transform(X_test)
print("Total Number of Words in the column "+commentTextColumn+" is "+str(df[commentTextColumn].apply(lambda x: len(x.split(' '))).sum()))
X = df[commentTextColumn]

y = df[agentAssignedColumn]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 30)
%time

logreg = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', LogisticRegression(n_jobs=1, C=1e5)),])

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)



print('accuracy %s' % accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred,target_names=uniqueTopics))
%time

nb = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB(alpha=0.01)),])

nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)



print('accuracy %s' % accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred,target_names=uniqueTopics))
%time

sgd = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-2, random_state=42, max_iter=5, tol=None)),])

sgd.fit(X_train, y_train)



y_pred = sgd.predict(X_test)



print('accuracy %s' % accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred,target_names=uniqueTopics))
forest = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', RandomForestClassifier(max_features='sqrt',n_estimators=1000, max_depth=3,random_state=0)),])

forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)



print('accuracy %s' % accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred,target_names=uniqueTopics))
xgboost = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', xgb.XGBClassifier(n_jobs=1,max_depth=3,learning_rate=0.01,n_estimators=1000)),])

xgboost.fit(X_train, y_train)

y_pred = xgboost.predict(X_test)



print('accuracy %s' % accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred,target_names=uniqueTopics))
%time



svc = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SVC(gamma='scale', decision_function_shape='ovo'))])

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)



print('accuracy %s' % accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred,target_names=uniqueTopics))
#Evaluating Logistic Regression, since it has comparatevely better accuracy(83%)

logregclf = LogisticRegression(n_jobs=1, C=0.5)

logregclf.fit(X_train_tfidf, y_train)

y_pred = logregclf.predict(X_test_tfidf)

print("---Misclassified Examples---")

for x, y, y_hat in zip(X_test, lbl_enc.inverse_transform(y_test), lbl_enc.inverse_transform(y_pred)):

    if y != y_hat:

        print(f'Cleaned Comment: {x} | Original Topic: {y} | Predicted Topic: {y_hat}')
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt

X, y = X_train_tfidf, y_train

title = "Learning Curves (Logistic Regression)"

# SVC is more expensive so we do a lower number of CV iterations:

cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

estimator = LogisticRegression()

plot_learning_curve(estimator, title, X, y, (0.5, 1.01), cv=cv, n_jobs=10)

plt.show();
%time

wv = gensim.models.KeyedVectors.load_word2vec_format(dataDirectory+"/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin.gz", binary=True)

wv.init_sims(replace=True)

list(islice(wv.vocab, 13030, 13050))
# The common way is to average the two word vectors. BOW based approaches which includes averaging.

def word_averaging(wv, words):

    all_words, mean = set(), []

    

    for word in words:

        if isinstance(word, np.ndarray):

            mean.append(word)

        elif word in wv.vocab:

            mean.append(wv.syn0norm[wv.vocab[word].index])

            all_words.add(wv.vocab[word].index)



    if not mean:

        logging.warning("cannot compute similarity with no input %s", words)

        # FIXME: remove these examples in pre-processing

        return np.zeros(wv.vector_size,)



    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)

    return mean



def  word_averaging_list(wv, text_list):

    return np.vstack([word_averaging(wv, post) for post in text_list ])



def w2v_tokenize_text(text):

    tokens = []

    for sent in nltk.sent_tokenize(text, language='english'):

        for word in nltk.word_tokenize(sent, language='english'):

            if len(word) < 2:

                continue

            tokens.append(word)

    return tokens
train, test = train_test_split(df, test_size=0.3, random_state = 42)



test_tokenized = test.apply(lambda r: w2v_tokenize_text(r[commentTextColumn]), axis=1).values

train_tokenized = train.apply(lambda r: w2v_tokenize_text(r[commentTextColumn]), axis=1).values



X_train_word_average = word_averaging_list(wv,train_tokenized)

X_test_word_average = word_averaging_list(wv,test_tokenized)
%time

logreg = LogisticRegression(n_jobs=1, C=1e5)

logreg = logreg.fit(X_train_word_average, train[agentAssignedColumn])

y_pred = logreg.predict(X_test_word_average)



print('accuracy %s' % accuracy_score(y_pred, test[agentAssignedColumn]))

print(classification_report(test[agentAssignedColumn], y_pred,target_names=uniqueTopics))
# Doc2vec, taking the linear combination of every term in the document creates a random walk with bias process in the word2vec space.

def label_sentences(corpus, label_type):

    """

    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.

    We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is

    a dummy index of the post.

    """

    labeled = []

    for i, v in enumerate(corpus):

        label = label_type + '_' + str(i)

        labeled.append(TaggedDocument(v.split(), [label]))

    return labeled
X_train, X_test, y_train, y_test = train_test_split(df[commentTextColumn], df[agentAssignedColumn], random_state=0, test_size=0.3)

X_train = label_sentences(X_train, 'Train')

X_test = label_sentences(X_test, 'Test')



all_data = X_train + X_test

all_data[:2]
model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)

model_dbow.build_vocab([x for x in tqdm(all_data)])



for epoch in range(30):

    model_dbow.train([x for x in tqdm(all_data)], total_examples=len(all_data), epochs=1)

    model_dbow.alpha -= 0.002

    model_dbow.min_alpha = model_dbow.alpha

def get_vectors(model, corpus_size, vectors_size, vectors_type):

    """

    Get vectors from trained doc2vec model

    :param doc2vec_model: Trained Doc2Vec model

    :param corpus_size: Size of the data

    :param vectors_size: Size of the embedding vectors

    :param vectors_type: Training or Testing vectors

    :return: list of vectors

    """

    vectors = np.zeros((corpus_size, vectors_size))

    for i in range(0, corpus_size):

        prefix = vectors_type + '_' + str(i)

        vectors[i] = model.docvecs[prefix]

    return vectors



train_vectors_dbow = get_vectors(model_dbow, len(X_train), 300, 'Train')

test_vectors_dbow = get_vectors(model_dbow, len(X_test), 300, 'Test')



logreg = LogisticRegression(n_jobs=1, C=1e9)

logreg = logreg.fit(train_vectors_dbow, y_train)

y_pred = logreg.predict(test_vectors_dbow)



print('accuracy %s' % accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred,target_names=uniqueTopics))
train_size = int(len(df) * .7)

print ("Train size: %d" % train_size)

print ("Test size: %d" % (len(df) - train_size))



train_comments = df[commentTextColumn][:train_size]

train_topics = df[agentAssignedColumn][:train_size]



test_comments = df[commentTextColumn][train_size:]

test_topics = df[agentAssignedColumn][train_size:]



max_words = 500

tokenize = text.Tokenizer(num_words=max_words, char_level=False)
%time

tokenize.fit_on_texts(train_comments) # only fit on train

x_train = tokenize.texts_to_matrix(train_comments)

x_test = tokenize.texts_to_matrix(test_comments)



encoder = LabelEncoder()

encoder.fit(train_topics)

y_train = encoder.transform(train_topics)

y_test = encoder.transform(test_topics)



num_classes = np.max(y_train) + 1

y_train = utils.to_categorical(y_train, num_classes)

y_test = utils.to_categorical(y_test, num_classes)



print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)

print('y_train shape:', y_train.shape)

print('y_test shape:', y_test.shape)
batch_size = 10

epochs = 10



# Build the model

model = Sequential()

model.add(Dense(512, input_shape=(max_words,)))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)



score = model.evaluate(x_test, y_test,batch_size=batch_size, verbose=1)

print('Test accuracy:', score[1])
# The maximum number of words to be used. (most frequent)

MAX_NB_WORDS = 500

# Max number of words in each complaint.

MAX_SEQUENCE_LENGTH = 15

# This is fixed.

EMBEDDING_DIM = 100



tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

tokenizer.fit_on_texts(df[commentTextColumn].values)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))



X = tokenizer.texts_to_sequences(df[commentTextColumn].values)

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', X.shape)



Y = pd.get_dummies(df[agentAssignedColumn]).values

print('Shape of label tensor:', Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.30, random_state = 42)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
# Build the model

model = Sequential()

model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))

model.add(SpatialDropout1D(0.2))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
epochs = 10

batch_size = 5



history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])



accr = model.evaluate(X_test,Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
plt.title('Loss')

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show();



plt.title('Accuracy')

plt.plot(history.history['acc'], label='train')

plt.plot(history.history['val_acc'], label='test')

plt.legend()

plt.show();
new_complaint = ['Avinash want to cancel service.']

seq = tokenizer.texts_to_sequences(new_complaint)

padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)

pred = model.predict(padded)

labels = uniqueTopics

print(pred, labels[np.argmax(pred)])
def removeFrequentOccuringWords(wordList):

    commonWords = ['customer', 'want', 'plan', 'month', 'new', 'inquire', 'check']

    wordList = [word for word in wordList if word not in commonWords]

    return ','.join(wordList)
labelsKeywords = pd.DataFrame(columns = ['Label', 'Fuzzy Words', 'Strict Words'])



labelColumnList = list(df[agentAssignedColumn].value_counts().to_dict())

strictWordList = ['']

fuzzyWordList = ['']

for reasoncode in labelColumnList:

    fuzzyWordList.append(','.join(removeFrequentOccuringWords(wordFrequencyListPlot(reasoncode)).split(',')[-2:]))

    strictWordList.append(','.join(removeFrequentOccuringWords(wordFrequencyListPlot(reasoncode)).split(',')[:6]))



labelsKeywords['Label'] = ['Others'] + labelColumnList

labelsKeywords['Fuzzy Words'] = fuzzyWordList

labelsKeywords['Strict Words'] = strictWordList

labelsKeywords.head()
def find_best_label(TrainingWords, StrictWords, text_in):   

    clnTxt = [st.stem(u.lower()) for u in  text_in.split()]        

    if(len(clnTxt)==0):

        clnTxt =  'Others'     

    NL = len(TrainingWords)

    probV = [0.0 for k in range(NL)]

    CM_A = []

    for j in range(NL):

        TrSet = set([ st.stem(u.lower())  for u in TrainingWords[j].split()])

        SWords = set([ u.lower().strip() for u in StrictWords[j].split(",")])  if len(StrictWords[j]) else []

        matching =  [s for s in SWords  if ((s in text_in) and len(set(s.split()).intersection(text_in.split()))==len(s.split())) ] 

        cm = list(TrSet.intersection(clnTxt))

        if len(matching)>0:

            probV[j] = 1

            CM_A.append(matching)

            break

        else:                

            if len(cm):

                CM_A.append(cm)

            else:

                CM_A.append([])

            probV[j] = (len(cm)/float(len(clnTxt)))

    return({'idx':np.argmax(probV), 'maxV':np.max(probV), 'wa':CM_A[np.argmax(probV)]})
%time

TestData = df.copy()

test_buffer = TestData.iloc[:,1]



TrainingData = labelsKeywords.copy()



fuzzy_word_list = TrainingData['Fuzzy Words']

strict_word_list = TrainingData['Strict Words']

train_buffer = [fuzzy_word_list[i]+" "+strict_word_list[i] for i in range(len(TrainingData))]



Labels = [a for a in TrainingData['Label']]



TestData['New Topic'] = 0.0

TestData['Conf'] = 0.0

TestData['Matching words'] = ["" for i in range(len(TestData))]



for i in range(len(test_buffer)):

    BL = find_best_label(train_buffer, TrainingData['Strict Words'], test_buffer[i])

    TestData['New Topic'][i] = Labels[BL['idx']]

    TestData['Conf'][i] = BL['maxV']

    TestData['Matching words'][i] = ", ".join(BL['wa'])    

    

TestData.head()
originalTopic = TestData[agentAssignedColumn]

newTopic = TestData['New Topic']



print('accuracy %s' % accuracy_score(newTopic, originalTopic))

print(classification_report(originalTopic, newTopic,target_names=uniqueTopics))
#Gensim Preprocessing

def lemmatize_stemming(text):

    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))



def preprocess(text):

    result = []

    for token in gensim.utils.simple_preprocess(text):

        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:

            result.append(lemmatize_stemming(token))

    return result



processed_docs = df[commentTextColumn].map(preprocess)

processed_docs[:5]
%time

#Modelling Step

NUM_TOPICS = len(uniqueTopics) #10 here

print("\n Number of topics = "+str(NUM_TOPICS)+" \n")

sizeDf=str(len(df))



dictionary = gensim.corpora.Dictionary(processed_docs)

dictionary.filter_extremes()

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]



lda_model_bow = gensim.models.LdaMulticore(bow_corpus, num_topics=NUM_TOPICS, id2word=dictionary, workers = 2, passes = 5, iterations = 100, eval_every=5)



# # Saving the Model

# modelFileName = 'bowGensimModel'+sizeDf+'('+str(NUM_TOPICS)+').gensim'

# lda_model_bow.save(modelFileName)

# print('\n Model Saved as: '+modelFileName)



for idx, topic in lda_model_bow.print_topics(-1):

    print('Topic: {} \nWords: {}'.format(idx, topic))
%time

lda_display = gensimvis.prepare(lda_model_bow, bow_corpus, dictionary, sort_topics = False)



# #If you want to save the visualization

# pyLDAvis.save_html(lda_display, 'bowGensimModel'+str(NUM_TOPICS)+'.html')



pyLDAvis.display(lda_display)
%time

# Based on the Dirichlet Equations, these could be a possible Topic Assignments

TopicList= {0:'Roaming Plans',

1:'Others',

2:'Add New Line',

3:'Cancel Service',

4:'Data Usage',

5:'Poor Connectivity',

6:'High Bill',

7:'Deactivate',

8:'Change Plans',

9:'Port Out'}





df['topicNumDistributionColumn'] = lda_model_bow.get_document_topics(bow_corpus, minimum_probability=0.1)



topiclist = df['topicNumDistributionColumn'].tolist()

newTopic = []

for element in topiclist: 

    newTopic.append(str(sorted(element, key=lambda x: -x[1])[-1:]))

    

df['NewTopic'] = newTopic







topicExpansionInNumberPrimary = []

for item in df['NewTopic']:

    itemToList = re.findall(r'\d+', item)

    if len(itemToList)==3:

        topicExpansionInNumberPrimary.append(TopicList[int(itemToList[0])])

    elif len(itemToList)==0:

        topicExpansionInNumberPrimary.append(None)

    else:

        topicExpansionInNumberPrimary.append(None)

df['NewTopic'] = topicExpansionInNumberPrimary



originalTopic = df[agentAssignedColumn]

newTopic = df['NewTopic']



print('accuracy %s' % accuracy_score(newTopic, originalTopic))

print(classification_report(originalTopic, newTopic,target_names=uniqueTopics))
processed_docs = df[commentTextColumn].to_list()

token_ = [doc.split(" ") for doc in processed_docs]

bigram = Phrases(token_, min_count=1, threshold=2,delimiter=b' ')





bigram_phraser = Phraser(bigram)



bigram_token = []

for sent in token_:

    bigram_token.append(bigram_phraser[sent])



#now you can make dictionary of bigram token 

dictBigram = gensim.corpora.Dictionary(bigram_token)

# dictBigram.filter_extremes(no_above=0.5, keep_n=100000)



#Convert the word into vector, and now you can use from gensim 

bow_corpus_bigram = [dictBigram.doc2bow(text) for text in bigram_token]
lda_model_bow_bigram = gensim.models.LdaMulticore(bow_corpus_bigram, num_topics=NUM_TOPICS, id2word=dictBigram, workers = 2, passes = 5, iterations = 100, eval_every=5)



# # Saving the Model

# modelFileName = 'bowGensimModelBigram'+sizeDf+'('+str(NUM_TOPICS)+').gensim'

# lda_model_bow_bigram.save(modelFileName)

# print('\n Model Saved as: '+modelFileName)



for idx, topic in lda_model_bow_bigram.print_topics(-1):

    print('Topic: {} \nWords: {}'.format(idx, topic))
%time

#Modelling Step

NUM_TOPICS = len(uniqueTopics) #10 here

print("\n Number of topics = "+str(NUM_TOPICS)+" \n")

sizeDf=str(len(df))



dictionary = gensim.corpora.Dictionary(processed_docs)

dictionary.filter_extremes()

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

tfidf = models.TfidfModel(bow_corpus)

corpus_tfidf = tfidf[bow_corpus]



lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=NUM_TOPICS, id2word=dictionary, passes=5, workers=2, iterations = 100, eval_every=5)

# # Saving the Model

# modelFileName = 'tfidfGensimModel'+sizeDf+'('+str(NUM_TOPICS)+').gensim'

# lda_model_bow.save(modelFileName)

# print('\n Model Saved as: '+modelFileName)



for idx, topic in lda_model_tfidf.print_topics(-1):

    print('Topic: {} \nWords: {}'.format(idx, topic))
%time

lda_display = gensimvis.prepare(lda_model_tfidf, corpus_tfidf, dictionary, sort_topics = False)



# #If you want to save the visualization

# pyLDAvis.save_html(lda_display, 'tfidfGensimModel'+str(NUM_TOPICS)+'.html')



pyLDAvis.display(lda_display)
%time

# Based on the Dirichlet Equations, these could be a possible Topic Assignments

TopicList= {0:'High Bill',

1:'Roaming Plans',

2:'Others',

3:'Change Plans',

4:'Cancel Service',

5:'Poor Connectivity',

6:'Add New Line',

7:'Port Out',

8:'Data Usage',

9:'Deactivate'}





df['topicNumDistributionColumn'] = lda_model_tfidf.get_document_topics(corpus_tfidf, minimum_probability=0.1)



topiclist = df['topicNumDistributionColumn'].tolist()

newTopic = []

for element in topiclist: 

    newTopic.append(str(sorted(element, key=lambda x: -x[1])[-1:]))

    

df['NewTopic'] = newTopic







topicExpansionInNumberPrimary = []

for item in df['NewTopic']:

    itemToList = re.findall(r'\d+', item)

    if len(itemToList)==3:

        topicExpansionInNumberPrimary.append(TopicList[int(itemToList[0])])

    elif len(itemToList)==0:

        topicExpansionInNumberPrimary.append(None)

    else:

        topicExpansionInNumberPrimary.append(None)

df['NewTopic'] = topicExpansionInNumberPrimary



originalTopic = df[agentAssignedColumn]

newTopic = df['NewTopic']



print('accuracy %s' % accuracy_score(newTopic, originalTopic))

print(classification_report(originalTopic, newTopic,target_names=uniqueTopics))
processed_docs = df[commentTextColumn].to_list()

token_ = [doc.split(" ") for doc in processed_docs]

bigram = Phrases(token_, min_count=1, threshold=2,delimiter=b' ')





bigram_phraser = Phraser(bigram)



bigram_token = []

for sent in token_:

    bigram_token.append(bigram_phraser[sent])



#now you can make dictionary of bigram token 

dictBigram = gensim.corpora.Dictionary(bigram_token)

# dictBigram.filter_extremes(no_above=0.5, keep_n=100000)



#Convert the word into vector, and now you can use from gensim 

corpus_bigram = [dictBigram.doc2bow(text) for text in bigram_token]



tfidf_model_bigram = models.TfidfModel(corpus_bigram)

corpus_tfidf_bigram = tfidf_model_bigram[corpus_bigram]
lda_model_tfidf_bigram = gensim.models.LdaMulticore(corpus_tfidf_bigram, num_topics=NUM_TOPICS, id2word=dictBigram, passes=5, workers=2, iterations = 100, eval_every=5)

# # Saving the Model

# modelFileName = 'tfidfGensimModelBigram'+sizeDf+'('+str(NUM_TOPICS)+').gensim'

# lda_model_bow.save(modelFileName)

# print('\n Model Saved as: '+modelFileName)



for idx, topic in lda_model_tfidf_bigram.print_topics(-1):

    print('Topic: {} \nWords: {}'.format(idx, topic))