# all of the imports

import pandas as pd

import numpy as np

import pickle 

import patsy

import matplotlib.pyplot as plt

import seaborn as sns





from sklearn.pipeline import make_pipeline

from sklearn.cross_validation import train_test_split

% matplotlib inline

from sklearn import preprocessing as pp

import warnings

warnings.filterwarnings('ignore')



#from sqlalchemy import create_engine

#cnx = create_engine('postgresql://username:password@IP:PORT/user')
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.naive_bayes import MultinomialNB



from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.preprocessing import Normalizer

from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score, classification_report



from sklearn.metrics import make_scorer

from sklearn.metrics import recall_score



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import roc_curve, auc
import nltk

import itertools

from nltk.probability import FreqDist



stopset = set(nltk.corpus.stopwords.words('english'))
def rotate(ax, degree):

    for tick in ax.get_xticklabels():

        tick.set_rotation(degree)
questions = pd.read_csv('../input/Questions.csv',encoding='latin1')
questions.info()
questions.head()
# extract all the code part 

a = questions['Body'].str.extractall(r'(<code>[^<]+</code>)')
# unstack and convert into a single column for cleaning

test = a.unstack('match')



test.columns = test.columns.droplevel()

# put all columns together

code = pd.DataFrame(test.apply(lambda x: x.str.cat(), axis=1,reduce=True))

# rename 

code.columns = ['CodeBody']

# remove the html tags finally

code['CodeBody'] = code['CodeBody'].str.replace(r'<[^>]+>|\n|\r',' ')
# remove the code part from questions

body = questions['Body'].str.replace(r'<code>[^<]+</code>',' ')

# build up the question part from questions

questions['QuestionBody'] = body.str.replace(r"<[^>]+>|\n|\r", " ")
# Join the codebody by index

questions = questions.join(code)

# final cleaned dataset

questions_final = questions.drop('Body',axis=1)
# assume all answers without userID are from the same guy ID 0

questions_final['OwnerUserId'].fillna(0,inplace=True)

questions_final.OwnerUserId = questions_final.OwnerUserId.astype(int)
tags = pd.read_csv('../input/Tags.csv',encoding='latin1')
tags.head()
#when I was writing data into sql I found few errors

tagID = set(tags.Id)

questionID = set(questions_final.Id)

errors=tagID-questionID

print(errors)

tags = tags[tags.Tag != 40115300]
tags = tags[tags.Tag.notnull()]

#tags.to_csv('tags_final.csv',index=False)
#tags.groupby('Id').count()

fig, ax = plt.subplots()

sns.distplot(tags.groupby('Id').count())

ax.set_xlabel('number of tags')
tagsByquestion = tags.groupby('Id',as_index=False).agg(lambda x: ' '.join(x))
fig, ax = plt.subplots()

sns.distplot(questions[questions.Score <=10].Score,kde=False)

ax.set_xlabel('distribution of scores')
dfFinal = questions_final.loc[(questions_final.Score>=5) | (questions_final.Score<0)]
texts = list(dfFinal.Title)

# Tokenize the titles

texts = [nltk.word_tokenize(text) for text in texts]

# pos tag the tokens

txtpos = [nltk.pos_tag(texts) for texts in texts]

# for titles we only care about verbs and nouns

txtpos = [[w for w in s if (w[1][0] == 'N' or w[1][0] == 'V') and w[0].lower() not in stopset] 

                  for s in txtpos]
qbodys = list(dfFinal.QuestionBody)

#break into sentences

qsents = [nltk.sent_tokenize(text) for text in qbodys]

# Tokenize the question body

qbodys = [nltk.word_tokenize(text) for text in qbodys]

# attach tags to the body

qpos = [nltk.pos_tag(texts) for texts in qbodys]
from collections import defaultdict



stats = defaultdict(dict)
import re



RE_URL = re.compile(r'https?://')



for index, body in enumerate(qsents):

    

    stats[index]['question'] = 0

    stats[index]['exclam'] = 0

    stats[index]['url'] = 0

    for sent in body:

        ss = sent.strip()

        if ss:

            if ss.endswith('?'):

                stats[index]['question'] += 1

            if ss.endswith('!'):

                stats[index]['exclam'] += 1

            stats[index]['url'] += len(RE_URL.findall(sent))

    stats[index]['finalthanks'] = 1 if body and 'thank' in body[-1].lower() else 0

    stats[index]['textLen'] = len(body)

    
df = pd.DataFrame.from_dict(stats,orient='index')



# this part should be done in the first place, I realize it only till the last phase

df['codeLen'] = [len(list) if list else 0 

                   for list in questions.loc[(questions_final.Score>=5) | (questions_final.Score<0),\

                                             "Body"].str.findall(r'(<code>[^<]+</code>)')]
tagNum = tags.groupby('Id')['Tag'].count()
def getSumTF(wordlist, qfile,cfile):

    if not wordlist or not qfile:

        return 0

    if cfile is np.nan:

        cfile = []

    if type(wordlist) is str:

        wordlist = wordlist.split(' ')

        wordset = set(wordlist)

    else:

        wordset = set([word for word,_ in wordlist])

    freq = 0

    freqdict = nltk.FreqDist(qfile)

    for word in wordset:

        freq+=freqdict[word]

        if cfile:

            if word in cfile:

                freq+=5

    return freq/len(wordlist)
tagNum.columns = ['Id', 'tagNum']
clist = list(dfFinal.CodeBody)
tagsByquestion.head()
dfFinal = dfFinal.merge(tagsByquestion,on='Id',how='left')
titleTFSum = []



for index, words in enumerate(txtpos):

    titleTFSum.append(getSumTF(words, qbodys[index], clist[index]))

    

tagTFSum = []



for index, words in enumerate(list(dfFinal.Tag)):

    tagTFSum.append(getSumTF(words, qbodys[index], clist[index]))
df['titleTFSum'] , df['tagTFSum'] = titleTFSum, tagTFSum



df['Id'] = list(dfFinal.Id)
tagNum = pd.DataFrame(tagNum).reset_index(level=0)
dfFinal = dfFinal.merge(df, on ='Id',how='left').merge(tagNum, on='Id',how='left')
dfFinal.loc[dfFinal.Score<0,'label'] = 'Bad'



dfFinal.loc[dfFinal.Score>=5,'label'] = 'Good'
dfFinal.head()
# this function was required for GaussianNB but not required for SGD

class DenseTransformer(BaseEstimator,TransformerMixin):



    def transform(self, X, y=None, **fit_params):

        return X.todense()



    def fit_transform(self, X, y=None, **fit_params):

        self.fit(X, y, **fit_params)

        return self.transform(X)



    def fit(self, X, y=None, **fit_params):

        return self
columns = ['question', 'exclam', 'finalthanks', 'textLen', 'url', 'codeLen',

       'titleTFSum', 'tagTFSum', 'tagNum']
class GetItemTransformer(BaseEstimator,TransformerMixin):

    def __init__(self, field):

        self.field = field

    def fit(self, X, y=None):

        return self

    def transform(self,X):

        if len(self.field)==1:

            if self.field[0] =='QuestionBody':

                return list(X.QuestionBody)

            else:

                return list(X.Title)

        return X.loc[:,self.field]
dftest = dfFinal.drop(['OwnerUserId','CreationDate','Score','CodeBody','Tag'],axis=1)
Y = dftest.label

X = dftest.drop(['label','Id'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.30)
'''This is our baseline of model accuracy. We need to beat this accuracy 

while trying to maximize our recall on bad labels.'''



def dummyGuess(x):

    return pd.Series(['Good'] * len(x))
accuracy_score(Y_test, dummyGuess(Y_test))
'''Using nltk tokenizer will significantly slow down the fitting time but will slightly increase the accuracy.

I trained the model with default tokenizer and then switched to nltk tokenizer later on'''



pipeline = Pipeline([

    ('features', FeatureUnion(

        transformer_list = [

        ('stats', Pipeline([

                ('extract', GetItemTransformer(columns)),

                ('substractK', SelectKBest(k=5))]))

        ,

        ('title',Pipeline([

            ('extract', GetItemTransformer(['Title'])),

            ('count', TfidfVectorizer(stop_words=stopset,min_df=0.03,max_df=0.7,tokenizer=nltk.word_tokenize)),

            #('Sum', SumTransformer())

        ])),

        ('question', Pipeline([

            ('extract', GetItemTransformer(['QuestionBody'])),

            ('tfidf', CountVectorizer(stop_words=stopset,min_df=0.01,max_df=0.8,tokenizer=nltk.word_tokenize)),

        ])),

    ],

    # the weight was trained seperately, 

    # I also controlled the weight to be fairly equal assignned.

    transformer_weights={

            'stats': 0.4,

            'title':0.2,

            'Question': 0.4

        }

            )),

    ('scaler',Normalizer()),    

    ('estimators', SGDClassifier(alpha=0.001,loss='modified_huber',penalty='l2')),

])
pipeline.fit(X_train, Y_train)
y = pipeline.predict(X_test)
accuracy_score(Y_test, y)
print(classification_report(Y_test, y))
test = pipeline.predict_proba(X_test)
predict = ['Bad' if pair[0]>=0.35 else 'Good' for pair in test]



print(classification_report(Y_test,predict))



print(accuracy_score(Y_test,predict))
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #print("Normalized confusion matrix")

    #else:

        #print('Confusion matrix, without normalization')



    #print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, "{0:.0f}%".format(cm[i, j]*100),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

cnf_matrix = confusion_matrix(Y_test,predict)

                              

np.set_printoptions(precision=2)



class_names =['Bad','Good']



plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,

                      title='Normalized confusion matrix')
plt.figure()



for model in models:

    pipeline.steps[2]= ('estimator',models[model])

    pipeline.fit(X_train, Y_train)

    test = pipeline.predict_proba(X_test)

    predict = [pair[0] for pair in test]

    # Get Receiver Operating Characteristic (ROC) and Area Under Curve (AUC)

    fpr, tpr, _ = roc_curve(Y_test, predict,pos_label='Bad')

    auc_ = auc(fpr, tpr)

    # Plot it

    plt.plot(fpr, tpr, label=model)

plt.legend()

plt.plot([[0,0],[1,1]])

plt.title('AUC ')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')
from sklearn.ensemble import RandomForestClassifier



models = {'RF': RandomForestClassifier(max_depth=20, n_estimators=20),

            'SGD': SGDClassifier(alpha=0.001,loss='modified_huber',penalty='l2'),

            'NB':MultinomialNB(),

            'LR':LogisticRegression(C=0.01)

                   }
bodys = list(questions_final.QuestionBody)

#break into sentences

sents = [nltk.sent_tokenize(text) for text in bodys]

# Tokenize the question body

bodys = [nltk.word_tokenize(text) for text in bodys]
titles = list(questions_final.Title)

# Tokenize the titles

titles = [nltk.word_tokenize(text) for text in titles]

# pos tag the tokens

titlepos = [nltk.pos_tag(texts) for texts in titles]

# for titles we only care about verbs and nouns

titlepos = [[w for w in s if (w[1][0] == 'N' or w[1][0] == 'V') and w[0].lower() not in stopset] 

                  for s in titlepos]
completeDf = pd.DataFrame.from_dict(dfstats,orient='index')



# this part should be done in the first place, I realize it only till the last phase

completeDf['codeLen'] = [len(list) if list else 0 

                   for list in questions.loc[:,"Body"].str.findall(r'(<code>[^<]+</code>)')]
qFinal.loc[qFinal.Score<0,'label'] = 'Bad'



qFinal.loc[qFinal.Score>=0,'label'] = 'Good'
Y = qTest.label

X = qTest.drop(['label','Id'], axis=1)
class SumTransformer(BaseEstimator,TransformerMixin):



    def transform(self, X, y=None, **fit_params):

        X = X.todense()

        return X.sum(axis=1)



    def fit_transform(self, X, y=None, **fit_params):

        #self.fit(X, y, **fit_params)

        return self.transform(X)



    def fit(self, X, y=None, **fit_params):

        return self
from sklearn.metrics import precision_score
pipeline2 = Pipeline([

    ('features', FeatureUnion(

        transformer_list = [

        ('stats', Pipeline([

                ('extract', GetItemTransformer(columns)),

                ('substractK', SelectKBest(k=5))]))

        ,

        ('title',Pipeline([

            ('extract', GetItemTransformer(['Title'])),

            ('count', TfidfVectorizer(stop_words=stopset,min_df=0.03,max_df=0.8,tokenizer=nltk.word_tokenize)),

            #('Sum', SumTransformer())

        ])),

        ('question', Pipeline([

            ('extract', GetItemTransformer(['QuestionBody'])),

            ('tfidf', CountVectorizer(stop_words=stopset,min_df=0.01,max_df=0.8)),

            # I didn't do this with previous dataset

            ('substractK', SelectKBest(k=500))

        ])),

    ],

    # the weight was trained seperately, 

    # I also controlled the weight to be fairly equal assignned.

    transformer_weights={

            'stats': 0.4,

            'title':0.2,

            'Question': 0.4

        }

            )),

    ('scaler',Normalizer()),

    ('estimators', SGDClassifier(alpha=0.001,loss='modified_huber',penalty='l2',class_weight={'Bad':15, 'Good':1})),

])
# the model will easily include all the Bad questions but the precision becomes very low

recall_scorer = make_scorer(precision_score, pos_label="Bad")
#grid_search2 = RandomizedSearchCV(pipeline2, param_grid,scoring=recall_scorer, verbose=5,n_jobs=5,n_iter=4)
#grid_search2.fit(X_train,Y_train)
#grid_search2.best_score_
#Y_pred = pipeline2.predict_proba(X_test)
#grid_search2.best_params_
#predict = [pair[0] for pair in Y_pred]