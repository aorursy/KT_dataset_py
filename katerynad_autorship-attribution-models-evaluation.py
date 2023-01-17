import pandas as pd

from pandas import Series,DataFrame

import numpy as np
#KimKardashianTweets data

df_kk=pd.read_csv('../input/KimKardashianTweets.csv')

len(df_kk)
#HillaryClintonTweets data

df_hc=pd.read_csv('../input/HillaryClintonTweets.csv')

len(df_hc)
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
from sklearn.cross_validation import train_test_split

twt_train, twt_test, author_train, author_test = train_test_split(df['text'], df['author'], test_size=0.4, random_state=42)
len(twt_train)
len(twt_test)
ScoreSummaryByModel = list()
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

from sklearn.cross_validation import cross_val_score



def ModelEvaluation (model,comment):

    pipeline = Pipeline([('vect', CountVectorizer())

                  , ('tfidf', TfidfTransformer())

                  , ('model', model)])

    scores = cross_val_score(pipeline, df['text'], df['author'], cv=10)	

    mean = scores.mean()	

    std = scores.std()	

    #The mean score and the 95% confidence interval of the score estimate (accuracy)

    ScoreSummaryByModel.append([comment,mean, std, "%0.3f (+/- %0.3f)" % (mean, std * 2)])

    print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
from sklearn.naive_bayes import MultinomialNB

ModelEvaluation (MultinomialNB(),'Naive Bayes classifier')
from sklearn.naive_bayes import BernoulliNB

ModelEvaluation (BernoulliNB(binarize=0.0),'Bernoulli Naive Bayes')
from sklearn.svm import LinearSVC

ModelEvaluation (LinearSVC(),'LinearSVC')
from sklearn.svm import SVC

ModelEvaluation (SVC(),'SVC, default rbf kernel')
ModelEvaluation (SVC(kernel='linear'),'SVC, linear kernel')
from sklearn.linear_model import SGDClassifier

ModelEvaluation (SGDClassifier(),'SGD')
df_ScoreSummaryByModel=DataFrame(ScoreSummaryByModel,columns=['Method','Mean','Std','Accuracy'])

df_ScoreSummaryByModel.sort_values(['Mean'],ascending=False,inplace=True)

df_ScoreSummaryByModel
from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

from nltk.stem.porter import PorterStemmer

def text_process(text):

    """

    Takes in a string of text, then performs the following:

    1. Tokenizes and removes punctuation

    2. Removes  stopwords

    3. Stems

    4. Returns a list of the cleaned text

    """



    # tokenizing

    tokenizer = RegexpTokenizer(r'\w+')

    text_processed=tokenizer.tokenize(text)

    

    # removing any stopwords

    stoplist = stopwords.words('english')

    stoplist.append('twitter')

    stoplist.append('pic')

    stoplist.append('com')

    stoplist.append('net')

    stoplist.append('gov')

    stoplist.append('tv')

    stoplist.append('www')

    stoplist.append('twitter')

    stoplist.append('num')

    stoplist.append('date')

    stoplist.append('time')

    stoplist.append('url')

    stoplist.append('ref')



    text_processed = [word.lower() for word in text_processed if word.lower() not in stoplist]

    

    # steming

    porter_stemmer = PorterStemmer()

    

    text_processed = [porter_stemmer.stem(word) for word in text_processed]

    



    return text_processed
ScoreSummaryByModelParams = list()
from sklearn.grid_search import GridSearchCV

def ModelParamsEvaluation (vectorizer,model,params,comment):

    pipeline = Pipeline([

    ('vect', vectorizer),

    ('tfidf', TfidfTransformer()),

    ('clf', model),

    ])

    grid_search = GridSearchCV(estimator=pipeline, param_grid=params, verbose=1)

    grid_search.fit(df['text'], df['author'])

    print("Best score: %0.3f" % grid_search.best_score_)

    print("Best parameters set:")

    best_parameters = grid_search.best_estimator_.get_params()

    for param_name in sorted(params.keys()):

        print("\t%s: %r" % (param_name, best_parameters[param_name]))

        ScoreSummaryByModelParams.append([comment,grid_search.best_score_,"\t%s: %r" % (param_name, best_parameters[param_name])])
p = {'vect__analyzer':('char', 'char_wb'),

    'vect__max_df': (0.5, 0.75, 1.0),

    'vect__ngram_range': ((2, 2), (3, 3)), 

    'clf__alpha': (1,0.1,0.01,0.001,0.0001,0)}

ModelParamsEvaluation(CountVectorizer(),BernoulliNB(),p,'Bernoulli Naive Bayes')
p = {

    'vect__max_df': (0.5, 0.75, 1.0),

    'vect__ngram_range': ((1, 1), (3, 3), (5,5),(2,5)), 

    'clf__alpha': (1,0.1,0.01,0.001,0.0001,0)}

ModelParamsEvaluation(CountVectorizer(analyzer='word'),BernoulliNB(),p,'Bernoulli Naive Bayes, analyzer=word')
p = {

    'vect__max_df': (0.5, 0.75, 1.0),

    'vect__ngram_range': ((1, 1), (3, 3), (5,5),(2,5)), 

    'clf__alpha': (1,0.1,0.01,0.001,0.0001,0)}

ModelParamsEvaluation(CountVectorizer(analyzer='word',tokenizer=text_process),BernoulliNB(),p,'Bernoulli Naive Bayes, analyzer=word, tokenizer=text_process')
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import FeatureUnion

word_vector =  CountVectorizer(analyzer='word',stop_words='english',ngram_range=(1, 1),max_df=0.5)

char_vector = CountVectorizer(analyzer='char_wb',ngram_range=(3, 3),max_df=0.75)

vectorizer = FeatureUnion([("chars", char_vector),("words", word_vector)])

p = {'clf__alpha': (1,0.1,0.01,0.001,0.0001,0)}

ModelParamsEvaluation(vectorizer,BernoulliNB(),p,'Bernoulli Naive Bayes, vectorizer')
word_vector =  CountVectorizer(analyzer='word',stop_words='english',ngram_range=(1, 5),max_df=0.5)

char_vector = CountVectorizer(analyzer='char_wb',ngram_range=(3, 3),max_df=0.75)

text_vector = CountVectorizer(analyzer='word',tokenizer=text_process,ngram_range=(3, 3),max_df=0.75)

vectorizer = FeatureUnion([("chars", char_vector),("words", word_vector),("text", text_vector)])

p = {'clf__alpha': (1,0.1,0.01,0.001,0.0001,0)}

ModelParamsEvaluation(vectorizer,BernoulliNB(),p,'Bernoulli Naive Bayes, vectorizer+text_vector')
df_ScoreSummaryByModelParams=DataFrame(ScoreSummaryByModelParams,columns=['Method','BestScore','BestParameter'])

df_ScoreSummaryByModelParams.sort_values(['BestScore'],ascending=False,inplace=True)

df_ScoreSummaryByModelParams
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve, auc,precision_score, accuracy_score, recall_score, f1_score

from scipy import interp

#Visualization

import matplotlib.pyplot as plt

%matplotlib inline
def ROCCurves (Actual, Predicted):

    '''

    Plot ROC curves for the multiclass problem

    based on http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    '''

    # Compute ROC curve and ROC area for each class

    n_classes=2

    fpr = dict()

    tpr = dict()

    roc_auc = dict()

    for i in range(n_classes):

        fpr[i], tpr[i], _ = roc_curve(Actual, Predicted)

        roc_auc[i] = auc(fpr[i], tpr[i])



    # Compute micro-average ROC curve and ROC area

    fpr["micro"], tpr["micro"], _ = roc_curve(Actual.ravel(), Predicted.ravel())

    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    ##############################################################################

    # Plot ROC curves for the multiclass problem



    # Compute macro-average ROC curve and ROC area



    # First aggregate all false positive rates



    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))



    # Then interpolate all ROC curves at this points

    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):

        mean_tpr += interp(all_fpr, fpr[i], tpr[i])



    # Finally average it and compute AUC

    mean_tpr /= n_classes



    fpr["macro"] = all_fpr

    tpr["macro"] = mean_tpr

    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



    # Plot all ROC curves

    plt.figure()

    plt.plot(fpr["micro"], tpr["micro"],

         label='micro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["micro"]),

         linewidth=2)



    plt.plot(fpr["macro"], tpr["macro"],

         label='macro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["macro"]),

         linewidth=2)



    for i in range(n_classes):

        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'

                                   ''.format(i, roc_auc[i]))



    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Some extension of Receiver operating characteristic to multi-class')

    plt.legend(loc="lower right")
def ConfusionMatrix(author_test_b,author_predictions_b):

    cm=confusion_matrix(author_test_b,author_predictions_b)

    plt.matshow(cm)

    plt.title('Confusion matrix')

    plt.colorbar()

    plt.ylabel('Actual')

    plt.xlabel('Predicted')

    plt.show()
def PredictionEvaluation(author_test_b,author_predictions_b):

    print ('Precision: %0.3f' % (precision_score(author_test_b,author_predictions_b)))

    print ('Accuracy: %0.3f' % (accuracy_score(author_test_b,author_predictions_b)))

    print ('Recall: %0.3f' % (recall_score(author_test_b,author_predictions_b)))

    print ('F1: %0.3f' % (f1_score(author_test_b,author_predictions_b)))

    print ('Confussion matrix:')

    print (confusion_matrix(author_test_b,author_predictions_b))

    print ('ROC-AUC: %0.3f' % (roc_auc_score(author_test_b,author_predictions_b)))
word_vector = CountVectorizer(analyzer='word',stop_words='english',ngram_range=(1, 5),max_df=0.5)

char_vector = CountVectorizer(analyzer='char_wb',ngram_range=(3, 3),max_df=0.75)

text_vector = CountVectorizer(analyzer='word',tokenizer=text_process,ngram_range=(3, 3),max_df=0.75)

vectorizer  = FeatureUnion([("chars", char_vector),("words", word_vector),("text", text_vector)])
pipeline = Pipeline([

    ('vect', vectorizer),

    ('tfidf', TfidfTransformer()),

    ('clf', BernoulliNB(alpha=0.001)),

    ])
pipeline.fit(twt_train,author_train)

author_predictions = pipeline.predict(twt_test)
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

author_test_b = lb.fit_transform(author_test.values)

author_predictions_b  = lb.fit_transform(author_predictions)
PredictionEvaluation(author_test_b,author_predictions_b)
ROCCurves (author_test_b,author_predictions_b)
ConfusionMatrix(author_test,author_predictions)
pipeline = Pipeline([

    ('vect', CountVectorizer(analyzer='char',ngram_range=(3, 3),max_df=1)),

    ('tfidf', TfidfTransformer()),

    ('clf', SGDClassifier(alpha=0.0001, penalty='elasticnet')),

    ])
pipeline.fit(twt_train,author_train)

author_predictions = pipeline.predict(twt_test)
author_test_b = lb.fit_transform(author_test.values)

author_predictions_b  = lb.fit_transform(author_predictions)
PredictionEvaluation(author_test_b,author_predictions_b)
ROCCurves (author_test_b,author_predictions_b)
ConfusionMatrix(author_test_b,author_predictions_b)