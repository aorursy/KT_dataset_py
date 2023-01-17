glove_file = '../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt'
import numpy as np # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)



df_train = pd.read_csv('../input/spooky-author-identification/train.csv')

df_test = pd.read_csv('../input/spooky-author-identification/test.csv')

df_sample = pd.read_csv('../input/spooky-author-identification/sample_submission.csv')



#df.dropna(axis=0)

#df.set_index('id', inplace = True)



df_sample.head()
df_test.head()
print(df_train.shape)

print(df_test.shape)
import re

from nltk.corpus import stopwords



stopWords = set(stopwords.words('english'))
len(df_train.columns)
#creating a function to encapsulate preprocessing, to mkae it easy to replicate on  submission data

def processing(df):

    #lowering and removing punctuation

    df['processed'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]','', x.lower()))

    

    #numerical feature engineering

    #total length of sentence

    df['length'] = df['processed'].apply(lambda x: len(x))

    

    #get number of words

    df['words'] = df['processed'].apply(lambda x: len(x.split(' ')))

    

    df['words_not_stopword'] = df['processed'].apply(lambda x: len([t for t in x.split(' ') if t not in stopWords]))

    

    # number of unique words

    df["num_unique_words"] = df["text"].apply(lambda x: len(set(str(x).split())))

    

    # number of characters

    df["num_chars"] = df["text"].apply(lambda x: len(str(x)))

    

    # nmber of punctuations

    #df["num_punctuations"] = df["text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

    

    # number of upper case 

    df["num_words_upper"] = df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

    

    #number of titles characters

    df["num_words_title"] = df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

    

    #get the average word length

    df['avg_word_length'] = df['processed'].apply(lambda x: np.mean([len(t) for t in x.split(' ') if t not in stopWords]) if len([len(t) for t in x.split(' ') if t not in stopWords]) > 0 else 0)

    

    #get number of commas

    df['commas'] = df['text'].apply(lambda x: x.count(','))



    return(df)



df_train = processing(df_train)



df_train.head()
df_train.columns
from sklearn.model_selection import train_test_split



features= [c for c in df_train.columns.values if c  not in ['id','text','author']]

numeric_features= [c for c in df_train.columns.values if c  not in ['id','text','author','processed']]

target = 'author'



X_train, X_test, y_train, y_test = train_test_split(df_train[features], df_train[target], test_size=0.40, random_state=42)

X_train.head()
from sklearn.base import BaseEstimator, TransformerMixin



class TextSelector(BaseEstimator, TransformerMixin):

    """

    Transformer to select a single column from the data frame to perform additional transformations on

    Use on text columns in the data

    """

    def __init__(self, key):

        self.key = key



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        return X[self.key]

    

class NumberSelector(BaseEstimator, TransformerMixin):

    """

    Transformer to select a single column from the data frame to perform additional transformations on

    Use on numeric columns in the data

    """

    def __init__(self, key):

        self.key = key



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        return X[[self.key]]
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer



text = Pipeline([

                ('selector', TextSelector(key='processed')),

                ('tfidf', TfidfVectorizer( stop_words='english'))

            ])



text.fit_transform(X_train)
from sklearn.preprocessing import StandardScaler



length =  Pipeline([

                ('selector', NumberSelector(key='length')),

                ('standard', StandardScaler())

            ])



words =  Pipeline([

                ('selector', NumberSelector(key='words')),

                ('standard', StandardScaler())

            ])



words_not_stopword =  Pipeline([

                ('selector', NumberSelector(key='words_not_stopword')),

                ('standard', StandardScaler())

            ])



num_unique_words =  Pipeline([

                ('selector', NumberSelector(key='num_unique_words')),

                ('standard', StandardScaler()),

            ])



num_chars =  Pipeline([

                ('selector', NumberSelector(key='num_chars')),

                ('standard', StandardScaler()),

            ])



num_words_upper =  Pipeline([

                ('selector', NumberSelector(key='num_words_upper')),

                ('standard', StandardScaler()),

            ])



num_words_title =  Pipeline([

                ('selector', NumberSelector(key='num_words_title')),

                ('standard', StandardScaler()),

            ])



avg_word_length =  Pipeline([

                ('selector', NumberSelector(key='avg_word_length')),

                ('standard', StandardScaler())

            ])



commas =  Pipeline([

                ('selector', NumberSelector(key='commas')),

                ('standard', StandardScaler()),

            ])
from sklearn.pipeline import FeatureUnion



feats = FeatureUnion([('text', text), 

                      ('length', length),

                      ('words', words),

                      ('num_unique_words', num_unique_words),

                      ('num_chars', num_chars),

                      ('num_words_upper', num_words_upper),

                      ('num_words_title', num_words_title),

                      ('words_not_stopword', words_not_stopword),

                      ('avg_word_length', avg_word_length),

                      ('commas', commas)])



feature_processing = Pipeline([('feats', feats)])

feature_processing.fit_transform(X_train)
'''

from sklearn.ensemble import RandomForestClassifier



pipeline = Pipeline([

    ('features',feats),

    ('classifier', RandomForestClassifier(random_state = 42)),

])



pipeline.fit(X_train, y_train)



preds = pipeline.predict(X_test)

np.mean(preds == y_test)

'''
'''

from sklearn.ensemble import GradientBoostingClassifier



gbpipeline = Pipeline([

    ('features',feats),

    ('gbclassifier', GradientBoostingClassifier(random_state = 42)),

])



gbpipeline.fit(X_train, y_train)



preds = gbpipeline.predict(X_test)

np.mean(preds == y_test)

'''
# pipeline.get_params().keys()
# gbpipeline.get_params().keys()
'''

from sklearn.model_selection import GridSearchCV



hyperparameters = { 'features__text__tfidf__max_df': [0.9],

                    'features__text__tfidf__ngram_range': [(1,1)],

                   'classifier__max_depth': [160, 170, 180],

                    'classifier__min_samples_leaf': [4]

                  }

clf = GridSearchCV(pipeline, hyperparameters, cv=5)

 

# Fit and tune model

clf.fit(X_train, y_train)

'''
'''

from sklearn.model_selection import GridSearchCV



gbhyperparameters = { 'features__text__tfidf__max_df': [0.9],

                      'features__text__tfidf__ngram_range': [(1,1)],

                      'gbclassifier__n_estimators': [50, 100, 150],

                      'gbclassifier__min_samples_split': [2, 3], 

                      'gbclassifier__min_samples_leaf' : [2, 4]

                  }

gbclf = GridSearchCV(gbpipeline, gbhyperparameters, cv=5)

 

# Fit and tune model

gbclf.fit(X_train, y_train)

'''

#clf.best_params_
# gbclf.best_params_
'''

#refitting on entire training data using best settings

clf.refit



preds = clf.predict(X_test)

probs = clf.predict_proba(X_test)



np.mean(preds == y_test)

'''
'''

#refitting on entire training data using best settings

gbclf.refit



preds = gbclf.predict(X_test)

probs = gbclf.predict_proba(X_test)



np.mean(preds == y_test)

'''
'''

submission = pd.read_csv('../input/test.csv')



#preprocessing

submission = processing(submission)

predictions = clf.predict_proba(submission)



preds = pd.DataFrame(data=predictions, columns = clf.best_estimator_.named_steps['classifier'].classes_)



#generating a submission file

result = pd.concat([submission[['id']], preds], axis=1)

#result.set_index('id', inplace = True)

result.head()

'''

#result.to_csv('random_forest.csv', index=False)
''' 

submission = pd.read_csv('../input/test.csv')



#preprocessing

submission = processing(submission)

predictions = gbclf.predict_proba(submission)



preds = pd.DataFrame(data=predictions, columns = gbclf.best_estimator_.named_steps['classifier'].classes_)



#generating a submission file

result = pd.concat([submission[['id']], preds], axis=1)

#result.set_index('id', inplace = True)

result.head()



'''
# result.to_csv('gradient_boost.csv', index=False)


train_corpus = []

for i in range(len(df_train)):

    review = re.sub('[^a-zA-Z0-9]', ' ', df_train['text'][i])

    review = review.lower()

    review = review.split()

    review = [word for word in review if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    train_corpus.append(review)



test_corpus = []

for i in range(len(df_test)):

    review = re.sub('[^a-zA-Z0-9]', ' ', df_test['text'][i])

    review = review.lower()

    review = review.split()

    review = [word for word in review if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    test_corpus.append(review)

    

X_Train = np.array(train_corpus)

X_Test = np.array(test_corpus)

y = df_train.iloc[:, 2].values    







'''

import nltk.stem as stm # Import stem class from nltk

    import re

    stemmer = stm.PorterStemmer()



    # Crazy one-liner code here...

    # Explanation above...

    df_train.text = df_train.text.apply(lambda text: " ".join([stemmer.stem(word) for word in text.split(" ")]))

    df_test.text = df_test.text.apply(lambda text: " ".join([stemmer.stem(word) for word in text.split(" ")]))

'''
'''

X_Train = pd.DataFrame(X_Train)

X_Train.columns = ['text']



X_Train.head()

''' 
#''' 

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingClassifier



classifier = Pipeline([('vect', CountVectorizer()),

                      ('tfidf', TfidfTransformer()),

                      ('clf', MultinomialNB()),

                       #('clf', GradientBoostingClassifier(random_state = 42))

])



classifier.fit(X_Train, y)

#'''
#'''

from sklearn.model_selection import GridSearchCV

parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],

              'tfidf__use_idf': (True, False),

              'clf__alpha': (0.05, 1.0)

              #'clf__min_samples_split': [2, 3]

}



gs_clf = GridSearchCV(classifier, parameters)

gs_clf.fit(X_Train, y)

#'''
gs_clf.best_params_
#'''





gs_clf.refit







# Predicting the Test set results

y_pred_proba = gs_clf.predict_proba(X_Test)



tdf = pd.DataFrame(y_pred_proba)

tdf.columns = ['EAP', 'HPL', 'MWS']



submission = pd.read_csv('../input/spooky-author-identification/sample_submission.csv')

result = pd.concat([submission[['id']], tdf], axis=1)

result.head()



result.to_csv('naive_bayes_old.csv', index=False)

#'''
import xgboost as xgb

from tqdm import tqdm

from sklearn.svm import SVC

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB



from nltk.corpus import stopwords

stop_words = stopwords.words('english')
df_train.head()

df_test.head()
def multiclass_logloss(actual, predicted, eps=1e-15):

    """Multi class version of Logarithmic Loss metric.

    :param actual: Array containing the actual target classes

    :param predicted: Matrix with class predictions, one probability per class

    """

    # Convert 'actual' to a binary array if it's not already:

    if len(actual.shape) == 1:

        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))

        for i, val in enumerate(actual):

            actual2[i, val] = 1

        actual = actual2



    clip = np.clip(predicted, eps, 1 - eps)

    rows = actual.shape[0]

    vsota = np.sum(actual * np.log(clip))

    return -1.0 / rows * vsota
lbl_enc = preprocessing.LabelEncoder()

y = lbl_enc.fit_transform(df_train.author.values)

y
xtrain, xvalid, ytrain, yvalid = train_test_split(df_train.text.values, y, 

                                                  stratify = y, 

                                                  random_state = 42, 

                                                  test_size = 0.1, shuffle = True)
type(xvalid)
type(xtrain)
xtrain
print('Train and output {} & {}'.format( xtrain.shape, ytrain.shape))
print('Validation and output {} & {}'.format( xvalid.shape, yvalid.shape))
tfv = TfidfVectorizer(min_df=3,  max_features=None, 

            strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',

            ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,

            stop_words = 'english')



tfv
tfv.fit(list(xtrain) + list(xvalid))

xtrain_tfv =  tfv.transform(xtrain) 

xvalid_tfv = tfv.transform(xvalid)



xvalid_tfv.shape
clf = LogisticRegression(C=1.0)

clf.fit(xtrain_tfv, ytrain)

predictions = clf.predict_proba(xvalid_tfv)



print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
test_predictions = clf.predict_proba(xvalid_tfv)



print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
test_tfv =  tfv.transform(df_test['text']) 
test_predictions = clf.predict_proba(test_tfv)

test_predictions.shape
tdf = pd.DataFrame(test_predictions)

tdf.columns = ['EAP', 'HPL', 'MWS']



submission = pd.read_csv('../input/spooky-author-identification/sample_submission.csv')

result = pd.concat([submission[['id']], tdf], axis=1)

result.head()



result.to_csv('naive_tf_idf.csv', index=False)
ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), stop_words = 'english')



# Fitting Count Vectorizer to both training and test sets (semi-supervised learning)

ctv.fit(list(xtrain) + list(xvalid))

xtrain_ctv =  ctv.transform(xtrain) 

xvalid_ctv = ctv.transform(xvalid)

clf = LogisticRegression(C=1.0)

clf.fit(xtrain_ctv, ytrain)

predictions = clf.predict_proba(xvalid_ctv)



print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
test_predictions = clf.predict_proba(xvalid_ctv)



print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
test_ctv =  ctv.transform(df_test['text']) 
test_predictions = clf.predict_proba(test_ctv)

test_predictions.shape
tdf = pd.DataFrame(test_predictions)

tdf.columns = ['EAP', 'HPL', 'MWS']



submission = pd.read_csv('../input/spooky-author-identification/sample_submission.csv')

result = pd.concat([submission[['id']], tdf], axis=1)

result.head()



result.to_csv('naive_ctv.csv', index=False)
clf = MultinomialNB()

clf.fit(xtrain_tfv, ytrain)

predictions = clf.predict_proba(xvalid_tfv)



print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
clf = MultinomialNB()

clf.fit(xtrain_ctv, ytrain)

predictions = clf.predict_proba(xvalid_ctv)



print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingClassifier



classifier = Pipeline([('vect', CountVectorizer()),

                      ('tfidf', TfidfTransformer()),

                      ('clf', MultinomialNB()),

                       #('clf', GradientBoostingClassifier(random_state = 42))

])



classifier.fit(xvalid, yvalid)



from sklearn.model_selection import GridSearchCV



parameters = {'vect__ngram_range': [(1, 1)],

              'tfidf__use_idf': [True],

              'clf__alpha': ( 0.05, 0.01)

              #'clf__min_samples_split': [2, 3]

             }               





'''parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],

              'tfidf__use_idf': (True, False),

              'clf__alpha': (0.5, 1.0)

              #'clf__min_samples_split': [2, 3]

             } '''



gs_clf = GridSearchCV(classifier, parameters)

gs_clf.fit(xtrain, ytrain)
gs_clf.best_params_




gs_clf.refit



gs_clf.fit(xtrain, ytrain)



# Predicting the validation set results

y_pred_valid = gs_clf.predict_proba(xvalid)



#predictions = clf.predict_proba(xvalid_ctv)



print ("logloss: %0.3f " % multiclass_logloss(y_pred_valid, predictions))



y_pred_valid_f1 = gs_clf.predict(xvalid)



from sklearn.metrics import classification_report

print (classification_report(yvalid, y_pred_valid_f1))



# Predicting the Test set results

y_pred_proba = gs_clf.predict_proba(X_Test)



tdf = pd.DataFrame(y_pred_proba)

tdf.columns = ['EAP', 'HPL', 'MWS']



submission = pd.read_csv('../input/spooky-author-identification/sample_submission.csv')

result = pd.concat([submission[['id']], tdf], axis=1)

result.head()



result.to_csv('naive_bayes__2.csv', index=False)
from nltk import word_tokenize



embeddings_index = {}

f = open(glove_file)

for line in tqdm(f):

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
# this function creates a normalized vector for the whole sentence

def sent2vec(s):

    #words = str(s).lower().decode('utf-8')

    words = str(s).lower()

    words = word_tokenize(words)

    words = [w for w in words if not w in stop_words]

    words = [w for w in words if w.isalpha()]

    M = []

    for w in words:

        try:

            M.append(embeddings_index[w])

        except:

            continue

    M = np.array(M)

    v = M.sum(axis=0)

    if type(v) != np.ndarray:

        return np.zeros(300)

    return v / np.sqrt((v ** 2).sum())
xtrain_glove = [sent2vec(x) for x in tqdm(xtrain)]

xvalid_glove = [sent2vec(x) for x in tqdm(xvalid)]



type(xtrain_glove)

len(xvalid_glove[1])

xvalid_glove[0].shape
xtrain_glove1 = np.array(xtrain_glove)

xvalid_glove1 = np.array(xvalid_glove)



xvalid_glove1.shape



xvalid_glove1.reshape(1, -1)

xvalid_glove1.shape
xtrain_glove.shape
#clf = xgb.XGBClassifier(nthread=10, silent=False)

clf = MultinomialNB()



#clf.fit(xtrain_glove[0], ytrain)

#predictions = clf.predict_proba(xvalid_glove[0])



clf.fit(xtrain_glove1, ytrain)

predictions = clf.predict_proba(xvalid_glove1)



print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


clf.fit(xtrain_ctv, ytrain)

predictions = clf.predict_proba(xvalid_ctv)



print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))