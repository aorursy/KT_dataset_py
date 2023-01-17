import numpy as np
import pandas as pd
import zipfile

import matplotlib.pyplot as plt
import seaborn as sns

#plotting customizations
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

plt.style.use('fivethirtyeight')
# 538 colors ['008fd5', 'fc4f30', 'e5ae38', '6d904f', '8b8b8b', '810f7c']

import sys
print("The Python version is %s.%s.%s" % sys.version_info[:3])

from sklearn.model_selection import GridSearchCV
train_zip = zipfile.ZipFile('/kaggle/input/spooky-author-identification/train.zip')
df = pd.read_csv(train_zip.open('train.csv'))
test_zip = zipfile.ZipFile('/kaggle/input/spooky-author-identification/test.zip')
test = pd.read_csv(test_zip.open('test.csv'))
sample_zip = zipfile.ZipFile('/kaggle/input/spooky-author-identification/sample_submission.zip')
sample = pd.read_csv(sample_zip.open('sample_submission.csv'))
print(df.shape)
df.head()
from sklearn.pipeline import Pipeline, FeatureUnion, make_union, make_pipeline

# call in the standard scaler and FunctionTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer

# call in libraries for class creation
from sklearn.base import BaseEstimator, TransformerMixin

# call some text libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk.tag
import string
import re
class SeriesFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column 
        
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, y=None):
        return X[self.column] 
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column 
        
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, y=None):
        return X[[self.column]]
def cleaner(text):
    stemmer = PorterStemmer()
    stop = stopwords.words('english')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.translate(str.maketrans('', '', string.digits))
    text = text.lower().strip()
    final_text = []
    for w in text.split():
        if w not in stop:
            final_text.append(stemmer.stem(w.strip()))
    return ' '.join(final_text)
def word_count(df):
    df['word_count'] = df['text'].apply(lambda x: len(re.findall(r'\w+', x))).astype(int).to_frame()
    return df

# call the function on the df to transform it
word_count(df)


# make pipeline for word count
word_count_pipe = make_pipeline(FeatureExtractor('word_count'),
                             StandardScaler()
                            )
def unique_word_prop(df):
    split_text = df['text'].apply(lambda x: x.lower().split(' '))
    split_text = split_text.apply(lambda x: [''.join(c for c in w if c not in string.punctuation) for w in x])
    word_len = split_text.apply(lambda x: len(x))
    unique_len = split_text.apply(lambda x: len(set(x)))
    df['unique_prop'] = unique_len / word_len
    return df


# call the function on the df to transform it
unique_word_prop(df)

# make the pipeline for unique word proportion
unique_word_prop_pipe =  make_pipeline(FeatureExtractor('unique_prop'),
                                      StandardScaler()
                                      )
eng_stopwords = set(stopwords.words("english"))

def stopword_prop(df):
    text = df['text']
    split_text = text.apply(lambda x: x.lower().split(' '))
    split_text = split_text.apply(lambda x: [''.join(c for c in w if c not in string.punctuation) for w in x])
    split_text = split_text.apply(lambda x: [w for w in x if w])
    word_len = split_text.apply(lambda x: len(x))
    stopwords_count = split_text.apply(lambda x: len([w for w in x if w in eng_stopwords]))
    df['stopword_prop'] = (stopwords_count / word_len)
    return df

# call the function on the df to transform it
stopword_prop(df)
    
# make the pipeline for stopword proportion
stopword_prop_pipe =  make_pipeline(FeatureExtractor('stopword_prop'),
                                      StandardScaler()
                                   )
def noun_prop(df):
    text = df['text']
    split_text = text.apply(lambda x: x.lower().split(' '))
    split_text = split_text.apply(lambda x: [''.join(c for c in w if c not in string.punctuation) for w in x])
    split_text = split_text.apply(lambda x: [w for w in x if w])
    word_len = split_text.apply(lambda x: len(x))
    pos_list = split_text.apply(lambda x: nltk.pos_tag(x))
    noun_count = pos_list.apply(lambda x: len([w for w in x if w[1] in ('NN', 'NNP', 'NNPS', 'NNS')]))
    df['noun_prop'] = (noun_count / word_len)
    return df

# call the function on the df
noun_prop(df)

# make the pipeline for noun proportion
noun_prop_pipe =  make_pipeline(FeatureExtractor('noun_prop'),
                                      StandardScaler()
                               )
def adj_prop(df):
    text = df['text']
    split_text = text.apply(lambda x: x.lower().split(' '))
    split_text = split_text.apply(lambda x: [''.join(c for c in w if c not in string.punctuation) for w in x])
    split_text = split_text.apply(lambda x: [w for w in x if w])
    word_len = split_text.apply(lambda x: len(x))
    pos_list = split_text.apply(lambda x: nltk.pos_tag(x))
    adj_count = pos_list.apply(lambda x: len([w for w in x if w[1] in ('JJ', 'JJR', 'JJS')]))
    df['adj_prop'] =(adj_count / word_len)
    return df

# call the function on the df
adj_prop(df)

# make the pipeline for adj proportion
adj_prop_pipe =  make_pipeline(FeatureExtractor('adj_prop'),
                               StandardScaler()
                              )
def verb_prop(df):
    text = df['text']
    split_text = text.apply(lambda x: x.lower().split(' '))
    split_text = split_text.apply(lambda x: [''.join(c for c in w if c not in string.punctuation) for w in x])
    split_text = split_text.apply(lambda x: [w for w in x if w])
    word_len = split_text.apply(lambda x: len(x))
    pos_list = split_text.apply(lambda x: nltk.pos_tag(x))
    verb_count = pos_list.apply(lambda x: len([w for w in x if w[1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')]))
    df['verb_prop'] = (verb_count / word_len)
    return df

# call the function on the df
verb_prop(df)

# make the pipeline for verb proportion
verb_prop_pipe =  make_pipeline(FeatureExtractor('verb_prop'),
                                StandardScaler()
                               )
from textblob import TextBlob

def polarity(text):
    corpus = TextBlob(text)
    return corpus.sentiment.polarity

#call the function on the data
df['polarity'] = df.text.map(lambda x: polarity(x))

# polarity pipeline
polarity_pipe = make_pipeline(FeatureExtractor('polarity'),
                             StandardScaler()
                             )

def subjectivity(text):
    corpus = TextBlob(text)
    return corpus.sentiment.subjectivity

df['subjectivity'] = df.text.map(lambda x: subjectivity(x))

# subjectivity pipeline
subjectivity_pipe = make_pipeline(FeatureExtractor('subjectivity'),
                                  StandardScaler()
                                 )
count_vec_pipe = make_pipeline(SeriesFeatureExtractor('text'),
                           CountVectorizer(preprocessor=cleaner))
df.head()
fu = FeatureUnion([
                   ('count_vec_pipe', count_vec_pipe)
])
fu2 = FeatureUnion([
                    ('word_count_pipe', word_count_pipe),
                    ('unique_words_pipe', unique_word_prop_pipe),
                    ('stopword_prop_pipe', stopword_prop_pipe),
                    ('noun_prop_pipe', noun_prop_pipe),
                    ('adj_prop_pipe', adj_prop_pipe),
                    ('verb_prop_pipe', verb_prop_pipe),
                    ('polarity_pipe', polarity_pipe),
                    ('subjectivity_pipe', subjectivity_pipe),
                   ('count_vec_pipe', count_vec_pipe)
])
from sklearn.model_selection import train_test_split

X = df.drop(['id', 'author'], axis=1)
target = df['author'].copy()

X_train, X_test, y_train, y_test = train_test_split(X,
                                    target,
                                    test_size=0.33,
                                    random_state = 8675309)
y_train.value_counts()
y_train.value_counts()[0]/float(y_train.shape[0])
y_test.value_counts()[0]/float(y_test.shape[0])
from sklearn.linear_model import  LogisticRegressionCV, LogisticRegression

# create the pipeline
logit_model_pipe = Pipeline([('data', fu),
                            ('logit model', LogisticRegressionCV(cv=3))
])
logit_model_pipe.fit(X_train, y_train)
logit_model_pipe.score(X_train, y_train)
logit_model_pipe.score(X_test, y_test)
from sklearn.metrics import confusion_matrix, classification_report

#generate predicted y values
y_pred = logit_model_pipe.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# define the class names
class_names = ['EAP', 'HPL', 'MWS']

# make a data frame with meaningful column and index names
confusion = pd.DataFrame(cm,
                        index=['is_EAP','is_HPL', 'is_MWS'],
                        columns=['predict_EAP','predict_HPL', 'predict_MWS'])
# call the dataframe
confusion
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_curve, auc

# get y_score for predicted probabilities
y_predict_proba = logit_model_pipe.predict_proba(X_test)

# generate fpr, tpr, and auc for each class
fpr0, tpr0, thresh0 = roc_curve(y_test.map(lambda x: 1 if x == 'EAP' else 0), y_predict_proba[:,0])
roc_auc0 = auc(fpr0, tpr0)

fpr1, tpr1, thresh1 = roc_curve(y_test.map(lambda x: 1 if x == 'HPL' else 0), y_predict_proba[:,1])
roc_auc1 = auc(fpr1, tpr1)

fpr2, tpr2, thresh2 = roc_curve(y_test.map(lambda x: 1 if x == 'MWS' else 0), y_predict_proba[:,2])
roc_auc2 = auc(fpr2, tpr2)

# plot all three together, using the appropriate color for each author
plt.figure(figsize=[8,8])
plt.plot(fpr0, tpr0, label='EAP ROC curve (area = %0.2f)' % roc_auc0, linewidth=4, color = '#008fd5')
plt.plot(fpr1, tpr1, label='HPL ROC curve (area = %0.2f)' % roc_auc1, linewidth=4, color = '#fc4f30')
plt.plot(fpr2, tpr2, label='MWS ROC curve (area = %0.2f)' % roc_auc2, linewidth=4, color = '#e5ae38')
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('ROC Curve for Each Author', fontsize=18)
plt.legend(loc="lower right")
plt.show()
logit_model_pipe2 = Pipeline([('data', fu2),
                            ('model', LogisticRegression())
])


logit_params = {
    'model__penalty': ['l2'],
    'model__C': np.logspace(-4,4,50)
}

logit_model_pipe_gs = GridSearchCV(logit_model_pipe2,
                                   param_grid = logit_params,
                                   cv =5,
                                   n_jobs=-1,
                                   verbose=1)

logit_model_pipe_gs.fit(X_train, y_train)

print('The best model params are', logit_model_pipe_gs.best_params_)
print('The best training score is: ',logit_model_pipe_gs.best_estimator_.score(X_train, y_train))
print('The test score is: ',logit_model_pipe_gs.best_estimator_.score(X_test, y_test))
df_pred = test
X_pred = df_pred.drop(['id'], axis=1)
X_pred.head()
# perform the needed feature creations
word_count(X_pred)
unique_word_prop(X_pred)
stopword_prop(X_pred)
noun_prop(X_pred)
adj_prop(X_pred)
verb_prop(X_pred)
X_pred['polarity']=X_pred.text.map(lambda x: polarity(x))
X_pred['subjectivity']=X_pred.text.map(lambda x: subjectivity(x))

# cv = CountVectorizer(preprocessor=cleaner)
# cv.fit(X_train)
# x_preds = cv.transform(X_pred)

X_pred.head()
# Using the grid search logit model
# generate predictions and put them in a dataframe
predictions = pd.DataFrame(logit_model_pipe_gs.predict_proba(X_pred),
                               columns = ['EAP','HPL', 'MWS'])
predictions['id'] = df_pred['id']
predictions = predictions[['id','EAP','HPL', 'MWS']]
predictions.head()
predictions.to_csv('classification_submission2.csv',index=False)
from sklearn.feature_extraction.text import TfidfVectorizer

tfid_pipe = make_pipeline(SeriesFeatureExtractor('text'),
                           TfidfVectorizer(preprocessor=cleaner)
                         )
# create an expanded feature union using all the text features
fu3 = FeatureUnion([
                    ('word_count_pipe', word_count_pipe),
                    ('unique_words_pipe', unique_word_prop_pipe),
                    ('stopword_prop_pipe', stopword_prop_pipe),
                    ('noun_prop_pipe', noun_prop_pipe),
                    ('adj_prop_pipe', adj_prop_pipe),
                    ('verb_prop_pipe', verb_prop_pipe),
                    ('polarity_pipe', polarity_pipe),
                    ('subjectivity_pipe', subjectivity_pipe),
                   ('tfid_pipe', tfid_pipe)
])
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

rfrs_pipe = Pipeline([('data', fu3),
                     ('model', RandomForestClassifier())
                     ])

# run randomized search
# create the parameter dictionary
rfc_params = {'model__max_depth':range(1,20),
              'model__max_features':range(1,100)
#               'model__max_leaf_nodes':range(2,10),
#               'model__min_samples_leaf': range(2,10),
#               'model__min_samples_split': range(2,10)
               }

# declare the number of model iterations
n_iter_search = 50

# create the RSCV 
rfe_rs = RandomizedSearchCV(rfrs_pipe,
                                   param_distributions=rfc_params,
                                   n_iter=n_iter_search,
                                  verbose=2,
                                  n_jobs=-1,
                                  cv = 5,
                                  random_state = 8675309)

# fit the RSCV
rfe_rs.fit(X_train, y_train)

print(rfe_rs.best_score_)
print(rfe_rs.best_params_)
print(rfe_rs.best_params_)

rfe_rs_predictions = rfe_rs.best_estimator_.predict(X_test)
rfe_rs_predictions
rfe_rs.best_estimator_.score(X_test, y_test)
fu4 = FeatureUnion([
                    ('word_count_pipe', word_count_pipe),
                    ('unique_words_pipe', unique_word_prop_pipe),
                    ('stopword_prop_pipe', stopword_prop_pipe),
                    ('noun_prop_pipe', noun_prop_pipe),
                    ('adj_prop_pipe', adj_prop_pipe),
                    ('verb_prop_pipe', verb_prop_pipe),
                    ('polarity_pipe', polarity_pipe),
                    ('subjectivity_pipe', subjectivity_pipe)
])

# create the pipeline
logit_model_pipe3 = Pipeline([('data', fu4),
                            ('model', LogisticRegression())
])


logit_params = {
    'model__penalty': ['l1','l2'],
    'model__C': np.logspace(-4,4,50)
}

logit_model_pipe_gs2 = GridSearchCV(logit_model_pipe3,
                                   param_grid = logit_params,
                                   cv =5,
                                   n_jobs=-1,
                                   verbose=1)

logit_model_pipe_gs2.fit(X_train, y_train)
print('The best model params are', logit_model_pipe_gs2.best_params_)
print('The best training score is: ',logit_model_pipe_gs2.best_estimator_.score(X_train, y_train))
print('The test score is: ',logit_model_pipe_gs2.best_estimator_.score(X_test, y_test))
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPool1D
from sklearn.preprocessing import LabelBinarizer

# needed for using keras inside a GridSearchCV and pipeline
from keras.wrappers.scikit_learn import KerasClassifier

# set up the data to use in the network
X_nn = df['text'].copy()
y_nn = df['author'].copy()

# T/T split
X_nn_train, X_nn_test, y_nn_train, y_nn_test = train_test_split(X_nn,y_nn,
                                                               test_size = .33,
                                                               random_state=8675309)
# instantiate the transformations
count_vec = CountVectorizer(preprocessor=cleaner)
lb = LabelBinarizer()

# transform the data
X_nn_train = count_vec.fit_transform(X_nn_train)
X_nn_test =  count_vec.transform(X_nn_test)
y_nn_train = lb.fit_transform(y_nn_train)
y_nn_test = lb.transform(y_nn_test)
print(X_nn_train.shape)
print(y_nn_train.shape)
print(X_nn_test.shape)
print(y_nn_test.shape)
def model_func(input_dim = X_nn_train.shape[1],
               layer_one_neurons = X_nn_train.shape[1],
               layer_two_neurons = 500,
               layer_three_neurons = 250,
               layer_four_neurons = 100,
               layer_five_neurons = 15):
    
    # instantiate the model
    model = Sequential()
    
    # create an input layer
    model.add(Dense(layer_one_neurons,
                   input_dim = input_dim,
                   activation = 'relu'))
    
    # create a dropout layer
    model.add(Dropout(.33))
    
    # create a hidden layer
    model.add(Dense(layer_two_neurons,
                   activation = 'relu'))
    
    # create a hidden layer
    model.add(Dense(layer_three_neurons,
                   activation = 'relu'))
    
    # create a dropout layer
    model.add(Dropout(.33))
    
    # create a hidden layer
    model.add(Dense(layer_four_neurons,
                   activation = 'relu'))
    
    # create a dropout layer
    model.add(Dropout(.33))
    
    # create a hidden layer
    model.add(Dense(layer_five_neurons,
                   activation = 'relu'))
    
    # create the output layer
    # categorical classifier so neuron is # of columns in y
    model.add(Dense(y_nn_train.shape[1], activation='softmax'))
    
    # compile
    model.compile(loss = 'categorical_crossentropy',
                 optimizer = 'adam',
                 metrics = ['accuracy'])
    
    #return the model created
    return model

# ------------------------------------------------
# set up the grid search

# the model
ff_model = KerasClassifier(build_fn = model_func,
                           input_dim = X_nn_train.shape[1],
                          verbose = 1)

history = ff_model.fit(X_nn_train.todense(), y_nn_train,
         validation_data = (X_nn_test.todense(), y_nn_test),
         epochs = 10,
          batch_size = 200)
plt.plot(history.history['val_acc'], label='val acc')
plt.plot(history.history['acc'], label='train acc')
plt.xlabel('epochs')
plt.legend()
ff_model.score(X_nn_test.todense(), y_nn_test)
from keras.layers import GlobalAveragePooling1D

input_dim = X_nn_train.shape[1]
embedding_dims = 200

def create_model(embedding_dims=200,
                 optimizer='adam'):
    
    model = Sequential()
    
    model.add(Embedding(input_dim=input_dim,
                        output_dim=embedding_dims))
    
    model.add(GlobalAveragePooling1D())
    
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model


model = create_model()
history = model.fit(X_nn_train.todense(), y_nn_train,
                 batch_size=100,
                 validation_data=(X_nn_test.todense(), y_nn_test),
                 epochs=10)
plt.plot(history.history['val_acc'], label='val acc')
plt.plot(history.history['acc'], label='train acc')
plt.xlabel('epochs')
plt.legend()
names = ['Baseline',
        'Base Logit',
        'Full Logit',
        'RFC',
        'Neural Net',
        'CNN']
train = [0.399,
         0.965,
         0.958,
         0.550,
         0.997,
         0.399]
test = [0.412,
       0.807,
       0.811,
       0.553,
       0.795,
       0.412]

accuracy_dct = dict(zip(names, train))

accuracy_df = pd.DataFrame(list(accuracy_dct.items()), columns = ['names','train'])
accuracy_df['test'] = test
print(accuracy_df)

# plot as bar chart

names = accuracy_df['names']

# sort importances
indices = np.argsort(accuracy_df['test'])

# plot as bar chart
fig = plt.figure(figsize=(8, 8))
plt.barh(np.arange(len(names)), accuracy_df['train'][indices], alpha = .3,color = '#6d904f', label = 'train')
plt.barh(np.arange(len(names)), accuracy_df['test'][indices], alpha = .3, color = '#810f7c', label = 'test')
plt.yticks(np.arange(len(names)), np.array(names)[indices], size=12)
plt.xticks(size=12)
_ = plt.ylabel('Model')
_ = plt.xlabel('Accuracy')
_ = plt.title('Visualizing Model Performance')
plt.axvline(x=0.399, color='black', linestyle='--', lw=2)
plt.text(0.45, 0.1, 'Baseline',
        color='black', fontsize=13)
plt.legend(loc=(.78, .125), fontsize=14)