import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

from fuzzywuzzy import fuzz

from tqdm import tqdm



from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 



import xgboost as xgb



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC
data = pd.read_csv("/kaggle/input/60k-stack-overflow-questions-with-quality-rate/data.csv")

data.head()
data = data.drop(['Tags', 'CreationDate'], axis=1)

data['Y'] = data['Y'].map({'LQ_CLOSE':0, 'LQ_EDIT': 1, 'HQ':2})

data.head()

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^(a-zA-Z)\s]','', text)

    return text



data['Body'] = data['Body'].apply(clean_text)

data['Title'] = data['Title'].apply(clean_text)

data.head()
stop_words = set(stopwords.words('english')) 



def remove_stopword(words):

    list_clean = [w for w in words.split(' ') if not w in stop_words]

    

    return ' '.join(list_clean)



def remove_next_line(words):

    words = words.split('\n')

    

    return " ".join(words)



def remove_r_char(words):

    words = words.split('\r')

    

    return "".join(words)

    

data['Body'] = data['Body'].apply(remove_stopword)

data['Body'] = data['Body'].apply(remove_next_line)

data['Body'] = data['Body'].apply(remove_r_char)



data['Title'] = data['Title'].apply(remove_stopword)

data['Title'] = data['Title'].apply(remove_next_line)

data['Title'] = data['Title'].apply(remove_r_char)

data.head()
distribution = data.groupby('Y')['Body'].count().reset_index()
distribution
data['Num_words_body'] = data['Body'].apply(lambda x:len(str(x).split())) #Number Of words in Selected Text

data['Num_words_title'] = data['Title'].apply(lambda x:len(str(x).split())) #Number Of words in main text

data['difference_in_words'] = abs(data['Num_words_body'] - data['Num_words_title']) #Difference in Number of words text and Selected Text
data['Num_char_body'] = data['Body'].apply(lambda x:len("".join(set(str(x).replace(" ",""))))) 

data['Num_char_title'] = data['Title'].apply(lambda x:len("".join(set(str(x).replace(" ","")))))
data['len_common_words'] = data.apply(lambda x:len(set(str(x['Title']).split()).intersection(set(str(x['Body']).split()))),axis=1)
data.head(3)
data['fuzz_qratio'] = data.apply(lambda x:fuzz.QRatio(str(x['Title']),str(x['Body'])), axis=1)

data['fuzz_Wratio'] = data.apply(lambda x:fuzz.WRatio(str(x['Title']),str(x['Body'])), axis=1)

data['fuzz_partial_ratio'] = data.apply(lambda x:fuzz.partial_ratio(str(x['Title']),str(x['Body'])), axis=1)

data['fuzz_partial_token_set_ratio'] = data.apply(lambda x:fuzz.partial_token_set_ratio(str(x['Title']),str(x['Body'])), axis=1)

data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x:fuzz.partial_token_sort_ratio(str(x['Title']),str(x['Body'])), axis=1)

data['fuzz_token_set_ratio'] = data.apply(lambda x:fuzz.token_set_ratio(str(x['Title']),str(x['Body'])), axis=1)

data['fuzz_token_sort_ratio'] = data.apply(lambda x:fuzz.token_sort_ratio(str(x['Title']),str(x['Body'])), axis=1)
data.head(3)
data['Body_with_title'] = data['Title'] + " " + data['Body']
xtrain, xtest, ytrain, ytest = train_test_split(data.drop(['Id','Title','Body','Y'],axis=1).values, data['Y'].values, 

                                                  stratify=data['Y'].values, 

                                                  random_state=42, 

                                                  test_size=0.2, shuffle=True)
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, 

                                                  stratify=ytrain, 

                                                  random_state=42, 

                                                  test_size=0.2, shuffle=True)
len(ytrain)
len(yvalid)
tfv = TfidfVectorizer(min_df=3,  max_features=None, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english')
tfv.fit(list(xtrain[:,-1]))

xtrain_tfv =  tfv.transform(xtrain[:,-1]) 

xvalid_tfv = tfv.transform(xvalid[:,-1])
ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), stop_words = 'english')



# Fitting Count Vectorizer to both training and test sets (semi-supervised learning)

ctv.fit(xtrain[:,-1])

xtrain_ctv =  ctv.transform(xtrain[:,-1]) 

xvalid_ctv = ctv.transform(xvalid[:,-1])

# Fitting a simple Logistic Regression on TFIDF

clf = LogisticRegression()

clf.fit(xtrain_tfv, ytrain)

predictions = clf.predict(xvalid_tfv)
clf_ctv = LogisticRegression()

clf_ctv.fit(xtrain_ctv, ytrain)

predictions_ctv = clf_ctv.predict(xvalid_ctv)
def get_accuracy(clf, predictions, yvalid):

    return np.mean(predictions == yvalid)
from sklearn.metrics import multilabel_confusion_matrix

multilabel_confusion_matrix(yvalid, predictions)

multilabel_confusion_matrix(yvalid, predictions_ctv)
get_accuracy(clf, predictions, yvalid)
get_accuracy(clf_ctv, predictions_ctv, yvalid)
clf = xgb.XGBClassifier(max_depth=10, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1, silent=False)

clf.fit(xtrain_tfv, ytrain)

predictions = clf.predict(xvalid_tfv)
get_accuracy(clf, predictions, yvalid)
clf_ctv = xgb.XGBClassifier(max_depth=10, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

clf_ctv.fit(xtrain_ctv, ytrain)

predictions_ctv = clf_ctv.predict(xvalid_ctv)
get_accuracy(clf_ctv, predictions_ctv, yvalid)
clf = MultinomialNB()

clf.fit(xtrain_tfv, ytrain)

predictions = clf.predict(xvalid_tfv)

get_accuracy(clf, predictions, yvalid)
clf_ctv = MultinomialNB()

clf_ctv.fit(xtrain_ctv, ytrain)

predictions_ctv = clf_ctv.predict(xvalid_ctv)
get_accuracy(clf_ctv, predictions_ctv, yvalid)
# Apply SVD, I chose 120 components. 120-200 components are good enough for SVM model.

svd = decomposition.TruncatedSVD(n_components=180)

svd.fit(xtrain_tfv)

xtrain_svd = svd.transform(xtrain_tfv)

xvalid_svd = svd.transform(xvalid_tfv)



# Scale the data obtained from SVD. Renaming variable to reuse without scaling.

scl = preprocessing.StandardScaler()

scl.fit(xtrain_svd)

xtrain_svd_scl = scl.transform(xtrain_svd)

xvalid_svd_scl = scl.transform(xvalid_svd)
clf = SVC(C=1.0) # since we need probabilities

clf.fit(xtrain_svd_scl, ytrain)
predictions = clf.predict(xvalid_svd_scl)
get_accuracy(clf, predictions, yvalid)
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

clf.fit(xtrain_svd, ytrain)

predictions = clf.predict(xvalid_svd)
get_accuracy(clf, predictions, yvalid)
mll_scorer = metrics.make_scorer(get_accuracy, greater_is_better=True, needs_proba=False)
svd = TruncatedSVD()

    

# Initialize the standard scaler 

scl = preprocessing.StandardScaler()



# We will use logistic regression here..

xg_model = xgb.XGBClassifier()



# Create the pipeline 

clf = pipeline.Pipeline([('svd', svd),

                         ('scl', scl),

                         ('xg', xg_model)])
param_grid = {'svd__n_components' : [120, 150, 180],

              'xg__max_depth':[5,7,10],

              'xg__learning_rate':[0.1,0.01,0.5]}

def read_glove_vecs(glove_file):

    #input: file

    #output: word to 200d vector mapping output

    with open(glove_file, 'r') as f:

        words = set()

        word_to_vec_map = {}

        for line in f:

            line = line.strip().split()

            curr_word = line[0]

            words.add(curr_word)

            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

    return word_to_vec_map

#word_to_vec_map = read_glove_vecs('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt')

word_to_vec_map = read_glove_vecs('../input/glovetwitter27b100dtxt/glove.twitter.27B.200d.txt')
def prepare_sequence(ds, word_to_vec_map):

    #input: Series, and word_to_vec_map of size(vocab_size,200)

    #output: returns shape of (len(ds), 200)

    traintest_X = []

    for sentence in tqdm(ds):

        sequence_words = np.zeros((word_to_vec_map['cucumber'].shape))

        for word in sentence.split():

            if word in word_to_vec_map.keys():

                temp_X = word_to_vec_map[word]

            else:

                temp_X = word_to_vec_map['#']

            #print(temp_X)

            sequence_words+=(temp_X)/len(sentence)

            #print(sequence_words)

        traintest_X.append(sequence_words)

    return np.array(traintest_X)

prepare_sequence(xtrain[:,-1][0], word_to_vec_map)
#concatenate all sequences for training and testing set

train_w2v = prepare_sequence(xtrain[:,-1], word_to_vec_map)

valid_w2v = prepare_sequence(xvalid[:,-1], word_to_vec_map)
clf = LogisticRegression()

clf.fit(train_w2v, ytrain)

predictions = clf.predict(valid_w2v)
get_accuracy(clf, predictions, yvalid)
clf = xgb.XGBClassifier(max_depth=10, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1, silent=False)

clf.fit(train_w2v, ytrain)

predictions = clf.predict(valid_w2v)

get_accuracy(clf, predictions, yvalid)
clf2 = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1, silent=False)

clf2.fit(train_w2v, ytrain)

predictions = clf2.predict(valid_w2v)
get_accuracy(clf2, predictions, yvalid)