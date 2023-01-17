import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers.recurrent import LSTM,GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection,metrics,pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from keras.layers import GlobalMaxPooling1D,Conv1D,MaxPooling1D,Flatten,Bidirectional,SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from nltk import word_tokenize
from nltk.corpus import stopwords
PATH = '../input/spooky-author-identification'
train = pd.read_csv(f'{PATH}/train.zip')
test = pd.read_csv(f'{PATH}/test.zip')
sample = pd.read_csv(f'{PATH}/sample_submission.zip')
train.head()
test.head()
sample.head()
def multiclass_logloss(actual,predicted,eps=1e-15):
    
    #converting the 'actual' values to binary values if it's 
    #not binary values
    
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0],predicted.shape[1]))
        
        for i, val in enumerate(actual):
            actual2[i,val] = 1
        actual = actual2
    
    #clip function truncates the number between
    #a max number and min number
    clip = np.clip(predicted,eps,1-eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0/ rows * vsota 
encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(train["author"].values)
# we will use 10% of data for testing
X_train, X_test, y_train, y_test = train_test_split(train.text.values,y,random_state=42,test_size=0.1,shuffle=True)
from sklearn.feature_extraction.text import CountVectorizer
#we are going to use this example as our documents.

cat_in_the_hat_docs=[
       "One Cent, Two Cents, Old Cent, New Cent: All About Money (Cat in the Hat's Learning Library",
       "Inside Your Outside: All About the Human Body (Cat in the Hat's Learning Library)",
       "Oh, The Things You Can Do That Are Good for You: All About Staying Healthy (Cat in the Hat's Learning Library)",
       "On Beyond Bugs: All About Insects (Cat in the Hat's Learning Library)",
       "There's No Place Like Space: All About Our Solar System (Cat in the Hat's Learning Library)" 
      ]

#make object of countvectorizer
cv = CountVectorizer()
count_vector = cv.fit_transform(cat_in_the_hat_docs)
#now let's look at the  unique words countvectorizer was able to find
cv.vocabulary_
count_vector.shape
#using cumstom stopword list
custom_stop_words = ["all","in","the","is","and"]

cv = CountVectorizer(cat_in_the_hat_docs,stop_words=custom_stop_words)
count_vector = cv.fit_transform(cat_in_the_hat_docs)
count_vector.shape
#have a look at the stop words
cv.stop_words
cv = CountVectorizer(cat_in_the_hat_docs,min_df=2) #word that has occur in only one document
count_vector = cv.fit_transform(cat_in_the_hat_docs)

#now let's look at stop words
cv.stop_words_
#see the difference of _ at the end it's because we used min_df 
#instead of custom stop_words
#using max_df
cv = CountVectorizer(cat_in_the_hat_docs,max_df=0.5) #present in more than 50% of documents
count_vector = cv.fit_transform(cat_in_the_hat_docs)

cv.stop_words_
#using Tfidtrasnformer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

#we will use this toy example
docs=["the house had a tiny little mouse",
      "the cat saw the mouse",
      "the mouse ran away from the house",
      "the cat finally ate the mouse",
      "the end of the mouse story"
     ]

cv = CountVectorizer(docs,max_df=0.5)

count_vector = cv.fit_transform(docs)
print(count_vector.shape)

#calculate idf values
tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(count_vector)

df_idf = pd.DataFrame(tfidf_transformer.idf_,index=cv.get_feature_names(),columns=["idf_weights"])
df_idf

#using tfid_vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 

tfidf_vectorizer = TfidfVectorizer(smooth_idf=True,use_idf=True)
tfidf_vectorizer.fit_transform(docs)

#as you can see we don't need CountVectorizer in TfidfVectorizer

df_idf = pd.DataFrame(tfidf_vectorizer.idf_,index=tfidf_vectorizer.get_feature_names(),columns=["idf_weights"])
df_idf
# we can also pass countvectorizer parameters in TfidVectorizer
tfv = TfidfVectorizer(min_df=3,max_features=None,strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',
                      ngram_range=(1,3),use_idf=1,smooth_idf=1,stop_words='english')

# max_features confines maximum number of words 

tfv.fit(list(X_train) + list(X_test))
X_train_tfv = tfv.transform(X_train)
X_test_tfv = tfv.transform(X_test)



# Fitting Logistic Regression on TFIDF
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1.0)
clf.fit(X_train_tfv,y_train)
prediction = clf.predict_proba(X_test_tfv)

print("logloss: %0.3f" % multiclass_logloss(y_test,prediction))
ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english')

ctv.fit(list(X_train)+list(X_test))
X_train_ctv = ctv.transform(X_train)
X_test_ctv = ctv.transform(X_test)
clf = LogisticRegression(C=1.0)
clf.fit(X_train_ctv,y_train)
prediction = clf.predict_proba(X_test_ctv)

print("logloss: %0.3f" % multiclass_logloss(y_test,prediction))
clf = MultinomialNB()
clf.fit(X_train_tfv,y_train)

prediction = clf.predict_proba(X_test_tfv)

print("logloss: %0.3f" % multiclass_logloss(y_test,prediction))
clf = MultinomialNB()
clf.fit(X_train_ctv,y_train)

prediction = clf.predict_proba(X_test_ctv)

print("logloss: %0.3f" % multiclass_logloss(y_test,prediction))
svd = decomposition.TruncatedSVD(n_components=120)
svd.fit(X_train_tfv)
X_train_svd = svd.transform(X_train_tfv)
X_test_svd = svd.transform(X_test_tfv)

scl = preprocessing.StandardScaler()
scl.fit(X_train_svd)

X_train_svd_scl = scl.transform(X_train_svd)
X_test_svd_scl = scl.transform(X_test_svd)
svm = SVC(C=1.0,probability=True)

svm.fit(X_train_svd_scl,y_train)
prediction = svm.predict_proba(X_test_svd_scl)

print("logloss: %0.3f" % multiclass_logloss(y_test,prediction))
clf = xgb.XGBClassifier(max_depth=7,n_estimators=200,colsample_bytree=0.8,subsample=0.8,nthread=10,learning_rate=0.1)

clf.fit(X_train_tfv.tocsc(),y_train)
prediction = clf.predict_proba(X_test_tfv.tocsc())

print("logloss: %0.3f" % multiclass_logloss(y_test,prediction))
clf = xgb.XGBClassifier(max_depth=7,n_estimators=200,colsample_bytree=0.8,subsample=0.8,nthread=10,learning_rate=0.1)

clf.fit(X_train_svd,y_train)
prediction = clf.predict_proba(X_test_svd)

print("logloss: %0.3f" % multiclass_logloss(y_test,prediction))
# as Multiclass_logloss is user defined we need to define our own scorer for grid search
# greater_is_better is True by default but for our smaller the value of logloss better the result

mll_scorer = metrics.make_scorer(multiclass_logloss, greater_is_better=False,needs_proba=True)
svd = decomposition.TruncatedSVD()

scl = preprocessing.StandardScaler()

lr_model = LogisticRegression()

clf = pipeline.Pipeline([('svd',svd),
                         ('scl',scl),
                         ('lr',lr_model)])
params_grid = {'svd__n_components':[120,180],
               'lr__C':[0.1,1.0,10],
               'lr__penalty':['l1','l2']}
model = GridSearchCV(estimator=clf,param_grid=params_grid,scoring=mll_scorer,verbose=10,n_jobs=-1,iid=True,refit=True,cv=2)

#fitting the model
model.fit(X_train_tfv,y_train)

print('Best score: %0.3f' % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(params_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
nb = MultinomialNB()

clf = pipeline.Pipeline([('nb',nb)])

params_grid = {'nb__alpha':[0.001,0.01,0.1,1,10,100]}

model  = GridSearchCV(estimator=clf,param_grid=params_grid,scoring=mll_scorer,verbose=10,n_jobs=-1,refit=True,cv=2)

model.fit(X_train_tfv,y_train)

print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(params_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))