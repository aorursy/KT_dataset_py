####importing necessary python modules

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')   ####performing preprocessing for filtering out useless data i.e “the”, “a”, “an”, “in” 

####loading dataset from kaggle notebook directory
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


####prediction target
ytrain = train.CLASS.values 

####predictors
xtrain=train.CONTENT.values

#### test set
xvalid=test.CONTENT.values


#### TF-IDF (Term Frequency - Inverse Document Frequency)
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

#### Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfv.fit(list(xtrain) + list(xvalid))
xtrain_tfv =  tfv.transform(xtrain) 
xvalid_tfv = tfv.transform(xvalid)


#### Fitting a simple xgboost on tf-idf
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(xtrain_tfv.tocsc(), ytrain)
predictions = clf.predict(xvalid_tfv)

print(predictions)


#### Exporting files for submission
my_submission=pd.DataFrame({'ID':test.ID,'CLASS':predictions})
my_submission.to_csv('submission.csv',index=False)

