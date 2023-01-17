import pandas as pd
import numpy as np
from sklearn.model_selection import *
from sklearn.ensemble import *
from sklearn.metrics import *
import seaborn as sns
%pylab inline
trainDf = pd.read_json("../input/train.json")
trainDf['ingredients'] = trainDf['ingredients'].map(lambda x: str(x)[1:-1].split(','))
trainDf['ingredients_clean_string'] = [' , '.join(z).strip() for z in trainDf['ingredients']]  
from nltk.stem import WordNetLemmatizer 
import re
trainDf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in trainDf['ingredients']]       
testDf = pd.read_json("../input/test.json") 
testDf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testDf['ingredients']]
testDf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testDf['ingredients']]       
corpusStr = trainDf['ingredients_string']
from sklearn.feature_extraction.text import TfidfVectorizer
tfIdfVect = TfidfVectorizer(stop_words='english', 
ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
tfIdfVect.fit_transform(corpusStr).todense()
corpusts = testDf['ingredients_string']
vectorizerts = TfidfVectorizer(stop_words='english')
tfidfts=tfIdfVect.transform(corpusts)
predictors_tr = tfIdfVect
targets_tr = trainDf['cuisine']
predictors_ts = tfidfts
from sklearn.linear_model import LogisticRegression
parameters = {'C':[1,10]}
clf = LogisticRegression()
from sklearn import grid_search
classif = grid_search.GridSearchCV(clf, parameters)
classif
classif.fit(predictors_tr,targets_tr)
