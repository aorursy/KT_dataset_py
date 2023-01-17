import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import nltk

import string

from nltk.stem.porter import *

from collections import Counter

from sklearn import model_selection, preprocessing, ensemble

from scipy import sparse

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df=pd.read_json(open("../input/train.json","r"))

dft = pd.read_json(open("../input/test.json", "r"))



df['features1']=df['features'].apply(lambda x: " ".join([" ".join(i.split(" "))for i in x]) )

#print(df[df.features1.str.contains("_24")].iloc[0,-1])

df['fea_des']=df['description']+df['features1']

dft['features1']=dft['features'].apply(lambda x: " ".join([" ".join(i.split(" "))for i in x]) )

#print(df[df.features1.str.contains("_24")].iloc[0,-1])

dft['fea_des']=dft['description']+dft['features1']



def get_tokens(shakes):

    text = shakes

    lowers = text.apply(lambda x: x.lower())

    #remove the punctuation using the character deletion step of translate

    no_punctuation = lowers.apply(lambda x: x.translate(str.maketrans('','',string.punctuation)))

    tokens = no_punctuation.apply(lambda x: nltk.word_tokenize(x))

    return tokens





df['fea_des_token'] = get_tokens(df['fea_des'])

dft['fea_des_token'] = get_tokens(dft['fea_des'])



def stem_tokens(tokens, stemmer):

    stemmed = []

    for item in tokens:

        stemmed.append(stemmer.stem(item))

    return stemmed



stemmer = PorterStemmer()

df['fea_des_stem']= df['fea_des_token'].apply(lambda x: stem_tokens(x, stemmer))

dft['fea_des_stem']= dft['fea_des_token'].apply(lambda x: stem_tokens(x, stemmer))



df['fea_des_stem1']=df['fea_des_stem'].apply(lambda x: " ".join(x))

dft['fea_des_stem1']=dft['fea_des_stem'].apply(lambda x: " ".join(x))





tfidf = CountVectorizer(stop_words='english', max_features=200)

tr_sparse = tfidf.fit_transform(df['fea_des_stem1'])

te_sparse = tfidf.transform(dft["fea_des_stem1"])



df["num_photos"] = df["photos"].apply(len)

df["num_features"] = df["features"].apply(len)

df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))

df["created"] = pd.to_datetime(df["created"])

df["created_year"] = df["created"].dt.year

df["created_month"] = df["created"].dt.month

df["created_day"] = df["created"].dt.day





feats=["bathrooms","bedrooms","latitude","longitude","price","num_photos", "num_features", "num_description_words",

             "created_year", "created_month", "created_day"]

X=df[feats]

y=df["interest_level"]

#add in manager_id and building_id

categorical = ["manager_id", "building_id"]

for f in categorical:

        if df[f].dtype=='object':

            #print(f)

            lbl = preprocessing.LabelEncoder()

            lbl.fit(list(df[f].values) + list(dft[f].values))

            df[f] = lbl.transform(list(df[f].values))

            dft[f] = lbl.transform(list(dft[f].values))

            feats.append(f)

            

#prepare test data        

dft["num_photos"] = dft["photos"].apply(len)

dft["num_features"] = dft["features"].apply(len)

dft["num_description_words"] = dft["description"].apply(lambda x: len(x.split(" ")))

dft["created"] = pd.to_datetime(dft["created"])

dft["created_year"] = dft["created"].dt.year

dft["created_month"] = dft["created"].dt.month

dft["created_day"] = dft["created"].dt.day

Xt = dft[feats]

#####



X1 = sparse.hstack([df[feats], tr_sparse]).tocsr()

test_X=sparse.hstack([dft[feats],te_sparse]).tocsr()

target_num_map = {'high':0, 'medium':1, 'low':2}

y1 = np.array(df['interest_level'].apply(lambda x: target_num_map[x]))
seed_val=2017

num_rounds=10

param = {}

#param['objective'] = 'multi:softprob'

#param['learning_rate'] = 0.1

#param['max_depth'] = 6

#param['silent'] = 0

#param['num_class'] = 3

#param['eval_metric'] = "mlogloss"

param['min_child_weight'] = 1

#param['subsample'] = 0.8

#param['colsample_bytree'] = 0.8

#param['early_stopping_rounds']=  20

param['seed'] = seed_val

num_rounds = num_rounds

base_xgb=xgb.XGBClassifier(**param)

bb_xgb=xgb.train()







param_grid = {'subsample': [ 0.8], 'colsample_bytree': [0.8], 'max_depth':[6], 'early_stopping_rounds':[10]}

model = GridSearchCV(estimator=bb_xgb, param_grid=param_grid, n_jobs=1, cv=2, verbose=25)

#model.fit(X1, y1)
a=model.fit(X1,y1)