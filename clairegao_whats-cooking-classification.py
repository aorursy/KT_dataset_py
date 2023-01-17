# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Data Cleanning 
import json
import pandas as pd
import numpy as np
from collections import Counter


# Model 
from scipy import stats
from math import sqrt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

import os
print(os.listdir("../input"))
Train_data = pd.read_json('../input/train.json')
Test_data = pd.read_json('../input/test.json')
# Transfer list of dictionaries to Dataframe
Train_Raw = pd.DataFrame.from_dict(Train_data)
Train_Raw.head()
def Targetencoding(df):
    df["cuisine"].replace(["italian"], 1, inplace=True)
    df["cuisine"].replace(["mexican"], 2, inplace=True)
    df["cuisine"].replace(["southern_us"], 3, inplace=True)
    df["cuisine"].replace(["indian"], 4, inplace=True)
    df["cuisine"].replace(["chinese"], 5, inplace=True)
    df["cuisine"].replace(["french"], 6, inplace=True)
    df["cuisine"].replace(["cajun_creole"], 7, inplace=True)
    df["cuisine"].replace(["thai"], 8, inplace=True)
    df["cuisine"].replace(["japanese"], 9, inplace=True)
    df["cuisine"].replace(["greek"], 10, inplace=True)
    
    df["cuisine"].replace(["spanish"], 11, inplace=True)
    df["cuisine"].replace(["korean"], 12, inplace=True)
    df["cuisine"].replace(["vietnamese"], 13, inplace=True)
    df["cuisine"].replace(["moroccan"], 14, inplace=True)
    df["cuisine"].replace(["british"], 15, inplace=True)
    df["cuisine"].replace(["filipino"], 16, inplace=True)
    df["cuisine"].replace(["irish"], 17, inplace=True)
    df["cuisine"].replace(["jamaican"], 18, inplace=True)
    df["cuisine"].replace(["russian"], 19, inplace=True)
    df["cuisine"].replace(["brazilian"], 20, inplace=True)
    
    return df

Train_Raw = Targetencoding(Train_Raw)
Train_Raw.head()
Unique_Wordlist = sorted(list(set([element.lower().split(" ")[-1].replace('(','').replace(')','') for element in np.unique(np.hstack(Train_Raw.ingredients)).tolist()])))[1:]
Unique_Wordlist[:3]
def dataPreprocessor(df,k):
    # create stop word dictionary
    stop = ['sauce','mix','powder','paste']
    
    counter = Counter()
    counter.update([word.lower().split(" ")[-1] for word in np.hstack(df.ingredients).tolist() if word.lower().split(" ")[-1] not in stop])

    topk = counter.most_common(k)
    test = []
    
    for i in range(len(df)):
        tempCounter = Counter([word.lower().split(" ")[-1] for word in df.ingredients[i] if word.lower().split(" ")[-1] not in stop])
        topkinDoc = [tempCounter[word] if tempCounter[word] > 0 else 0 for (word,wordCount) in topk]
        
        test.append([df.id[i]]+[df.cuisine[i]]+topkinDoc)    # [df.id[i]]+
    
    data = pd.DataFrame(test)
    dfName = []
    for c in topk:
        dfName.append(c[0])
        
    data.columns = ['id','target'] + dfName     # 'id',
    return topk, data
# Data cleanning - transfer list of text in the "ingredients" column to boolean representation 

Train_word,clean_traindata = dataPreprocessor(Train_Raw,k=160)
clean_traindata.head()
clean_traindata.groupby('target')['id'].nunique().sort_values(ascending=False)

# Check, 20 types of dishes
Test_Raw = pd.DataFrame.from_dict(Test_data)
Test_Raw.head()
def dataPreprocessor2(df,k):
    # create stop word dictionary
    stop = ['sauce','mix','powder','paste']
    
    counter = Counter()
    counter.update([word.lower().split(" ")[-1] for word in np.hstack(df.ingredients).tolist() if word.lower().split(" ")[-1] not in stop])

    topk = counter.most_common(k)
    test = []
    
    for i in range(len(df)):
        tempCounter = Counter([word.lower().split(" ")[-1] for word in df.ingredients[i] if word.lower().split(" ")[-1] not in stop])

        topkinDoc = [tempCounter[word] if tempCounter[word] > 0 else 0 for (word,wordCount) in topk]
        test.append([df.id[i]]+topkinDoc)
    
    data = pd.DataFrame(test)
    dfName = []
    for c in topk:
        dfName.append(c[0])
        
    data.columns = ['id'] + dfName     
    return topk, data
Test_word,clean_testdata = dataPreprocessor2(Test_Raw,k=160)
clean_testdata.head()
# #Confidence Interval Function

# def mean_confidence_interval(data, confidence=0.95):
#     a = 1.0*np.array(data)
#     n = len(a)
#     mu,sd = np.mean(a),np.std(a)
#     z = stats.t.ppf(confidence, n)
#     h=z*sd/sqrt(n)
#     return mu, h
# def featureSizeAC(data, num_run, **params):

#     feature_precentage = np.linspace(0.1, 1, 10, endpoint=True)
    
#     columnsize = len(data.columns)-2
#     train_scores = []
#     test_scores = []
#     train_mean_fs = []
#     train_ci_fs = []
#     test_mean_fs = []
#     test_ci_fs = []
    
#     classifier = KNeighborsClassifier(n_neighbors=int(len(data)/20))
        
#     for i in range(len(feature_precentage)):
#         sliceindex = int(columnsize*feature_precentage[i])
#         features_df = data.iloc[:,2:sliceindex]
#         features = features_df.as_matrix()
#         target_df = data['target']
#         target = target_df.as_matrix()

#         for j in range(num_run):
#             features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.25, stratify = target)
            
#             clfModel = classifier.fit(features_train, target_train)
#             train_target_pred = clfModel.predict(features_train)
#             test_target_pred = clfModel.predict(features_test)

#             train_scores.append(metrics.accuracy_score(target_train, train_target_pred))
#             test_scores.append(metrics.accuracy_score(target_test, test_target_pred))      
    
#         train_mean,train_ci = mean_confidence_interval(train_scores)
#         test_mean,test_ci = mean_confidence_interval(test_scores) 
#         train_mean_fs.append(train_mean)
#         train_ci_fs.append(train_ci)
#         test_mean_fs.append(test_mean)
#         test_ci_fs.append(test_ci)

#     return train_mean_fs, train_ci_fs, test_mean_fs, test_ci_fs
# train_mean_fs, train_ci_fs, test_mean_fs, test_ci_fs = featureSizeAC(clean_traindata, 1, c=1.0)

# print("Train\
#     \nAverage Accuracy: {0} \
#     \nConfidence Interval: {1}\n".format(train_mean_fs, train_ci_fs)
#      )

# print("Test\
#     \nAverage Accuracy: {0} \
#     \nConfidence Interval: {1}".format(test_mean_fs, test_ci_fs)
#      )
## Tune Model 

# model = KNeighborsClassifier()

# features_df = clean_traindata.iloc[:,2:]
# features = features_df.as_matrix()
# target_df = clean_traindata['target']
# target = target_df.as_matrix()
# features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.25, stratify = target)

# params = {'n_neighbors':[int(len(clean_traindata)/20),int(len(clean_traindata)/40)]}

# model1 = GridSearchCV(model, param_grid=params, n_jobs=1)

# model1.fit(features_train,target_train)

# print("Best Hyper Parameters:\n",model1.best_params_)
# classifier = KNeighborsClassifier(n_neighbors= int(len(clean_traindata)/20))

# test_features = clean_testdata.iloc[:,1:]

# clfModel = classifier.fit(features, target)
# test_target_pred = clfModel.predict(test_features)
# Without tuning

classifier = KNeighborsClassifier(n_neighbors= int(len(clean_traindata)/20))
        
features_df = clean_traindata.iloc[:,2:]
features = features_df.as_matrix()
target_df = clean_traindata['target']
target = target_df.as_matrix()

test_features = clean_testdata.iloc[:,1:]

clfModel = classifier.fit(features, target)
test_target_pred = clfModel.predict(test_features)
output = pd.DataFrame(test_target_pred, columns = ['cuisine'])
output['id'] = clean_testdata['id'] 
output = output[['id','cuisine']]
output.head()
def Targetencoding_back(df):
    df["cuisine"].replace(1, "italian", inplace=True)
    df["cuisine"].replace(2, "mexican", inplace=True)
    df["cuisine"].replace(3, "southern_us", inplace=True)
    df["cuisine"].replace(4, "indian", inplace=True)
    df["cuisine"].replace(5, "chinese", inplace=True)
    df["cuisine"].replace(6, "french", inplace=True)
    df["cuisine"].replace(7, "cajun_creole", inplace=True)
    df["cuisine"].replace(8, "thai", inplace=True)
    df["cuisine"].replace(9, "japanese", inplace=True)
    df["cuisine"].replace(10, "greek", inplace=True)
    
    df["cuisine"].replace(11, "spanish", inplace=True)
    df["cuisine"].replace(12, "korean", inplace=True)
    df["cuisine"].replace(13, "vietnamese", inplace=True)
    df["cuisine"].replace(14, "moroccan", inplace=True)
    df["cuisine"].replace(15, "british", inplace=True)
    df["cuisine"].replace(16, "filipino", inplace=True)
    df["cuisine"].replace(17, "irish", inplace=True)
    df["cuisine"].replace(18, "jamaican", inplace=True)
    df["cuisine"].replace(19, "russian", inplace=True)
    df["cuisine"].replace(20, "brazilian", inplace=True)    
    
    return df
results = Targetencoding_back(output)
results.head()
results.groupby('cuisine')['id'].nunique().sort_values(ascending=False)
results.to_csv('results.csv', index=False)
