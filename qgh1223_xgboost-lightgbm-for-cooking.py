import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
def read_data(path):
    return json.load(open(path))
train = read_data('../input/train.json')
test = read_data('../input/test.json')
print(len(train))

cuisinelist=[]
ingredientlist=[]
for data in train:
    cuisinelist.append(data['cuisine'])
    ingredientstr=' '.join(data['ingredients'])
    ingredientlist.append(ingredientstr)
traindata=pd.concat([pd.Series(cuisinelist),
                     pd.Series(ingredientlist)],
                   axis=1)
traindata.columns=['cuisine','ingredients']
#print(traindata.head())
traindata.groupby('cuisine').size().plot(kind='barh')
vectorizer=TfidfVectorizer()
vectorizer=vectorizer.fit(traindata['ingredients'])
trainvector=vectorizer.transform(traindata['ingredients']).astype('float')
print(trainvector.shape)
x_train, x_test, y_train, y_test = train_test_split(trainvector,traindata['cuisine'],test_size=0.1)
le=LabelEncoder()
label=le.fit_transform(traindata['cuisine'])
x_train, x_test, y_train, y_test = train_test_split(trainvector,
                                                    label,
                                                    test_size=0.1)
clf=XGBClassifier(silent=1,
                 learning_rate=0.5,
                 min_child_weight=1,
                 max_depth=5,
                 gamma=0,
                 subsample=1,
                 max_delta_step=0,
                 colsample_bytree=1,
                 reg_lambda=1,
                 n_estimators=150,
                 seed=1000)
clf.fit(x_train,y_train)
print('train:'+str(clf.score(x_train,y_train)))
print('test:'+str(clf.score(x_test,y_test)))
xgb_params={
    'eta':0.05,
    'max_depth':5,
    'subsample':1,
    'colsample_bytree':1,
    'objective':'multi:softmax',
    'eval_metric':'merror',
    'silent':1
}
dtrain=xgb.DMatrix(x_train,y_train)
dtest=xgb.DMatrix(x_test)
''''model=xgb.cv(xgb_params,dtrain,num_boost_round=300,
            early_stopping_rounds=50,
            verbose_eval=20,
            show_stdv=False)
model[['train-merror-mean','test-merror-merror']].plot()'''
pre_label=clf.predict(x_train)
pre_label_final=le.inverse_transform(pre_label)
true_label_final=le.inverse_transform(y_train)
labelseries=pd.DataFrame(true_label_final)
labelseries.columns=['label']
classes=labelseries.groupby('label').size().index
cm=confusion_matrix(true_label_final,pre_label_final)
plt.imshow(cm,interpolation='nearest')
plt.xticks(range(len(classes)),classes,rotation=90)
plt.yticks(range(len(classes)),classes)
plt.colorbar()
cm1=[]
for i in range(len(cm)):
    sum1=np.sum(cm[i])
    arr1=[]
    for j in range(len(cm[i])):
        arr1.append(cm[i][j]/sum1)
    cm1.append(np.asarray(arr1))
plt.imshow(cm1,interpolation='nearest')
plt.xticks(range(len(classes)),classes,rotation=90)
plt.yticks(range(len(classes)),classes)
plt.colorbar()
testingredientlist=[]
idlist=[]
for data in test:
    idlist.append(data['id'])
    ingredientstr=' '.join(data['ingredients'])
    testingredientlist.append(ingredientstr)
testvector=vectorizer.transform(pd.Series(testingredientlist)).astype('float')
predictlabel=clf.predict(testvector)
predict_label=le.inverse_transform(predictlabel)
testresult=pd.concat([pd.Series(idlist),pd.Series(predict_label)],axis=1)
testresult.columns=['id','cuisine']
testresult.to_csv('result.csv',index=False)