import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

import numpy as np



heart = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
heart.head()
for i,col in enumerate(['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction', 'high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']):

  plt.figure(i)

  sb.catplot(x = col, y = 'DEATH_EVENT', data = heart, kind = 'point', aspect = 2)
heart.drop(['diabetes', 'sex', 'smoking'], axis = 1, inplace= True)
heart.head()
from sklearn.model_selection import train_test_split
feature = heart.drop('DEATH_EVENT', axis = 1)

label = heart['DEATH_EVENT']
train_feature,test_feature,train_label,test_label = train_test_split(feature,label,test_size = 0.5, random_state = 50)

val_feature,test_feature, val_label,test_label = train_test_split(test_feature,test_label,test_size = 0.5, random_state = 50)
train_feature.count()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
LogisticRegression()
def ML_results(results):



  mean = results.cv_results_['mean_test_score']

  std = results.cv_results_['std_test_score']

  for mean, std, params in zip(mean, std, results.cv_results_['params']):

    print('mean: ',round(mean,3),' std: ',round(std * 2,3),' for ',format(params))

  print('\n Final parameter decided: ',format(results.best_params_))
LR = LogisticRegression(max_iter=500)

parameter = {

    'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]

}

cv = GridSearchCV(LR, parameter, cv = 5)

cv.fit(train_feature,train_label.values.ravel())

LR = cv.best_estimator_

LR
from sklearn.ensemble import GradientBoostingClassifier
GradientBoostingClassifier()
gb = GradientBoostingClassifier()

parameter = {

    'learning_rate' : [0.01,0.1,1,10],

    'max_depth' : [1,2,3,4],

    'n_estimators' : [50,100,200,500]

}

cv = GridSearchCV(gb, parameter, cv = 5)

cv.fit(train_feature, train_label.ravel())

gb = cv.best_estimator_

gb
from time import time

from sklearn.metrics import accuracy_score, precision_score, recall_score

def eval_model(name, model , feature, label):

  start = time()

  pred = model.predict(feature)

  end = time()

  accuracy = round(accuracy_score(label,pred),4)

  presision = round(precision_score(label,pred),4)

  recall = round(recall_score(label,pred),4)

  print('{} == Accuracy: {}, Precision: {}, recall: {},Latency: {}'.format(name,

                                                                     accuracy,

                                                                     presision,

                                                                     recall,

                                                                     round((end-start),3)))

  return pred
print('Valdiation')

eval_model('LR',LR,val_feature,val_label)

eval_model('GB',gb,val_feature, val_label)
print('testing')

pred = eval_model('LR',LR,test_feature,test_label)

pred1 = eval_model('GB',gb,test_feature,test_label)

pred
test_label.values
actual = np.empty([75], dtype = int)

j=0

for i in test_label:

  actual[j] = i

  j+=1

actual
plt.figure(figsize=(15,10), dpi = 200)

plt.grid()

plt.plot(pred)

plt.plot(actual)
plt.figure(figsize=(15,10), dpi = 200)

plt.grid()

plt.plot(pred1)

plt.plot(actual)