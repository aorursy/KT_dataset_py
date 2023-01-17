import numpy as np
import pandas as pd
import os
import tqdm
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report

import warnings
warnings.filterwarnings("ignore")

print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.head()
train.info()
train['difficulty'].value_counts()
train['target'].value_counts()
train = train.drop(['Id'], axis=1)
train.head()
Xtrain, Xtest, ytrain, ytest = train_test_split(train.iloc[:,:2], train['target'], test_size = 0.1, random_state = 0)
diff1 = Xtrain[Xtrain['difficulty'] == 1]
diff2 = Xtrain[Xtrain['difficulty'] == 2]
diff3 = Xtrain[Xtrain['difficulty'] == 3]
diff4 = Xtrain[Xtrain['difficulty'] == 4]
diff1['ciphertext'] = diff1['ciphertext'].apply(lambda x: x.replace('1', ' '))
diff2['ciphertext'] = diff2['ciphertext'].apply(lambda x: x.replace('8', ' '))
diff3['ciphertext'] = diff3['ciphertext'].apply(lambda x: x.replace('8', ' '))
diff4['ciphertext'] = diff4['ciphertext'].apply(lambda x: x.replace('8', ' '))
diff1.head()
diff_test1 = Xtest[Xtest['difficulty'] == 1]
diff_test2 = Xtest[Xtest['difficulty'] == 2]
diff_test3 = Xtest[Xtest['difficulty'] == 3]
diff_test4 = Xtest[Xtest['difficulty'] == 4]

diff_test1['ciphertext'] = diff_test1['ciphertext'].apply(lambda x: x.replace('1', ' ')).fillna(0)
diff_test2['ciphertext'] = diff_test2['ciphertext'].apply(lambda x: x.replace('8', ' ')).fillna(0)
diff_test3['ciphertext'] = diff_test3['ciphertext'].apply(lambda x: x.replace('8', ' ')).fillna(0)
diff_test4['ciphertext'] = diff_test4['ciphertext'].apply(lambda x: x.replace('8', ' ')).fillna(0)
diff1.head()
start = time.time()
vect1 = TfidfVectorizer(analyzer = 'char_wb', lowercase = False, ngram_range=(1, 6))
train_vect1 = vect1.fit_transform(diff1['ciphertext'])
test_vect1 = vect1.transform(diff_test1['ciphertext'])
print('Time: ' + str(time.time() - start) + 's')

start = time.time()
vect2 = TfidfVectorizer(analyzer = 'char_wb', lowercase = False, ngram_range=(1, 6))
train_vect2 = vect2.fit_transform(diff2['ciphertext'])
test_vect2 = vect2.transform(diff_test2['ciphertext'])
print('Time: ' + str(time.time() - start) + 's')

start = time.time()
vect3 = TfidfVectorizer(analyzer = 'char_wb', lowercase = False, ngram_range=(1, 6), max_features = 660000)
train_vect3 = vect3.fit_transform(diff3['ciphertext'])
test_vect3 = vect3.transform(diff_test3['ciphertext'])
print('Time: ' + str(time.time() - start) + 's')

start = time.time()
vect4 = TfidfVectorizer(analyzer = 'char_wb', lowercase = False, ngram_range=(1, 6), max_features = 660000)
train_vect4 = vect4.fit_transform(diff4['ciphertext'])
test_vect4 = vect4.transform(diff_test4['ciphertext'])
print('Time: ' + str(time.time() - start) + 's')
model1 = LogisticRegression(tol=0.001, C=13.0, random_state=34, solver='sag', max_iter=100, multi_class='auto', verbose=1, n_jobs=-1)
model1.fit(train_vect1, ytrain.loc[diff1.index])

model2 = LogisticRegression(tol=0.001, C=59.0, random_state=29, solver='sag', max_iter=100, multi_class='auto', verbose=1, n_jobs=-1)
model2.fit(train_vect2, ytrain.loc[diff2.index])

model3 = LogisticRegression(tol=0.001, C=10.0, random_state=0, solver='sag', max_iter=100, multi_class='auto', verbose=1, n_jobs=-1)
model3.fit(train_vect3, ytrain.loc[diff3.index])

model4 = LogisticRegression(tol=0.001, C=10.0, random_state=0, solver='sag', max_iter=100, multi_class='auto', verbose=1, n_jobs=-1)
model4.fit(train_vect4, ytrain.loc[diff4.index])
pred1 = model1.predict(test_vect1)
pred2 = model2.predict(test_vect2)
pred3 = model3.predict(test_vect3)
pred4 = model4.predict(test_vect4)
print(accuracy_score(pred1, ytest.loc[diff_test1.index]))
print(accuracy_score(pred2, ytest.loc[diff_test2.index]))
print(accuracy_score(pred3, ytest.loc[diff_test3.index]))
print(accuracy_score(pred4, ytest.loc[diff_test4.index]))
print(f1_score(pred1, ytest.loc[diff_test1.index], average='macro')) # 0.6561271380779565
print(f1_score(pred2, ytest.loc[diff_test2.index], average='macro')) # 0.6593521513806591 
print(f1_score(pred3, ytest.loc[diff_test3.index], average='macro')) # 0.4219906210294547
print(f1_score(pred4, ytest.loc[diff_test4.index], average='macro'))
test1 = pd.read_csv('../input/test.csv')
test = test1.copy()
test.head()
test_diff1 = test[test['difficulty'] == 1]
test_diff2 = test[test['difficulty'] == 2]
test_diff3 = test[test['difficulty'] == 3]
test_diff4 = test[test['difficulty'] == 4]
test_diff1['ciphertext'] = test_diff1['ciphertext'].apply(lambda x: x.replace('1', ' '))
test_diff2['ciphertext'] = test_diff2['ciphertext'].apply(lambda x: x.replace('8', ' '))
test_diff3['ciphertext'] = test_diff3['ciphertext'].apply(lambda x: x.replace('8', ' '))
test_diff4['ciphertext'] = test_diff4['ciphertext'].apply(lambda x: x.replace('8', ' '))
start = time.time()
test_vect_1 = vect1.transform(test_diff1['ciphertext'])
test_vect_2 = vect2.transform(test_diff2['ciphertext'])
test_vect_3 = vect3.transform(test_diff3['ciphertext'])
test_vect_4 = vect4.transform(test_diff4['ciphertext'])
print('Time taken: ' + str(time.time() - start))
test_pred1 = model1.predict(test_vect_1)
test_pred2 = model2.predict(test_vect_2)
test_pred3 = model3.predict(test_vect_3)
test_pred4 = model4.predict(test_vect_4)
test_diff1['pred'] = test_pred1
test_diff2['pred'] = test_pred2
test_diff3['pred'] = test_pred3
test_diff4['pred'] = test_pred4
test_diff1.head()
test_diff = pd.concat([test_diff1, test_diff2, test_diff3, test_diff4])
test_diff = test_diff.set_index('Id').loc[test1['Id']]
test_diff = test_diff.drop(['difficulty', 'ciphertext'], axis=1)
test_diff = test_diff.reset_index()
test_diff.columns = ['Id', 'Predicted']
test_diff.to_csv('submission.csv', index=False)