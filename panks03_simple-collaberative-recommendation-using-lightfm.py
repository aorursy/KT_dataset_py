import pandas as pd
import numpy as np
import os
print(os.getcwd())

os.chdir("/kaggle/input/unzipped/")

print(os.getcwd())
train= pd.read_csv("/kaggle/input/unzipped/train.csv")

train.shape
from sklearn.preprocessing import LabelEncoder
user_encoder= LabelEncoder()
song_encoder= LabelEncoder()

train['msno']= user_encoder.fit_transform(train['msno'])
train['song_id']=song_encoder.fit_transform(train['song_id'])
train.describe(include='object')
#Unique Users, Unique Songs
train.msno.nunique(), train.song_id.nunique()
train.song_id.value_counts().head() #Top 5 songs
train.msno.value_counts().head() #Top 5 users
from lightfm import LightFM
from scipy.sparse import coo_matrix

u= train['msno']
s= train['song_id']
t= train['target']

lu= u.nunique()
ls= s.nunique()
#Size comparision- in bytes!

import sys

print("Original train size:",sys.getsizeof(train))

matrix= coo_matrix((t,(u,s)), shape=(lu, ls))

print("Coordinate matrix size: ", sys.getsizeof(matrix))
model= LightFM(no_components=30, k=5, learning_rate=0.05, random_state=33)

model.fit(matrix,epochs= 50, num_threads= 4)
#Let us analyze recommendation for user 0
user_id=0
#Predict probability if there will be recurring listening event(s) triggered within a month?
model.predict(0, [313171,
175398,
338865,
101401,
155979])
preds= model.predict(user_id, list(range(len(song_encoder.classes_))))
preds= pd.DataFrame(zip(preds, song_encoder.classes_), columns=['pred', 'song_id'])
preds= preds.sort_values('pred', ascending= False)
preds.head() #for him: user_id
#TOP 5 RECOMMENDATIONS FOR A USER
tried= train[train['msno']==user_id]['song_id'].values
list(preds[~preds['song_id'].isin(tried)]['song_id'].values[:5])
#We will consider only 20 percent of data for evaluation
train=train.sample(frac=0.2)
# train=train.sample(frac=0.5)
matrix
#no.of cores each processor is having
!lscpu | grep 'Core(s) per socket'
#no.of threads each core is having
!lscpu | grep 'Thread(s) per core'
!cat /proc/meminfo | grep 'MemAvailable'
from lightfm.evaluation import auc_score, precision_at_k
print("auc:",auc_score(model, matrix, num_threads=4).mean())
print("prec:",precision_at_k(model, matrix, num_threads=4).mean())
from lightfm.cross_validation import random_train_test_split
train, test= random_train_test_split(matrix, test_percentage=0.2)
random_train_test_split(matrix)
from lightfm.evaluation import auc_score, precision_at_k
model= LightFM(loss='warp')
scores=[]
for e in range(25):
    model.fit_partial(train, epochs=1, num_threads=4)
    auc_train= auc_score(model, train, num_threads=4).mean()
    auc_test= auc_score(model, test, num_threads=4).mean()
    scores.append((auc_train, auc_test))
    
scores = np.array(scores)
from matplotlib import pyplot as plt

%matplotlib inline

plt.plot(scores[:,0], label='train')
plt.plot(scores[:,1], label='test')
#Loss- 'bpr'
model= LightFM(loss='bpr')

scores=[]
for e in range(25):
    model.fit_partial(train, epochs=1, num_threads=4)
    auc_train = auc_score(model, train, num_threads=4).mean()
    auc_test= auc_score(model, test, num_threads=4).mean()
    scores.append((auc_train, auc_test))
    
scores= np.array(scores)
from matplotlib import pyplot as plt

plt.plot(scores[:,0], label='train')
plt.plot(scores[:,1], label='test')
plt.legend()
from copy import deepcopy

model= LightFM(loss='bpr')

count = 0
best = 0
scores = []
for e in range(50):
    if count>5:
        break
    model.fit_partial(train, epochs=1)
    auc_train= auc_score(model, train).mean()
    auc_test= auc_score(model, test).mean()
    print(f'Epoch: {e}, Train AUC={auc_train:.3f}, Test AUC={auc_test:.3f}')
    scores.append((auc_train, auc_test))
    if auc_test > best:
        best_model = deepcopy(model)
        best = auc_test
    else:
        count += 1

model= deepcopy(best_model)
