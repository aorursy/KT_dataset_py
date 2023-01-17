import numpy as np

import pandas as pd

from catboost import Pool, CatBoostClassifier
#cloud = 'drive/My Drive/techjam_final_2019/'

cloud = ''
demo_data = pd.read_csv(cloud+'tj19data/demo.csv')

test_data = pd.read_csv(cloud+'tj19data/test.csv')

train_data = pd.read_csv(cloud+'tj19data/train.csv')

txn_data = pd.read_csv(cloud+'tj19data/txn.csv')

#from sklearn.utils import shuffle

#txn_data = shuffle(txn_data, random_state=43)

#txn_data = txn_data[:len(txn_data)//66]

len(txn_data)
demo_data.dropna(inplace=True)

txn_data.dropna(inplace=True)
txn_data.iloc[100]
demo_data.iloc[0]
id2label = dict()

for id_,label in zip(train_data['id'],train_data['label']):

  id2label[id_] = [label]
for ind in range(len(demo_data)):

  if demo_data.iloc[ind][0] in id2label:



    if len(id2label[demo_data.iloc[ind][0]]) == 1:

      c0 = [0]*2

      c0[int(demo_data.iloc[ind][1])-1] = 1

      c1 = [0]*13

      c1[int(demo_data.iloc[ind][2])-1] = 1

      c2 = [0]*5

      c2[int(demo_data.iloc[ind][3])-95] = 1

      c3 = [0]*2

      c3[int(demo_data.iloc[ind][4])] = 1

      c4 = [0]*2

      c4[int(demo_data.iloc[ind][5])] = 1



      id2label[demo_data.iloc[ind][0]] += [c0,c1,c2,c3,c4,0,0,0]



    else:

      id2label[demo_data.iloc[ind][0]][1][int(demo_data.iloc[ind][1])-1] += 1 

      id2label[demo_data.iloc[ind][0]][2][int(demo_data.iloc[ind][2])-1] += 1

      id2label[demo_data.iloc[ind][0]][3][int(demo_data.iloc[ind][3])-95] += 1

      id2label[demo_data.iloc[ind][0]][4][int(demo_data.iloc[ind][4])] += 1

      id2label[demo_data.iloc[ind][0]][5][int(demo_data.iloc[ind][5])] += 1



      id2label[demo_data.iloc[ind][0]][6] += demo_data.iloc[ind][6]

      id2label[demo_data.iloc[ind][0]][7] += demo_data.iloc[ind][7]

      id2label[demo_data.iloc[ind][0]][8] += demo_data.iloc[ind][8]

      
for k in id2label.keys():

    if len(id2label[k]) == 1:

      id2label[k] += [[0]*2,[0]*13,[0]*5,[0]*2,[0]*2,0,0,0]
c5_s=set()

c6_s=set()

c7_s=set()

for ind in range(len(txn_data)):

  c5_s.add(txn_data.iloc[ind][3])

  c6_s.add(txn_data.iloc[ind][4])

  c7_s.add(txn_data.iloc[ind][5])

max(c5_s),min(c5_s),max(c6_s),min(c6_s),max(c7_s),min(c7_s)
c5_min = min(c5_s)

c6_min = min(c6_s)

c7_min = min(c7_s)



c5_max = max(c5_s)+1

c6_max = max(c6_s)+1

c7_max = max(c7_s)+1



for ind in range(len(txn_data)):

  if txn_data.iloc[ind][0] in id2label:

    if len(id2label[txn_data.iloc[ind][0]]) == 9:

      c5 = [0]*(c5_max-c5_min)

      c5[int(txn_data.iloc[ind][3])-c5_min] = 1

      c6 = [0]*(c6_max-c6_min)

      c6[int(txn_data.iloc[ind][4])-c6_min] = 1

      c7 = [0]*(c7_max-c7_min)

      c7[int(txn_data.iloc[ind][5])-c7_min] = 1

      

      id2label[txn_data.iloc[ind][0]] += [c5,c6,c7,0,0,0,0,0]



    else:

      #print(len(id2label[txn_data.iloc[ind][0]]))

      id2label[txn_data.iloc[ind][0]][9][int(txn_data.iloc[ind][3])-c5_min] += 1 

      id2label[txn_data.iloc[ind][0]][10][int(txn_data.iloc[ind][4])-c6_min] += 1

      id2label[txn_data.iloc[ind][0]][11][int(txn_data.iloc[ind][5])-c7_min] += 1



      id2label[txn_data.iloc[ind][0]][12] += txn_data.iloc[ind][6]

      id2label[txn_data.iloc[ind][0]][13] += txn_data.iloc[ind][7]

      id2label[txn_data.iloc[ind][0]][14] += txn_data.iloc[ind][8]

      id2label[txn_data.iloc[ind][0]][15] += txn_data.iloc[ind][9]

      id2label[txn_data.iloc[ind][0]][16] += txn_data.iloc[ind][10]
for k in id2label.keys():

    if len(id2label[k]) == 9:

      id2label[k] += [[0]*(c5_max-c5_min),[0]*(c6_max-c6_min),[0]*(c7_max-c7_min),0,0,0,0,0]
id2info = np.array(list(id2label.values()))

id2info.shape
data = []

for i in id2info:

  tmp = []

  for j in i:

    if isinstance(j, list):

      tmp += [*j]

    else:

      tmp += [j]

  data.append(tmp)

  del tmp
data = np.array(data)

data.shape
X = data[:,1:]

y = data[:,0]
id2label_test = dict()

for id_ in test_data['id']:

  id2label_test[id_] = [0]



for ind in range(len(demo_data)):

  if demo_data.iloc[ind][0] in id2label_test:



    if len(id2label_test[demo_data.iloc[ind][0]]) == 1:

      c0 = [0]*2

      c0[int(demo_data.iloc[ind][1])-1] = 1

      c1 = [0]*13

      c1[int(demo_data.iloc[ind][2])-1] = 1

      c2 = [0]*5

      c2[int(demo_data.iloc[ind][3])-95] = 1

      c3 = [0]*2

      c3[int(demo_data.iloc[ind][4])] = 1

      c4 = [0]*2

      c4[int(demo_data.iloc[ind][5])] = 1



      id2label_test[demo_data.iloc[ind][0]] += [c0,c1,c2,c3,c4,0,0,0]



    else:

      id2label_test[demo_data.iloc[ind][0]][1][int(demo_data.iloc[ind][1])-1] += 1 

      id2label_test[demo_data.iloc[ind][0]][2][int(demo_data.iloc[ind][2])-1] += 1

      id2label_test[demo_data.iloc[ind][0]][3][int(demo_data.iloc[ind][3])-95] += 1

      id2label_test[demo_data.iloc[ind][0]][4][int(demo_data.iloc[ind][4])] += 1

      id2label_test[demo_data.iloc[ind][0]][5][int(demo_data.iloc[ind][5])] += 1



      id2label_test[demo_data.iloc[ind][0]][6] += demo_data.iloc[ind][6]

      id2label_test[demo_data.iloc[ind][0]][7] += demo_data.iloc[ind][7]

      id2label_test[demo_data.iloc[ind][0]][8] += demo_data.iloc[ind][8]

      

for k in id2label_test.keys():

    if len(id2label_test[k]) == 1:

      id2label_test[k] += [[0]*2,[0]*13,[0]*5,[0]*2,[0]*2,0,0,0]

    

c5_min = min(c5_s)

c6_min = min(c6_s)

c7_min = min(c7_s)



c5_max = max(c5_s)+1

c6_max = max(c6_s)+1

c7_max = max(c7_s)+1



for ind in range(len(txn_data)):

  if txn_data.iloc[ind][0] in id2label_test:

    if len(id2label_test[txn_data.iloc[ind][0]]) == 9:

      c5 = [0]*(c5_max-c5_min)

      c5[int(txn_data.iloc[ind][3])-c5_min] = 1

      c6 = [0]*(c6_max-c6_min)

      c6[int(txn_data.iloc[ind][4])-c6_min] = 1

      c7 = [0]*(c7_max-c7_min)

      c7[int(txn_data.iloc[ind][5])-c7_min] = 1

      

      id2label_test[txn_data.iloc[ind][0]] += [c5,c6,c7,0,0,0,0,0]



    else:

      #print(len(id2label_test[txn_data.iloc[ind][0]]))

      id2label_test[txn_data.iloc[ind][0]][9][int(txn_data.iloc[ind][3])-c5_min] += 1 

      id2label_test[txn_data.iloc[ind][0]][10][int(txn_data.iloc[ind][4])-c6_min] += 1

      id2label_test[txn_data.iloc[ind][0]][11][int(txn_data.iloc[ind][5])-c7_min] += 1



      id2label_test[txn_data.iloc[ind][0]][12] += txn_data.iloc[ind][6]

      id2label_test[txn_data.iloc[ind][0]][13] += txn_data.iloc[ind][7]

      id2label_test[txn_data.iloc[ind][0]][14] += txn_data.iloc[ind][8]

      id2label_test[txn_data.iloc[ind][0]][15] += txn_data.iloc[ind][9]

      id2label_test[txn_data.iloc[ind][0]][16] += txn_data.iloc[ind][10]

    

for k in id2label_test.keys():

    if len(id2label_test[k]) == 9:

      id2label_test[k] += [[0]*(c5_max-c5_min),[0]*(c6_max-c6_min),[0]*(c7_max-c7_min),0,0,0,0,0]

    

id2info_test = np.array(list(id2label_test.values()))

id2info_test.shape





data_test = []

for i in id2info_test:

    tmp = []

    for j in i:

        if isinstance(j, list):

            tmp += [*j]

        else:

            tmp += [j]

    data_test.append(tmp)

    del tmp

    

len(data_test[0]),len(data_test[1])



data_test = np.array(data_test)[:,1:]

data_test.shape
from sklearn.model_selection import train_test_split

train_data, eval_data, train_label, eval_label = train_test_split(X, y, test_size=0.33, random_state=42)
train_data.shape
#{'learn': {'Accuracy': 0.40182089552238803, 'MultiClass': 1.560120235471146}}
import tensorflow as tf

def map13eval(preds, dtrain):

    actual = dtrain.get_label()

    predicted = preds.argsort(axis=1)[:,-np.arange(1,14)]

    metric = 0.

    for i in range(13):

        metric += np.sum(actual==predicted[:,i])/(i+1)

    metric /= actual.shape[0]

    return 'MAP@13', metric





def weighted_categorical_cross_entropy( y_pred,y_true):

    

    w = tf.reduce_sum(y_true)/tf_cast(tf_size(y_true), tf_float32)

    loss = w * tf.nn.softmax_cross_entropy_with_logits(onehot_labels, logits)

    return 'WCCE',loss
import xgboost as xgb



train_ = xgb.DMatrix(train_data, label=train_label)

test_ = xgb.DMatrix(eval_data, label=eval_label)



watchlist = [ (train_,'train'), (test_, 'test') ]

param = dict()

param['objective'] = 'multi:softprob'

#param['eval_metric'] = ['logloss']#,'merror','mlogloss']

param['num_class'] = 13

num_round = 100000000

bst = xgb.train(param, train_, num_round, watchlist, early_stopping_rounds=50, maximize=True,feval=map13eval) #feval=map5eval
yprob = bst.predict( test_ ).reshape( eval_label.shape[0], 13 )

ylabel = np.argmax(yprob, axis=1)



print ('predicting, classification error=%f' % \

       (np.sum( ylabel != eval_label) / float(eval_label.shape[0])))
#predicting, classification error=0.697424
yprob_test = bst.predict( xgb.DMatrix(data_test) ).reshape( data_test.shape[0], 13 )

#ylabel_test = np.argmax(yprob_test, axis=1)

yprob_test.shape
import csv

classes = ['class'+str(i) for i in range(13)]

with open(cloud+'file_XGB.csv', mode='w') as file:

    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['id']+classes)

    [writer.writerow([i,*j]) for i,j in zip(test_data['id'],yprob_test)]
r = pd.read_csv('file_XGB.csv')

r