# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import hashlib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
tf.logging.set_verbosity(tf.logging.DEBUG)

inv_map={}

def label2num( dfcolumn ):
    reduced_list=sorted(set(dfcolumn))
    map_dict={}
    for s in reduced_list:
        map_dict[s]=int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**8
    return dfcolumn.map(map_dict)

def label2inc( dfcolumn ):
    reduced_list=sorted(set(dfcolumn))
    counter=0
    map_dict={}
    for s in reduced_list:
        map_dict[s]=counter
        counter=counter+1
    print(map_dict)
    global inv_map
    inv_map = {v: k for k, v in map_dict.items()}
    return dfcolumn.map(map_dict)

all_data = pd.read_csv("../input/gen_data.csv")
text_feature_list = ['Day', 'Holiday', 'Strongest Wifi', 'Connected Wifi']
class_label = "Mode"

for field in text_feature_list:
    all_data[field]=label2num(all_data[field])
print(all_data[:5])

train = all_data[::2]
test = all_data[1::2]

print("Training = ")
print(train[:5])

print("Test = ")
print(test[:5])
x_train = train.drop([class_label, 'Id'], axis=1).astype(np.float32).values
y_train = label2inc(train[class_label]).astype(np.integer).values
#print(x_train)
#print(y_train)
x_test = test.drop([class_label, 'Id'], axis=1).astype(np.float32).values
y_test = label2inc(test[class_label]).astype(np.integer).values
#print(x_test)
#print(y_test)

params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
  num_classes=3, num_features=8, num_trees=4, max_nodes=100, split_after_samples=50)

print("Params =")
print(params)
classifier = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(
    params=params, model_dir="./")
classifier.fit(x=x_train, y=y_train)
res = classifier.predict(x_test)
l1=[w['classes'] for w in res]
print(inv_map)
print("Index\tPredicted Result\tGround Truth")
count=0
for index in range(0,len(l1)):
    print(index,"\t",l1[index]," ",inv_map[l1[index]],"\t\t",y_test[index]," ",inv_map[y_test[index]])
    count=count+1

summary={}
for prediction in range(0,3):
    for truth in range(0,3):
        summary['Prediction : '+str(inv_map[prediction])+'\tTruth : '+str(inv_map[truth])] = len([w for w in range(0,count) if l1[w]==prediction and y_test[w]==truth ])
print('\n\nSummary of results - \n')
for entry in summary:
    print(entry,"\nCount : ",summary[entry],"\n")
