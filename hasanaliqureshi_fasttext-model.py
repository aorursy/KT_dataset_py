import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import fasttext

import bz2

import csv

from sklearn.metrics import roc_auc_score

import os

print(os.listdir("../input"))
# Load the training data 

data = bz2.BZ2File("../input/train.ft.txt.bz2")

data = data.readlines()

data = [x.decode('utf-8') for x in data]

print(len(data)) 
# 3.6mil rows! Lets inspect a few records to see the format and get a feel for the data

data[1:5]
# Data Prep

data = pd.DataFrame(data)

data.to_csv("train.txt", index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")



# Modelling

# This routine takes about 5 to 10 minutes 

model = fasttext.train_supervised('train.txt',label_prefix='__label__', thread=4, epoch = 10)

print(model.labels, 'are the labels or targets the model is predicting')
# Load the test data 

test = bz2.BZ2File("../input/test.ft.txt.bz2")

test = test.readlines()

test = [x.decode('utf-8') for x in test]

print(len(test), 'number of records in the test set') 



# To run the predict function, we need to remove the __label__1 and __label__2 from the testset.  

new = [w.replace('__label__2 ', '') for w in test]

new = [w.replace('__label__1 ', '') for w in new]

new = [w.replace('\n', '') for w in new]



# Use the predict function 

pred = model.predict(new)



# check the first record outputs

print(pred[0][0], 'is the predicted label')

print(pred[0][1], 'is the probability score')
print(model.predict("This love the design of this watch, but its showing wrong time."))
# Lets recode the actual targets to 1's and 0's from both the test set and the actual predictions  

labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test]

pred_labels = [0 if x == ['__label__1'] else 1 for x in pred[0]]



# run the accuracy measure. 

print(roc_auc_score(labels, pred_labels))