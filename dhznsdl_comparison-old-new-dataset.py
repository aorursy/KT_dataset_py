import numpy as np 

import pandas as pd 



#=== New Dataset

train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')

#=== Old Dataset

train_old = pd.read_csv('../input/toxic-old-dataset/train.csv')

test_old = pd.read_csv('../input/toxic-old-dataset/test.csv')



#=== print Length of dataset

print("Length\ntrain_new:\t{}\ntest_new:\t{}\ntrain_old:\t{}\ntest_old:\t{}\n".format(len(train), len(test), len(train_old), len(test_old)))
#=== look the new dataset

train.head()
train2train = train['comment_text'].isin(train_old['comment_text']).sum()

test2train = train['comment_text'].isin(test_old['comment_text']).sum()



train2test = test['comment_text'].isin(train_old['comment_text']).sum()

test2test = test['comment_text'].isin(test_old['comment_text']).sum()



print("old train to new train:\t{} \nold test to new train:\t{} \nold train to new test:\t{} \nold test to new test:\t{}\n"

      .format(train2train, test2train, train2test, test2test))
len(train_old) - train2train 
# missing old train data

train_old[train_old['comment_text']. isin(train['comment_text'])==False]
# test data from old test data

test_old[test_old['comment_text']. isin(test['comment_text'])]