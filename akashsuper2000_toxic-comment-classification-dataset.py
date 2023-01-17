import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split
df = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')

df.shape
df2 = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv')

df2.shape
df[df['toxic']==1].shape, df[df['toxic']==0].shape
df2[df2['toxic']>=0.7].shape
df = df.iloc[:,1:3]
df2 = df2.iloc[:,1:3]
df
df2
df2_train = df2.copy()

df2_train.loc[df2_train['toxic']>=0.7, 'toxic'] = 1

df2_train = df2_train[df2_train['toxic'] == 1]

df2_train
train = pd.concat([df, df2_train])

train['toxic'] = train['toxic'].astype('int')

train
train[train['toxic']==1].shape, train[train['toxic']==0].shape
train_pos = train[train['toxic']==1].sample(frac=1)

train_neg = train[train['toxic']==0].sample(frac=1)
tf = pd.concat([train_pos.iloc[:60000,:], train_neg.iloc[:60000,:]])



val = pd.concat([train_pos.iloc[60000:70000,:], train_neg.iloc[60000:70000,:]])



test = pd.concat([train_pos.iloc[70000:80000,:], train_neg.iloc[70000:80000,:]])
tf.shape, val.shape, test.shape
tf.to_csv('train.csv', index=False)

val.to_csv('validation.csv', index=False)

test.to_csv('test.csv', index=False)