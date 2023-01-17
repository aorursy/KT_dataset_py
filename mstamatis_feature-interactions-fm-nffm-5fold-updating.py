#Download ctrNet-tool 

#You can find the code in https://github.com/guoday/ctrNet-tool

!git clone https://github.com/guoday/ctrNet-tool.git

!cp -r ctrNet-tool/* ./

!rm -r ctrNet-tool data .git

!ls -all



import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

import ctrNet

import pandas as pd

import numpy as np

import tensorflow as tf

from sklearn.model_selection import train_test_split

from src import misc_utils as utils

import os

import gc

import random
train = pd.read_csv("../input/train.csv")

train.head(10)
train.isnull().sum().sum()
test = pd.read_csv("../input/test.csv")

test.head(10)
X_train = train.drop(['ID_code', 'target'], axis = 1)

X_train.head(10)
y_train = train['target']

y_train.head()
X_test = test.drop(['ID_code'], axis = 1)

X_test.head()
test.isnull().sum().sum()
X_train.var()
train, dev,y,_ = train_test_split(X_train,y_train,test_size=0.02, random_state=2019)



features=train.columns.tolist()[1:-1]
#features = train.columns.tolist()[1:-1][:100]

hparam=tf.contrib.training.HParams(

            model='fm', #['fm','ffm','nffm']

            k=16,

            hash_ids=int(1e5),

            batch_size=8,

            optimizer="adam", #['adadelta','adagrad','sgd','adam','ftrl','gd','padagrad','pgd','rmsprop']

            learning_rate=0.003,

            num_display_steps=100,

            num_eval_steps=1000,

            epoch=3,

            metric='auc', #['auc','logloss']

            init_method='uniform', #['tnormal','uniform','normal','xavier_normal','xavier_uniform','he_normal','he_uniform']

            init_value=0.1,

            feature_nums=len(features),

            kfold=5)

utils.print_hparams(hparam)



index=set(range(train.shape[0]))

K_fold=[]

for i in range(hparam.kfold):

    if i == hparam.kfold-1:

        tmp=index

    else:

        tmp=random.sample(index,int(1.0/hparam.kfold*train.shape[0]))

    index=index-set(tmp)

    print("Number:",len(tmp))

    K_fold.append(tmp)

    



for i in range(hparam.kfold):

    print("Fold",i)

    dev_index=K_fold[i]

    dev_index=random.sample(dev_index,int(0.1*len(dev_index)))

    train_index=[]

    for j in range(hparam.kfold):

        if j!=i:

            train_index+=K_fold[j]

    model=ctrNet.build_model(hparam)

    model.train(train_data=(train.iloc[train_index][features],y_train[train_index]),\

                dev_data=(train.iloc[dev_index][features],y_train[dev_index]))

    print("Training Done! Inference...")

    if i==0:

        preds=model.infer(dev_data=(test[features],y_train))/hparam.kfold

    else:

        preds+=model.infer(dev_data=(test[features],y_train))/hparam.kfold
result = pd.DataFrame({"ID_code": test['ID_code'], "target": preds})

result.head()
result.to_csv("submission.csv", index=False)