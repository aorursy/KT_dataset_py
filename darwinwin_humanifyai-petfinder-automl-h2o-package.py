# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import h2o

from h2o.automl import H2OAutoML

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print(os.listdir("../input/train"))
train_df = pd.read_csv(r'../input/train/train.csv')

print(train_df.shape)

print(train_df.head(10))

train_df.describe()
test_df = pd.read_csv(r'../input/test/test.csv')

print(test_df.shape)

print(test_df.head(10))
test_df.columns
#h2o.shutdown(prompt=False)

#h2o.cluster().shutdown()

h2o.init()
dropFeatureList=['Description','RescuerID']

for feature in dropFeatureList:

    train_df.drop(feature,axis=1,inplace=True)

    test_df.drop(feature,axis=1,inplace=True)

print(train_df.columns,test_df.columns)

aml_df = h2o.H2OFrame(train_df)

Y='AdoptionSpeed'

all_colums = list(aml_df.columns)

X= all_colums.remove(Y)

#remove features-

print(all_colums,X)

aml_df['AdoptionSpeed'] = aml_df['AdoptionSpeed'].asfactor()

if True:

    aml = H2OAutoML(max_runtime_secs=10000, seed=42)

    aml.train(x=X, y=Y, training_frame=aml_df)

print('Training Done!!')
lb = aml.leaderboard

#print(lb.head(rows=lb.nrows))  # Entire set

lb.head()
test_df_h2o= h2o.H2OFrame(test_df)

dfres = aml.leader.predict(test_df_h2o)
dfres['predict']
sample_df = pd.read_csv(r'../input/test/sample_submission.csv')

test_h2o = h2o.H2OFrame(test_df)

submission = test_h2o[:, "PetID"]

#test_h2o['PetID'].shape

final_sub = submission.cbind(dfres[:,'predict'])

#h2o.exportFile(final_sub,'submission.csv')

pd_final_sub = pd.DataFrame(final_sub.as_data_frame())

pd_final_sub.rename(columns={'predict': 'AdoptionSpeed'}, inplace=True)

pd_final_sub.to_csv('submission.csv',index=False)



#submission= pd.concat([(test_df.PetID),pd.DataFrame(dfres['predict'])],axis=1)
pd_final_sub.AdoptionSpeed.value_counts()
print(os.listdir('.'))
print(test_df.columns)