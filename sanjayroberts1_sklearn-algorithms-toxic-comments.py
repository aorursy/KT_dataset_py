# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pickle

#with open('../input/coolmodel/SGDC_classwieghts_15_85.pkl', 'rb') as fid:

#    lr = pickle.load(fid)



#with open('../input/allmodels/SGDC.pkl', 'rb') as fid:

#    lr = pickle.load(fid)



#with open('../input/allmodels/naivebayes.pkl', 'rb') as fid:

#    lr = pickle.load(fid)

    

#with open('../input/allmodels/naivebayes_weights_class_weights.pkl', 'rb') as fid:

#    lr = pickle.load(fid)

    

#with open('../input/allmodels/randomforest.pkl', 'rb') as fid:

#    lr = pickle.load(fid)

    

with open('../input/allmodels/randomforest_classweights.pkl', 'rb') as fid:

    lr = pickle.load(fid)
#test_df = pd.read_pickle("../input/testdata/sgdc_weights_test.pkl")



#test_df = pd.read_pickle("../input/allmodels/sgdc_test.pkl")



#test_df = pd.read_pickle("../input/allmodels/nb_test.pkl")



#test_df = pd.read_pickle("../input/mixed-model/nb_weights_test.pkl")



#test_df = pd.read_pickle("../input/allmodels/rf_test.pkl")



test_df = pd.read_pickle("../input/mixed-model/rf_weights_test.pkl")
test_df.shape
test_df[:10]
k = pd.DataFrame({'prediction': test_df})
k['id']=k.index

k = k.reset_index(drop=True)
k.head()
df = k[['id','prediction']]
df['id'] = df['id'] + 7000000
df
df.to_csv('submission.csv', index=False)