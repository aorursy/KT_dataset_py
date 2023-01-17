# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import ExtraTreesClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#train_categorical=pd.read_csv("../input/train_categorical.csv", nrows=1000)

#train_date=pd.read_csv("../input/train_date.csv", nrows=1000)

train_numeric_all=pd.read_csv("../input/train_numeric.csv",nrows=10000)

split_index = np.random.rand(len(train_numeric_all)) < 0.8

train_numeric=train_numeric_all[split_index]

train_numeric_valid=train_numeric_all[~split_index]
train_numeric.fillna(value = -99, inplace = True)

    

id_train = train_numeric['Id']

ytrain = train_numeric['Response']

train_numeric.drop(['Id', 'Response'], axis = 1, inplace = True)
clf = ExtraTreesClassifier(n_estimators = 50, n_jobs = -1,

                                   min_samples_leaf = 10, verbose = 1)

clf.fit(xtrain, ytrain)                                   

    

test_nummeric = pd.read_csv('../input/test_numeric.csv')

test_nummeric.fillna(value = -99, inplace = True)

    

id_test = test_nummeric['Id']

test_nummeric.drop(['Id'], axis = 1, inplace = True)

    

pred = clf.predict_proba(test_nummeric)
bosch_result=pd.DataFrame({'Id':id_test,

                           'Response':pred})

bosch_result.to_csv('bosch_result.csv', index = False)