# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import xgboost as xgb

from sklearn.metrics import precision_score, recall_score, accuracy_score

from sklearn.model_selection import GridSearchCV

from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import label_binarize

# import lightgbm as lgb
trainDf = pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')

testDf = pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')

# df.corr()
attr = ['chem_0','chem_1','chem_2','chem_3','chem_4','chem_5','chem_6','chem_7','attribute']

# attr = ['chem_0','chem_1','chem_4','chem_6']

fulltrainData = trainDf[attr]

fulltrainLabel = trainDf['class']

fulltrainLabel = fulltrainLabel-1



trainData = trainDf[attr].iloc[:100]

trainLabel = trainDf['class'].iloc[:100]

trainLabel = trainLabel-1



valData = trainDf[attr].iloc[100:]

valLabel = trainDf['class'].iloc[100:]

valLabel = valLabel-1;



testData = testDf[attr]

testID = testDf['id']
# train_data = lgb.Dataset(trainData, label=trainData)

fullD_train = xgb.DMatrix(fulltrainData, label=fulltrainLabel)

D_train = xgb.DMatrix(trainData, label=trainLabel)

D_val   = xgb.DMatrix(valData, label=valLabel)

D_test  = xgb.DMatrix(testData)
param = {    'booster' : 'gbtree',#dart    

                      'eta' : 0.2,    

                      'objective' : 'multi:softmax',    

                      'gamma' : 0,    'max_depth' : 6,    

                      'grow_poilicy' : 'depthwise',    

                      'num_class' : 7,

                     }

step = 10
model = xgb.train(param, D_train, step)

preds = model.predict(D_val)



print("Precision = {}".format(precision_score(valLabel, preds, average='macro')))

print("Recall = {}".format(recall_score(valLabel, preds, average='macro')))

print("Accuracy = {}".format(accuracy_score(valLabel, preds)))
preds = (model.predict(D_test)+1).astype(np.int)

answer = pd.DataFrame({'id':testID, 'class':preds})

answer.head()

answer.to_csv('submission10.csv',index=None)