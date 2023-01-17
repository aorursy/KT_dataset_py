# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

#datapath = 'C:/Users/prateek.g/Downloads/titanic/'

datapath = '../input/hr-hackathon/'

#!pip install autoviml
### If you want to see the sitepackages version use this

from autoviml.Auto_ViML import Auto_ViML

train = pd.read_csv(datapath+'train_jqd04QH.csv')

#test = train[-15:]

test = pd.read_csv(datapath+'test_KaymcHn.csv')

print(train.shape)

print(test.shape)

print(train.head())

target = 'target'

train[target].value_counts()
sample_submission=''

scoring_parameter = 'roc_auc'
#### If Boosting_Flag = True => XGBoost, Fase=>ExtraTrees, None=>Linear Model

m, feats, trainm, testm = Auto_ViML(train, target, test, sample_submission,

                                    scoring_parameter=scoring_parameter,

                                    hyper_param='GS',feature_reduction=True,

                                     Boosting_Flag=True,Binning_Flag=False,

                                    Add_Poly=0, Stacking_Flag=True,                                    

                                    Imbalanced_Flag=True, 

                                    verbose=1)
submission=pd.read_csv('../input/hr-hackathon/sample_submission_sxfcbdx.csv')

submission=submission.drop('target',axis=1)
submission['target']=testm['target_Stacked_Linear Discriminant_predictions']

submission.to_csv('submission.csv',index=False)
######## Use this to Test Classification Problems Only ####

modelname='Naive_Bayes'

def accu(results, y_cv):

    return (results==y_cv).astype(int).sum(axis=0)/(y_cv.shape[0])

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import balanced_accuracy_score

try:

    print('Test results since target variable is present in test data:')

    print(confusion_matrix(test[target].values,testm[target+'_'+modelname+'_predictions'].values))

    print('\nBalanced Accuracy = %0.2f%%\n' %(100*balanced_accuracy_score(test[target].values, testm[target+'_'+modelname+'_predictions'].values)))

    print(classification_report(test[target].values,testm[target+'_'+modelname+'_predictions'].values))

except:

    print('No target variable present in test data. No results')
testm.head()