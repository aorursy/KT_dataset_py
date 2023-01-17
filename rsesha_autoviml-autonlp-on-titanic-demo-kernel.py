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
train = pd.read_csv('/kaggle/input/titanic/train.csv',index_col=None)
print(train.shape)
test = pd.read_csv('/kaggle/input/titanic/test.csv',index_col=None)
print(test.shape)
subm = pd.read_csv('/kaggle/input/titanic/gender_submission.csv',index_col=None)
print(subm.shape)
!pip install autoviml
from autoviml.Auto_ViML import Auto_ViML
target = 'Survived'
#### Set Boosting_Flag = None, KMeans_Featurizer = True, scoring_parameter='balanced-accuracy'   ###########
m, feats, trainm, testm = Auto_ViML(train, target, test,
                            sample_submission=subm,
                            scoring_parameter='balanced-accuracy', KMeans_Featurizer=True,
                            hyper_param='GS',feature_reduction=True,
                             Boosting_Flag=None,Binning_Flag=True,
                            Add_Poly=0, Stacking_Flag=True,Imbalanced_Flag=False,
                            verbose=0)
subm[target] = testm[target+'_predictions'].astype(int).values
filename='sample_submission_log.csv'
savefile = '/kaggle/working/Survived/sample_submission_log.csv'
savefile
subm.to_csv(savefile,index=False)
from autoviml.Auto_NLP import Auto_NLP
score_type = 'accuracy'
modeltype = 'Classification'
nlp_column = 'Name'
train_nlp, test_nlp, best_nlp_transformer, _ = Auto_NLP(
                nlp_column, train, test, target, score_type,
                modeltype,top_num_features=50, verbose=0)
m4, feats4, trainm4, testm4 = Auto_ViML(train_nlp, target, test_nlp, 
                                    sample_submission=subm,
                                    scoring_parameter='balanced-accuracy',
                                    hyper_param='GS',feature_reduction=True,
                                     Boosting_Flag="CatBoost",Binning_Flag=False,
                                    Add_Poly=0, Stacking_Flag=False,Imbalanced_Flag=False, 
                                    verbose=2)
subm[target] = testm4['Survived_CatBoost_predictions'].values.astype(int)
subm.head()
subm.to_csv('sample_submission4.csv',index=False)
m2, feats2, trainm2, testm2 = Auto_ViML(train_nlp, target, test_nlp, 
                                    sample_submission=subm,
                                    scoring_parameter='balanced-accuracy',
                                    hyper_param='GS',feature_reduction=True,
                                     Boosting_Flag=None,Binning_Flag=False,
                                    Add_Poly=0, Stacking_Flag=False,
                                        Imbalanced_Flag=False, 
                                    verbose=2)
subm[target] = (testm2['Survived_proba_1']>0.5).astype(int).values
subm.head()
subm.to_csv('sample_submission2.csv',index=False)
m3, feats3, trainm3, testm3 = Auto_ViML(train_nlp, target, test_nlp, 
                                    sample_submission=subm,
                                    scoring_parameter='balanced-accuracy',
                                    hyper_param='RS',feature_reduction=True,
                                     Boosting_Flag=True,Binning_Flag=False,
                                    Add_Poly=0, Stacking_Flag=False,
                                        Imbalanced_Flag=False, 
                                    verbose=2)
subm[target] = (testm3['Survived_proba_1']>0.5).astype(int).values
subm.head()
subm.to_csv('sample_submission3.csv',index=False)
