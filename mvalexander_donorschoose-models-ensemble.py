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
#override with LB
EDA_LB_val = .73395
print("EDA_LB_val: {}".format(EDA_LB_val))

DEEP_LB_val = .79336
print("DEEP_LB_val: {}".format(DEEP_LB_val))

WORD_LB_val = .78723
print("WORD_LB_val: {}".format(WORD_LB_val))
from scipy.special import expit, logit
 
almost_zero = 1e-10
almost_one  = 1-almost_zero
EDA_input_file = "../input/fork-of-donorschoose-lr-model-cat-smote/LR_stack_glove100_submission.csv"
DEEP_input_file = "../input/fork-of-donorschoose-full-cnn-yes-smote/deep_model_fasttext300_fullSMOTE_submission.csv"
WORD_input_file = "../input/donorschoose-word-model/WORD_model_smote_fas_submission.csv"

EDA_df = pd.read_csv(EDA_input_file).rename(columns={'project_is_approved': 'EDA_project_is_approved'})
DEEP_df = pd.read_csv(DEEP_input_file).rename(columns={'project_is_approved': 'DEEP_project_is_approved'})
WORD_df = pd.read_csv(WORD_input_file).rename(columns={'project_is_approved': 'WORD_project_is_approved'})
ensemble_df = pd.merge(EDA_df, DEEP_df, on='id')
ensemble_df = pd.merge(ensemble_df, WORD_df, on='id')
ensemble_df.head()
power = 68
EDA_weights = EDA_LB_val ** power
DEEP_weights = DEEP_LB_val ** power
WORD_weights = WORD_LB_val ** power

EDA_numbers = ensemble_df['EDA_project_is_approved'].clip(almost_zero,almost_one).apply(logit) * EDA_weights
DEEP_numbers = ensemble_df['DEEP_project_is_approved'].clip(almost_zero,almost_one).apply(logit) * DEEP_weights
WORD_numbers = ensemble_df['WORD_project_is_approved'].clip(almost_zero,almost_one).apply(logit) * WORD_weights

totalweights = EDA_weights + DEEP_weights + WORD_weights

ensemble_df['project_is_approved'] = (EDA_numbers + DEEP_numbers + WORD_numbers) / totalweights
ensemble_df['project_is_approved'] = ensemble_df['project_is_approved'].apply(expit)
ensemble_df[['id', 'project_is_approved']].to_csv("ensemble_EDA_plus_DEEPfasttext300_plus_WORDfasttext_submission.csv", index=False)
ensemble_df.head()
