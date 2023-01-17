!pip install pycaret
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pycaret.regression import *

import numpy as np 

import pandas as pd 

from pandas_profiling import ProfileReport 

import seaborn as sns

import matplotlib.pyplot as plt

import warnings



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/hackathon/task_2-Tuberculosis_infection_estimates_for_2018.csv', encoding='utf8')

df.head()
report_df = ProfileReport(df)

report_df
index_int_float = ['iso_numeric', 'year', 'e_hh_size', 'prevtx_data_available', 'newinc_con04_prevtx', 'ptsurvey_newinc', 'ptsurvey_newinc_con04_prevtx', 'e_prevtx_eligible', 'e_prevtx_eligible_lo', 'e_prevtx_eligible_hi', 'e_prevtx_kids_pct', 'e_prevtx_kids_pct_lo', 'e_prevtx_kids_pct_hi']      



plt.figure(figsize=[20,12])

i = 1

for col in index_int_float :

    plt.subplot(4,10,i)

    sns.violinplot(x=col, data= df, orient='v')

    sns.despine()

    i = i+1

plt.tight_layout()

plt.show()
index_str = ['g_whoregion', 'country', 'iso2', 'iso3', 'source_hh']



plt.figure(figsize=[30,10])

i = 1

for col in index_str :

    plt.subplot(4,10,i)

    sns.scatterplot(x=col, y = 'e_hh_size' ,data= df)

    sns.despine()

    i = i+1

plt.tight_layout()

plt.show()
int_features = ['iso_numeric', 'year', 'prevtx_data_available', 'newinc_con04_prevtx', 'ptsurvey_newinc', 'ptsurvey_newinc_con04_prevtx', 'e_prevtx_eligible', 'e_prevtx_eligible_lo', 'e_prevtx_eligible_hi', 'e_prevtx_kids_pct', 'e_prevtx_kids_pct_lo', 'e_prevtx_kids_pct_hi']

        



float_features = [ ]



obj_features = ['g_whoregion', 'country', 'iso2', 'iso3', 'source_hh']



exp_reg = setup(df, #Train Data

                target = 'e_hh_size',  #Target

                categorical_features = obj_features, # Categorical Features

                numeric_features = int_features + float_features, # Numeric Features

                normalize = True, # Normalize Dataset

                remove_outliers = True, # Remove 5% Outliers

                remove_multicollinearity = True, # Remove Multicollinearity

                silent = True # Process Automation

               )
compare_models(blacklist = ['tr', 'catboost'], sort = 'RMSLE')
model_br = create_model('br')

model_lightgbm = create_model('lightgbm')

model_xgboost = create_model('xgboost')

model_ridge = create_model('ridge')
tuned_br = tune_model('br')

tuned_lightgbm = tune_model('lightgbm')

tuned_xgboost = tune_model('xgboost')

tuned_ridge = tune_model('ridge')
plot_model(tuned_br, plot = 'learning')
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRGF3rClhVjQr2cVSoBEbwOs4eaQLl3KD8CeQ&usqp=CAU',width=400,height=400)