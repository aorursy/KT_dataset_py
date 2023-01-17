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
df = pd.read_csv('../input/payner/payner.csv', encoding='ISO-8859-2')

df.head()
report_df = ProfileReport(df)

report_df
index_int_float = ['popularity', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration', 'time_signature']      



plt.figure(figsize=[20,12])

i = 1

for col in index_int_float :

    plt.subplot(4,10,i)

    sns.violinplot(x=col, data= df, orient='v')

    sns.despine()

    i = i+1

plt.tight_layout()

plt.show()
index_str = ['track_id','track_name', 'artist_1', 'artist_2', 'artist_3', 'datetime', 'time_signature']



plt.figure(figsize=[30,10])

i = 1

for col in index_str :

    plt.subplot(4,10,i)

    sns.scatterplot(x=col, y = 'popularity' ,data= df)

    sns.despine()

    i = i+1

plt.tight_layout()

plt.show()


int_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration', 'time_signature', 'mode']

        



float_features = [ ]



obj_features = ['track_name', 'track_id', 'artist_1', 'artist_2', 'artist_3', 'datetime', 'time_signature']



exp_reg = setup(df, #Train Data

                target = 'popularity',  #Target

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
plot_model(tuned_lightgbm, plot = 'learning')
plot_model(tuned_xgboost, plot = 'learning')
plot_model(tuned_ridge, plot = 'learning')
blender = blend_models(estimator_list = [tuned_br, tuned_lightgbm, tuned_xgboost, tuned_ridge])
display(plot_model(blender, plot = 'learning'))
predictions = predict_model(blender, data = df)

df['popularity'] = np.expm1(predictions['Label'])

df.to_csv('submission.csv',index=False)