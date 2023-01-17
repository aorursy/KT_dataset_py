# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nmpy_plot as nmp

import nmpy_df as nmd

import nmpy_corr as nmc

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

from sklearn.experimental import enable_iterative_imputer 

from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer



from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
app_train = pd.read_csv('../input/home-credit-default-risk/application_train.csv', sep=',', encoding='utf-8')

app_test = pd.read_csv('../input/home-credit-default-risk/application_test.csv', sep=',', encoding='utf-8')
print('application_train :', app_train.shape)

print('application_test :', app_test.shape)
app_train
app_train_cropped = app_train.drop(columns=app_train.columns[116:122].tolist())

app_train_cropped.drop(columns=app_train_cropped.columns[96:116].tolist(), inplace=True)

app_train_cropped.drop(columns=app_train_cropped.columns[44:91].tolist(), inplace=True)

app_train_cropped.drop(columns=app_train_cropped.columns[32:34].tolist(), inplace=True)

app_train_cropped.drop(columns=app_train_cropped.columns[22:28].tolist(), inplace=True)

app_train_cropped.shape
nmd.df_overview(app_train_cropped, obj_ncols=2)
int2cat = ['REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_OWN_ASSET','REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY']

large_cat = ['OCCUPATION_TYPE','ORGANIZATION_TYPE','DEF_30_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','CNT_FAM_MEMBERS','CNT_CHILDREN']



app_train_dist = app_train_cropped.iloc[:,2:].copy()

for feature in app_train_dist.dtypes.index:

    if feature == "NAME_EDUCATION_TYPE":

        nmp.cat_per_target_bar(app_train_cropped,feature,'TARGET', root_path='../working/', sort_cat=['Academic degree','Higher education','Incomplete higher','Secondary / secondary special','Lower secondary'])

        nmp.cat_distrib_bar(app_train_cropped,feature, root_path='../working/')

    elif feature in int2cat:

        #print('int to cat =', feature)

        nmp.cat_per_target_bar(app_train_cropped,feature,'TARGET', root_path='../working/', sort_cat='alphabet')  

        nmp.cat_distrib_bar(app_train_cropped,feature, root_path='../working/')

    elif feature in large_cat:

        #print('int to cat =', feature)

        nmp.cat_per_target_bar(app_train_cropped,feature,'TARGET',bar_width=0.05, root_path='../working/', sort_cat='count') 

        nmp.cat_distrib_bar(app_train_cropped,feature, root_path='../working/')

    elif app_train_dist[feature].dtypes == 'object':

        #print('cat =', feature)

        nmp.cat_per_target_bar(app_train_cropped,feature,'TARGET', root_path='../working/', sort_cat='count')

        nmp.cat_distrib_bar(app_train_cropped,feature, root_path='../working/')

    elif app_train_dist[feature].dtypes == 'int64' or app_train_dist[feature].dtypes == 'float64':

        #print('num =', feature)

        nmp.num_per_target_kde(app_train_cropped,feature,'TARGET',y_scale='linear', root_path='../working/')

        nmp.num_per_target_kde(app_train_cropped,feature,'TARGET',y_scale='log', root_path='../working/')

        nmp.num_per_target_hist(app_train_cropped,feature,'TARGET', stacked=True, y_scale='linear', root_path='../working/', sort_cat=True)

        nmp.num_per_target_hist(app_train_cropped,feature,'TARGET', stacked=False, y_scale='log', root_path='../working/', sort_cat=True)
nmc.associations(app_train_cropped,figsize=(30,30),theil_u=True, mark_columns=True, clustering=True)