# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

plt.style.use('fivethirtyeight')



plt.figure(figsize=(25,25))



import pandas_profiling as pp



# Any results you write to the current directory are saved as output.
import gc

gc.collect()
# !pip install pretrainedmodels



%reload_ext autoreload

%autoreload 2

%matplotlib inline



!pip install fastai==1.0.57

import fastai



from fastai import *

from fastai.vision import *

from fastai.tabular import *



# from torchvision.models import *

# import pretrainedmodels



from utils import *

import sys



from fastai.callbacks.hooks import *



from fastai.callbacks.tracker import EarlyStoppingCallback

from fastai.callbacks.tracker import SaveModelCallback
import pandas as pd 

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

from sklearn.manifold import TSNE



from sklearn.experimental import enable_hist_gradient_boosting 

from sklearn.ensemble import HistGradientBoostingClassifier



import gc

from datetime import datetime 

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier

from catboost import CatBoostClassifier

from sklearn import svm

import lightgbm as lgb

from lightgbm import LGBMClassifier

import xgboost as xgb

from xgboost.sklearn import XGBClassifier  

from xgboost.sklearn import XGBRegressor



from scipy.special import erfinv

import matplotlib.pyplot as plt

import torch

from torch.utils.data import *

from torch.optim import *

from fastai.tabular import *

import torch.utils.data as Data

from fastai.basics import *

from fastai.callbacks.hooks import *

from tqdm import tqdm_notebook as tqdm



from hyperopt import hp, tpe

from hyperopt.fmin import fmin

from hyperopt import STATUS_OK



from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.metrics import make_scorer



import warnings

warnings.filterwarnings('ignore')
def to_gauss(x): return np.sqrt(2)*erfinv(x)  #from scipy



def normalize(data, exclude=None):

    # if not binary, normalize it

    norm_cols = [n for n, c in data.drop(exclude, 1).items() if len(np.unique(c)) > 2]

    n = data.shape[0]

    for col in norm_cols:

        sorted_idx = data[col].sort_values().index.tolist()# list of sorted index

        uniform = np.linspace(start=-0.99, stop=0.99, num=n) # linsapce

        normal = to_gauss(uniform) # apply gauss to linspace

        normalized_col = pd.Series(index=sorted_idx, data=normal) # sorted idx and normalized space

        data[col] = normalized_col # column receives its corresponding rank

    return data
df_all = pd.read_csv('../input/train.csv')
df_all.head()
df_all.shape
df_all.columns
df_all['Response'].value_counts()
sns.set_color_codes()

plt.figure(figsize=(12,12))

sns.countplot(df_all.Response).set_title('Dist of Response variables')
df_all.describe()
df_all.dtypes
df_all.shape
f, axes = plt.subplots(1, 2, figsize=(10,5))

sns.boxplot(x = 'BMI', data=df_all,  orient='v' , ax=axes[0])

sns.distplot(df_all['BMI'],  ax=axes[1])
f, axes = plt.subplots(1, 2, figsize=(10,5))

sns.boxplot(x = 'Ins_Age', data=df_all,  orient='v' , ax=axes[0])

sns.distplot(df_all['Ins_Age'],  ax=axes[1])
f, axes = plt.subplots(1, 2, figsize=(10,5))

sns.boxplot(x = 'Ht', data=df_all,  orient='v' , ax=axes[0])

sns.distplot(df_all['Ht'],  ax=axes[1])
f, axes = plt.subplots(1, 2, figsize=(10,5))

sns.boxplot(x = 'Wt', data=df_all,  orient='v' , ax=axes[0])

sns.distplot(df_all['Wt'],  ax=axes[1])
#create a funtion to create new target variable based on conditions

# 0 means reject

# 1 means accept



def new_target(row):

    if (row['Response']<=7):

        val=0

    elif (row['Response']==8):

        val=1

    else:

        val=2

    return val





#create a copy of original dataset

new_data=df_all.copy()



#create a new column

new_data['Final_Response']=new_data.apply(new_target,axis=1)



#print unique values of target variable

print("Unique values in Target Variable: {}".format(new_data.Final_Response.dtype))

print("Unique values in Target Variable: {}".format(new_data.Final_Response.unique()))

print("Total Number of unique values : {}".format(len(new_data.Final_Response.unique())))



#distribution plot for target classes

sns.countplot(x=new_data.Final_Response).set_title('Distribution of rows by response categories')
new_data.drop(['Response'], axis=1, inplace=True)

df_all = new_data

del new_data
df_all.rename(columns={'Final_Response': 'Response'}, inplace=True)
#1

df_all['Product_Info_2_char'] = df_all.Product_Info_2.str[0]

#2

df_all['Product_Info_2_num'] = df_all.Product_Info_2.str[1]



#3

df_all['BMI_Age'] = df_all['BMI'] * df_all['Ins_Age']

#4

df_all['Age_Wt'] = df_all['Ins_Age'] * df_all['Wt']

#5

df_all['Age_Ht'] = df_all['Ins_Age'] * df_all['Ht']



med_keyword_columns = df_all.columns[df_all.columns.str.startswith('Medical_Keyword_')]

#6

df_all['Med_Keywords_Count'] = df_all[med_keyword_columns].sum(axis=1)



#7

df_all['Ins_Age_sq'] = df_all['Ins_Age'] * df_all['Ins_Age']

#8

df_all['Ht_sq'] = df_all['Ht'] * df_all['Ht']

#9

df_all['Wt_sq'] = df_all['Wt'] * df_all['Wt']

#10

df_all['BMI_sq'] = df_all['BMI'] * df_all['BMI']



#11

df_all['Ins_Age_cu'] = df_all['Ins_Age'] * df_all['Ins_Age'] * df_all['Ins_Age']

#12

df_all['Ht_cu'] = df_all['Ht'] * df_all['Ht'] * df_all['Ht']

#13

df_all['Wt_cu'] = df_all['Wt'] * df_all['Wt'] * df_all['Wt']

#14

df_all['BMI_cu'] = df_all['BMI'] * df_all['BMI'] * df_all['BMI']



# BMI Categorization

conditions = [

    (df_all['BMI'] <= df_all['BMI'].quantile(0.25)),

    (df_all['BMI'] > df_all['BMI'].quantile(0.25)) & (df_all['BMI'] <= df_all['BMI'].quantile(0.75)),

    (df_all['BMI'] > df_all['BMI'].quantile(0.75))]



choices = ['under_weight', 'average', 'overweight']

#15

df_all['BMI_Wt'] = np.select(conditions, choices)



# Age Categorization

conditions = [

    (df_all['Ins_Age'] <= df_all['Ins_Age'].quantile(0.25)),

    (df_all['Ins_Age'] > df_all['Ins_Age'].quantile(0.25)) & (df_all['Ins_Age'] <= df_all['Ins_Age'].quantile(0.75)),

    (df_all['Ins_Age'] > df_all['Ins_Age'].quantile(0.75))]



choices = ['young', 'average', 'old']

#16

df_all['Old_Young'] = np.select(conditions, choices)



# Height Categorization

conditions = [

    (df_all['Ht'] <= df_all['Ht'].quantile(0.25)),

    (df_all['Ht'] > df_all['Ht'].quantile(0.25)) & (df_all['Ht'] <= df_all['Ht'].quantile(0.75)),

    (df_all['Ht'] > df_all['Ht'].quantile(0.75))]



choices = ['short', 'average', 'tall']

#17

df_all['Short_Tall'] = np.select(conditions, choices)



# Weight Categorization

conditions = [

    (df_all['Wt'] <= df_all['Wt'].quantile(0.25)),

    (df_all['Wt'] > df_all['Wt'].quantile(0.25)) & (df_all['Wt'] <= df_all['Wt'].quantile(0.75)),

    (df_all['Wt'] > df_all['Wt'].quantile(0.75))]



choices = ['thin', 'average', 'fat']

#18

df_all['Thin_Fat'] = np.select(conditions, choices)



#19

df_all['min'] = df_all[med_keyword_columns].min(axis=1)

#20

df_all['max'] = df_all[med_keyword_columns].max(axis=1)

#21

df_all['mean'] = df_all[med_keyword_columns].mean(axis=1)

#22

df_all['std'] = df_all[med_keyword_columns].std(axis=1)

#23

df_all['skew'] = df_all[med_keyword_columns].skew(axis=1)

#24

df_all['kurt'] = df_all[med_keyword_columns].kurtosis(axis=1)

#25

df_all['med'] = df_all[med_keyword_columns].median(axis=1)
def new_target(row):

    if (row['BMI_Wt']=='overweight') or (row['Old_Young']=='old')  or (row['Thin_Fat']=='fat'):

        val='extremely_risky'

    else:

        val='not_extremely_risky'

    return val



#26

df_all['extreme_risk'] = df_all.apply(new_target,axis=1)
df_all.extreme_risk.value_counts()
# Risk Categorization

conditions = [

    (df_all['BMI_Wt'] == 'overweight') ,

    (df_all['BMI_Wt'] == 'average') ,

    (df_all['BMI_Wt'] == 'under_weight') ]



choices = ['risk', 'non-risk', 'risk']

#27

df_all['risk_bmi'] = np.select(conditions, choices)
df_all.risk_bmi.value_counts()
def new_target(row):

    if (row['BMI_Wt']=='average') or (row['Old_Young']=='average')  or (row['Thin_Fat']=='average') or (row['Short_Tall']=='average'):

        val='average'

    else:

        val='non_average'

    return val



#28

df_all['average_risk'] = df_all.apply(new_target,axis=1)
df_all.average_risk.value_counts()
def new_target(row):

    if (row['BMI_Wt']=='under_weight') or (row['Old_Young']=='young')  or (row['Thin_Fat']=='thin') or (row['Short_Tall']=='short'):

        val='low_end'

    else:

        val='non_low_end'

    return val



#29

df_all['low_end_risk'] = df_all.apply(new_target,axis=1)
df_all.low_end_risk.value_counts()
def new_target(row):

    if (row['BMI_Wt']=='overweight') or (row['Old_Young']=='old')  or (row['Thin_Fat']=='fat') or (row['Short_Tall']=='tall'):

        val='high_end'

    else:

        val='non_high_end'

    return val



#30

df_all['high_end_risk'] = df_all.apply(new_target,axis=1)
df_all.high_end_risk.value_counts()
plt.figure(figsize=(12,10))

sns.countplot(x = 'extreme_risk', hue = 'Response', data = df_all)
plt.figure(figsize=(12,10))

sns.countplot(x = 'average_risk', hue = 'Response', data = df_all)
plt.figure(figsize=(12,10))

sns.countplot(x = 'low_end_risk', hue = 'Response', data = df_all)
plt.figure(figsize=(12,10))

sns.countplot(x = 'high_end_risk', hue = 'Response', data = df_all)
plt.figure(figsize=(12,10))

sns.countplot(x = 'BMI_Wt', hue = 'Response', data = df_all)
plt.figure(figsize=(12,10))

sns.countplot(x = 'Old_Young', hue = 'Response', data = df_all)
plt.figure(figsize=(12,10))

sns.countplot(x = 'Thin_Fat', hue = 'Response', data = df_all)
plt.figure(figsize=(12,10))

sns.countplot(x = 'risk_bmi', hue = 'Response', data = df_all)
from sklearn.manifold import TSNE



def tsne_plot(x1, y1, name="graph.png"):

    tsne = TSNE(n_components=2)

    X_t = tsne.fit_transform(x1)



    plt.figure(figsize=(12, 8))

    #plt.scatter(X_t[np.where(y1 == 8), 0], X_t[np.where(y1 == 8), 1], marker='o', color='red', linewidth='1', alpha=0.8, label='8')

    plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', color='red', linewidth='1', alpha=0.8, label='0')

    plt.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker='o', color='green', linewidth='1', alpha=0.8, label='1')

#     plt.scatter(X_t[np.where(y1 == 3), 0], X_t[np.where(y1 == 3), 1], marker='o', color='yellow', linewidth='1', alpha=0.8, label='3')

#     plt.scatter(X_t[np.where(y1 == 4), 0], X_t[np.where(y1 == 4), 1], marker='o', color='blue', linewidth='1', alpha=0.8, label='4')

#     plt.scatter(X_t[np.where(y1 == 5), 0], X_t[np.where(y1 == 5), 1], marker='o', color='magenta', linewidth='1', alpha=0.8, label='5')

#     plt.scatter(X_t[np.where(y1 == 6), 0], X_t[np.where(y1 == 6), 1], marker='o', color='black', linewidth='1', alpha=0.8, label='6')

#     plt.scatter(X_t[np.where(y1 == 7), 0], X_t[np.where(y1 == 7), 1], marker='o', color='brown', linewidth='1', alpha=0.8, label='7')



    plt.legend(loc='best');

    plt.savefig(name);

    plt.show();



gc.collect()
df_all.shape
df_train = df_all
del df_all
df_train.head()
df_train.drop(['Id'], axis=1, inplace=True)
df_train.dtypes
exclude = ['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 

           'Product_Info_6', 'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 

           'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 

           'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4', 

           'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 'Medical_History_3', 

           'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 'Medical_History_9', 

           'Medical_History_11', 'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 'Medical_History_16', 'Medical_History_17', 

           'Medical_History_18', 'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 'Medical_History_22', 'Medical_History_23', 

           'Medical_History_25', 'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 'Medical_History_30', 

           'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 

           'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 'Medical_History_39', 'Medical_History_40', 'Medical_History_41', 

           'Response', 'Product_Info_2_char', 'Product_Info_2_num', 'BMI_Wt', 'Old_Young', 'Thin_Fat', 'Short_Tall', 'risk_bmi',

          'Medical_Keyword_1', 'Medical_Keyword_2', 'Medical_Keyword_3', 'Medical_Keyword_4',

          'Medical_Keyword_5', 'Medical_Keyword_6', 'Medical_Keyword_7', 'Medical_Keyword_8',

          'Medical_Keyword_9', 'Medical_Keyword_10', 'Medical_Keyword_11', 'Medical_Keyword_12',

          'Medical_Keyword_13', 'Medical_Keyword_14', 'Medical_Keyword_15', 'Medical_Keyword_16',

          'Medical_Keyword_17', 'Medical_Keyword_18', 'Medical_Keyword_19', 'Medical_Keyword_20',

          'Medical_Keyword_21', 'Medical_Keyword_22', 'Medical_Keyword_23', 'Medical_Keyword_24', 'Medical_Keyword_25',

          'Medical_Keyword_26', 'Medical_Keyword_27', 'Medical_Keyword_28', 'Medical_Keyword_29',

          'Medical_Keyword_30', 'Medical_Keyword_31', 'Medical_Keyword_32', 'Medical_Keyword_33',

          'Medical_Keyword_34', 'Medical_Keyword_35', 'Medical_Keyword_36', 'Medical_Keyword_37',

          'Medical_Keyword_38', 'Medical_Keyword_39', 'Medical_Keyword_40', 'Medical_Keyword_41', 

          'Medical_Keyword_42', 'Medical_Keyword_43', 'Medical_Keyword_44',

          'Medical_Keyword_45', 'Medical_Keyword_46', 'Medical_Keyword_47', 'Medical_Keyword_48', 'extreme_risk', 

           'average_risk', 'high_end_risk', 'low_end_risk']



norm_data = normalize(df_train, exclude=exclude)
cont_names = ['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 

              'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 

              'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5', 'Medical_History_1', 

              'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32', 'BMI_Age', 'Med_Keywords_Count',

             'min', 'max', 'mean', 'std', 'skew', 'med', 'kurt', 'Age_Wt', 'Age_Ht', 

              'Ins_Age_sq', 'Ht_sq','Wt_sq',

              'Ins_Age_cu','Ht_cu','Wt_cu', 'BMI_sq', 'BMI_cu'

             ]



dep_var = 'Response'

procs = [FillMissing, Categorify]



cat_names= ['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 

           'Product_Info_6', 'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 

           'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 

           'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4', 

           'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 'Medical_History_3', 

           'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 'Medical_History_9', 

           'Medical_History_11', 'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 'Medical_History_16', 'Medical_History_17', 

           'Medical_History_18', 'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 'Medical_History_22', 'Medical_History_23', 

           'Medical_History_25', 'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 'Medical_History_30', 

           'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 

           'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 'Medical_History_39', 'Medical_History_40', 'Medical_History_41', 

            'Product_Info_2_char', 'Product_Info_2_num', 'BMI_Wt', 'Old_Young', 'Thin_Fat', 'Short_Tall', 'risk_bmi','extreme_risk','average_risk','high_end_risk',

          'Medical_Keyword_1', 'Medical_Keyword_2', 'Medical_Keyword_3', 'Medical_Keyword_4',

          'Medical_Keyword_5', 'Medical_Keyword_6', 'Medical_Keyword_7', 'Medical_Keyword_8',

          'Medical_Keyword_9', 'Medical_Keyword_10', 'Medical_Keyword_11', 'Medical_Keyword_12',

          'Medical_Keyword_13', 'Medical_Keyword_14', 'Medical_Keyword_15', 'Medical_Keyword_16',

          'Medical_Keyword_17', 'Medical_Keyword_18', 'Medical_Keyword_19', 'Medical_Keyword_20',

          'Medical_Keyword_21', 'Medical_Keyword_22', 'Medical_Keyword_23', 'Medical_Keyword_24', 'Medical_Keyword_25',

          'Medical_Keyword_26', 'Medical_Keyword_27', 'Medical_Keyword_28', 'Medical_Keyword_29',

          'Medical_Keyword_30', 'Medical_Keyword_31', 'Medical_Keyword_32', 'Medical_Keyword_33',

          'Medical_Keyword_34', 'Medical_Keyword_35', 'Medical_Keyword_36', 'Medical_Keyword_37',

          'Medical_Keyword_38', 'Medical_Keyword_39', 'Medical_Keyword_40', 'Medical_Keyword_41', 

          'Medical_Keyword_42', 'Medical_Keyword_43', 'Medical_Keyword_44', 'low_end_risk',

          'Medical_Keyword_45', 'Medical_Keyword_46', 'Medical_Keyword_47', 'Medical_Keyword_48'

           ]
df_train.shape, norm_data.shape
valid_sz = 5000

valid_idx = range(len(norm_data)-valid_sz, len(norm_data))



data = (TabularList.from_df(norm_data, procs = procs, cont_names=cont_names, cat_names=cat_names)

        .split_by_idx(valid_idx)

        .label_from_df(cols=dep_var)

        .databunch(bs=1024))
# data.add_test(TabularList.from_df(df_test, cont_names=cont_names))
data.show_batch()
df_t = data.train_ds.inner_df

df_v = data.valid_ds.inner_df
df_t.shape, df_v.shape
df = df_t.append(df_v, ignore_index=True)
df.shape
pd.set_option('float_format', '{:f}'.format)

df.describe()
# Categorical boolean mask

categorical_feature_mask = df.dtypes=='category'

# filter categorical columns using mask and turn it into a list

categorical_cols = df.columns[categorical_feature_mask].tolist()
# import labelencoder

from sklearn.preprocessing import LabelEncoder

# instantiate labelencoder object

le = LabelEncoder()
# apply le on categorical feature columns

df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))

df[categorical_cols].head(10)
sample_size=500

df_grp = df.groupby('Response').apply(lambda x: x.sample(sample_size))
df_grp = df_grp.reset_index(drop=True)
X = df_grp.drop(['Response'], axis = 1).values

Y = df_grp["Response"].values
tsne_plot(X, Y, 'graph')
var = df.columns.values



i = 0

t0 = df.loc[df['Response'] == 0]

t1 = df.loc[df['Response'] == 1]





sns.set_style('whitegrid')

plt.figure()

fig, ax = plt.subplots(30,6,figsize=(60,50))



for feature in var:

    i += 1

    plt.subplot(30,6,i)

    sns.kdeplot(t0[feature], bw=0.5,label="Response = 0")

    sns.kdeplot(t1[feature], bw=0.5,label="Response = 1")

    

    plt.xlabel(feature, fontsize=12)

    locs, labels = plt.xticks()

    plt.tick_params(axis='both', which='major', labelsize=12)

plt.show();
df['Response'].value_counts()
import pandas as pd

import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage

from scipy.spatial.distance import pdist



import numpy as np

import numpy as np

from pandas import *

import matplotlib.pyplot as plt

#from hcluster import pdist, linkage, dendrogram

from numpy.random import rand



X_ = df.T.values #Transpose values 

Y_ = pdist(X_)

Z_ = linkage(Y_)



plt.figure(figsize=(24,24))

#dendrogram(Z, labels = df.columns, orientation='bottom')

fig = ff.create_dendrogram(Z_, labels=df.columns, color_threshold=1.5)

fig.update_layout(width=1500, height=1000)

fig.show()
corr_df = pd.DataFrame(df.drop("Response", axis=1).apply(lambda x: x.corr(df.Response)))
corr_df.columns = ['corr']
corr_df.sort_values(by='corr')
df.head()
df_small = df[['BMI','Medical_Keyword_15', 'Medical_History_4','Medical_History_23', 

              'Product_Info_4','InsuredInfo_6', 'Ht', 'Wt', 'Ins_Age', 'Med_Keywords_Count',

              'extreme_risk', 'high_end_risk', 'low_end_risk', 'Thin_Fat', 'BMI_Age', 'Age_Ht', 'Age_Wt', 'Medical_Keyword_15']]



x = df_small.reset_index(drop=True)



x.columns = ['BMI','Medical_Keyword_15', 'Medical_History_4','Medical_History_23', 

              'Product_Info_4','InsuredInfo_6', 'Ht', 'Wt', 'Ins_Age', 'Med_Keywords_Count',

              'extreme_risk', 'high_end_risk', 'low_end_risk', 'Thin_Fat', 'BMI_Age', 'Age_Ht', 'Age_Wt', 'Medical_Keyword_15']
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

gmix = GaussianMixture(n_components=18, random_state=42, reg_covar=1e-3)

gmix.fit(x)

df['test_cluster'] = gmix.predict(x)
from sklearn.cluster import KMeans,DBSCAN,Birch

gc.collect()

brc = Birch(n_clusters=14)



df_small = df[['BMI','Medical_Keyword_15', 'Medical_History_4','Medical_History_23', 

              'Product_Info_4','InsuredInfo_6', 'Ht', 'Wt', 'Ins_Age', 'Med_Keywords_Count',

              'extreme_risk', 'high_end_risk', 'low_end_risk', 'Thin_Fat']]



x = df_small.reset_index(drop=True)



x.columns = ['BMI','Medical_Keyword_15', 'Medical_History_4','Medical_History_23', 

              'Product_Info_4','InsuredInfo_6', 'Ht', 'Wt', 'Ins_Age', 'Med_Keywords_Count',

              'extreme_risk', 'high_end_risk', 'low_end_risk', 'Thin_Fat']
clustering = brc.fit(x).labels_
df['big_cluster'] = clustering
gc.collect()

from sklearn.cluster import KMeans,DBSCAN,Birch

#from hdbscan import HDBSCAN



x1=df[['BMI','Medical_Keyword_15']].reset_index(drop=True)

x2=df[['Medical_History_4','Medical_History_23']].reset_index(drop=True)

x3=df[['BMI','Med_Keywords_Count']].reset_index(drop=True)

x4=df[['Product_Info_4','InsuredInfo_6']].reset_index(drop=True)

x5=df[['BMI', 'Ins_Age']].reset_index(drop=True)

x6=df[['Thin_Fat', 'Medical_History_15']].reset_index(drop=True)

x7=df[['BMI_Age', 'Age_Ht']].reset_index(drop=True)

x8=df[['BMI_Age', 'Age_Wt']].reset_index(drop=True)

x9=df[['BMI', 'Wt']].reset_index(drop=True)

x10=df[['BMI', 'Ht']].reset_index(drop=True)



x11=df[['extreme_risk', 'Medical_History_23']].reset_index(drop=True)

x12=df[['extreme_risk', 'Medical_History_4']].reset_index(drop=True)

x13=df[['extreme_risk','Medical_Keyword_15']].reset_index(drop=True)

x14=df[['extreme_risk','Med_Keywords_Count']].reset_index(drop=True)



x15=df[['high_end_risk', 'Medical_History_23']].reset_index(drop=True)

x16=df[['high_end_risk', 'Medical_History_4']].reset_index(drop=True)

x17=df[['high_end_risk','Medical_Keyword_15']].reset_index(drop=True)

x18=df[['high_end_risk','Med_Keywords_Count']].reset_index(drop=True)



x19=df[['low_end_risk', 'Medical_History_23']].reset_index(drop=True)

x20=df[['low_end_risk', 'Medical_History_4']].reset_index(drop=True)

x21=df[['low_end_risk','Medical_Keyword_15']].reset_index(drop=True)

x22=df[['low_end_risk','Med_Keywords_Count']].reset_index(drop=True)



x23=df[['extreme_risk', 'Product_Info_4']].reset_index(drop=True)

x24=df[['extreme_risk', 'InsuredInfo_6']].reset_index(drop=True)

x25=df[['extreme_risk','BMI']].reset_index(drop=True)

x26=df[['extreme_risk','Thin_Fat']].reset_index(drop=True)



x27=df[['high_end_risk', 'Product_Info_4']].reset_index(drop=True)

x28=df[['high_end_risk', 'InsuredInfo_6']].reset_index(drop=True)

x29=df[['high_end_risk','BMI']].reset_index(drop=True)

x30=df[['high_end_risk','Thin_Fat']].reset_index(drop=True)



x31=df[['low_end_risk', 'Product_Info_4']].reset_index(drop=True)

x32=df[['low_end_risk', 'InsuredInfo_6']].reset_index(drop=True)

x33=df[['low_end_risk','BMI']].reset_index(drop=True)

x34=df[['low_end_risk','Thin_Fat']].reset_index(drop=True)



x1.columns=['bmi','m_k_15'];x2.columns=['m_h_4','m_h_23'];x3.columns=['bmi','med_key'];x4.columns=['i_i_6','p_i_4']

x5.columns=['bmi', 'age']; x6.columns=['thinfat', 'mh15']; x7.columns = ['bmiage', 'ageht']; x8.columns = ['bmiage', 'agewt'];

x9.columns=['bmi', 'wt']; x10.columns=['bmi', 'ht']; x11.columns=['xrisk', 'mh23']; x12.columns=['xrisk', 'mh4'];

x13.columns=['xrisk', 'mk15']; x14.columns=['xrisk', 'mkc'];x15.columns=['hrisk', 'mh23']; x16.columns=['hrisk', 'mh4'];

x17.columns=['hrisk', 'mk15']; x18.columns=['hrisk', 'mkc'];x19.columns=['lrisk', 'mh23']; x20.columns=['lrisk', 'mh4'];

x21.columns=['lrisk', 'mk15']; x22.columns=['lrisk', 'mkc'];x23.columns=['xrisk', 'pi4']; x24.columns=['xrisk', 'ii6'];

x25.columns=['xrisk', 'bmi']; x26.columns=['xrisk', 'tf'];x27.columns=['hrisk', 'pi4']; x28.columns=['hrisk', 'ii6'];

x29.columns=['hrisk', 'bmi']; x30.columns=['hrisk', 'tf'];x31.columns=['lrisk', 'pi4']; x32.columns=['lrisk', 'ii6'];

x33.columns=['lrisk', 'bmi']; x34.columns=['lrisk', 'tf']



brc = Birch(n_clusters=2)



clustering1 = brc.fit(x1).labels_

clustering2 = brc.fit(x2).labels_

clustering3 = brc.fit(x3).labels_

clustering4 = brc.fit(x4).labels_

clustering5 = brc.fit(x5).labels_

clustering6 = brc.fit(x6).labels_

clustering7 = brc.fit(x7).labels_

clustering8 = brc.fit(x8).labels_

clustering9 = brc.fit(x9).labels_

clustering10 = brc.fit(x10).labels_

clustering11 = brc.fit(x11).labels_

clustering12 = brc.fit(x12).labels_

clustering13 = brc.fit(x13).labels_

clustering14 = brc.fit(x14).labels_

clustering15 = brc.fit(x15).labels_

clustering16 = brc.fit(x16).labels_

clustering17 = brc.fit(x17).labels_

clustering18 = brc.fit(x18).labels_

clustering19 = brc.fit(x19).labels_

clustering20 = brc.fit(x20).labels_

clustering21 = brc.fit(x21).labels_

clustering22 = brc.fit(x22).labels_

clustering23 = brc.fit(x23).labels_

clustering24 = brc.fit(x24).labels_

clustering25 = brc.fit(x25).labels_

clustering26 = brc.fit(x26).labels_

clustering27 = brc.fit(x27).labels_

clustering28 = brc.fit(x28).labels_

clustering29 = brc.fit(x29).labels_

clustering30 = brc.fit(x30).labels_

clustering31 = brc.fit(x31).labels_

clustering32 = brc.fit(x32).labels_

clustering33 = brc.fit(x33).labels_

clustering34 = brc.fit(x34).labels_



df['bmi_mk15'] = clustering1

df['mh4_mh23'] = clustering2

df['bmi_medkey'] = clustering3

df['ii6_pi_4'] = clustering4

df['bmi_age'] = clustering5

df['thinfat_mh15'] = clustering6

df['bmiage_ageht'] = clustering7

df['bmiage_agewt'] = clustering8

df['bmiwt'] = clustering9

df['bmiht'] = clustering10

df['xrisk_mh23'] = clustering11

df['xrisk_mh4'] = clustering12

df['xrisk_mk15'] = clustering13

df['xrisk_mkc'] = clustering14

df['hrisk_mh23'] = clustering15

df['hrisk_mh4'] = clustering16

df['hrisk_mk15'] = clustering17

df['hrisk_mkc'] = clustering18

df['lrisk_mh23'] = clustering19

df['lrisk_mh4'] = clustering20

df['lrisk_mk15'] = clustering21

df['lrisk_mkc'] = clustering22

df['xrisk_pi4'] = clustering23

df['xrisk_ii6'] = clustering24

df['xrisk_bmi'] = clustering25

df['xrisk_tf'] = clustering26

df['hrisk_pi4'] = clustering27

df['hrisk_ii6'] = clustering28

df['hrisk_bmi'] = clustering29

df['hrisk_tf'] = clustering30

df['lrisk_pi4'] = clustering31

df['lrisk_ii6'] = clustering32

df['lrisk_bmi'] = clustering33

df['lrisk_tf'] = clustering34



gc.collect()
df.head(3)
df.shape
df.columns[df.isna().any()].tolist()
df.shape
df.columns
def correlation(df, threshold):

    col_corr = set() # Set of all the names of deleted columns

    corr_matrix = df.corr()

    for i in range(len(corr_matrix.columns)):

        for j in range(i):

            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):

                colname = corr_matrix.columns[i] # getting the name of column

                col_corr.add(colname)

                if colname in df.columns:

                    del df[colname] # deleting the column from the dataset



    print(df.shape)
correlation(df, 0.95)

df.shape
df.columns
X = df.drop(['Response'], axis=1).values

Y = df['Response'].values
from sklearn.feature_selection import SelectFromModel







forest_1 = SelectFromModel(LGBMClassifier( n_estimators=200, 

                          objective='binary', class_weight='balanced', 

                         ), 

                         threshold='2*median')







forest_2 = SelectFromModel(ExtraTreesClassifier(bootstrap=True, criterion='gini', max_depth=10, max_features='auto',class_weight='balanced',

                              

            max_leaf_nodes=None, min_impurity_decrease=0.0,

            min_impurity_split=None, min_samples_leaf=20,

            min_samples_split=7, min_weight_fraction_leaf=0.0,

            n_estimators=200, n_jobs=1, oob_score=False, random_state=42,

            verbose=0, warm_start=False), 

                         threshold='2*median')







forest_3 = SelectFromModel(XGBClassifier(objective="binary:logistic", random_state=42, n_estimators=200,

                       reg_alpha=1, colsample_bylevel=0.7, colsample_bytree=0.7, gamma=5), 

                         threshold='2*median')



forest_1.fit(X, Y)

forest_2.fit(X, Y)

forest_3.fit(X, Y)
gc.collect()
df_without_label = df.drop(['Response'], axis=1)

selected_feat_1= df_without_label.columns[(forest_1.get_support())]

selected_feat_2= df_without_label.columns[(forest_2.get_support())]

selected_feat_3= df_without_label.columns[(forest_3.get_support())]
print(selected_feat_1), print(selected_feat_2), print(selected_feat_3)

print(len(selected_feat_1)), print(len(selected_feat_2)), print(len(selected_feat_3))

print(len(selected_feat_1) + len(selected_feat_2) + len(selected_feat_3))
selected_feat = selected_feat_1.union(selected_feat_2)

len(selected_feat)
selected_feat_new = selected_feat.union(selected_feat_3)

len(selected_feat_new)
importances = forest_1.estimator_.feature_importances_



data={'Feature_Name':df.drop(['Response'], axis=1).columns,

      'Feature_Importance': importances

     }



feature_df=pd.DataFrame(data)



feature_df.sort_values(by=['Feature_Importance'],ascending=False,inplace=True)



fig, ax = plt.subplots(figsize=(20,25))

sns.barplot(data=feature_df,y='Feature_Name',x='Feature_Importance')
importances = forest_2.estimator_.feature_importances_



data={'Feature_Name':df.drop(['Response'], axis=1).columns,

      'Feature_Importance': importances

     }



feature_df=pd.DataFrame(data)



feature_df.sort_values(by=['Feature_Importance'],ascending=False,inplace=True)



fig, ax = plt.subplots(figsize=(20,25))

sns.barplot(data=feature_df,y='Feature_Name',x='Feature_Importance')
importances = forest_3.estimator_.feature_importances_



data={'Feature_Name':df.drop(['Response'], axis=1).columns,

      'Feature_Importance': importances

     }



feature_df=pd.DataFrame(data)



feature_df.sort_values(by=['Feature_Importance'],ascending=False,inplace=True)



fig, ax = plt.subplots(figsize=(20,25))

sns.barplot(data=feature_df,y='Feature_Name',x='Feature_Importance')
df[selected_feat_new].head()
feature_mask_1 = df[selected_feat_new].dtypes=='int64'

feature_mask_2 = df[selected_feat_new].dtypes == 'float64'





int_cols = df[selected_feat_new].columns[feature_mask_1].tolist()

#int_cols = int_cols.remove('Response')

float_cols = df[selected_feat_new].columns[feature_mask_2].tolist()
cont_names = float_cols



dep_var = 'Response'

procs = [FillMissing, Categorify]



cat_names = int_cols
df.Response.value_counts()
df_sel_feat = df[selected_feat_new]

df_sel_feat['Response'] = df['Response']

df_sel_feat.head()
df_sel_feat.shape
var = df_sel_feat.columns.values



i = 0

t0 = df_sel_feat.loc[df_sel_feat['Response'] == 0]

t1 = df_sel_feat.loc[df_sel_feat['Response'] == 1]





sns.set_style('whitegrid')

plt.figure()

fig, ax = plt.subplots(24,4,figsize=(30,30), dpi=60)



for feature in var:

    i += 1

    plt.subplot(24,4,i)

    sns.kdeplot(t0[feature], bw=0.5,label="Response = 0")

    sns.kdeplot(t1[feature], bw=0.5,label="Response = 1")

    

    plt.xlabel(feature, fontsize=12,)

    locs, labels = plt.xticks()

    plt.tick_params(axis='both', which='major', labelsize=12)

plt.show();
df_sel_feat.shape
df_sel_feat.head(2)
df_sel_feat_wo_response = df_sel_feat.drop(['Response'], axis=1)

X = df_sel_feat.drop(['Response'], axis=1)

Y = df_sel_feat['Response']



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify = Y, random_state=42)
model = XGBClassifier(objective="binary:logistic", random_state=42, n_estimators=200,

                       reg_alpha=1, colsample_bylevel=0.7, colsample_bytree=0.7, gamma=5)



model_xgb = model.fit(X_train, y_train)
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(model_xgb).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = df_sel_feat_wo_response.columns.tolist(), top=100)
from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots



# Create the data that we will plot

pdp_goals = pdp.pdp_isolate(model=model_xgb, dataset=X_test, model_features=X_test.columns.tolist(), feature='BMI')



# plot it

pdp.pdp_plot(pdp_goals, 'BMI')

plt.show()
pdp_goals = pdp.pdp_isolate(model=model_xgb, dataset=X_test, model_features=X_test.columns.tolist(), feature='Medical_History_15')



# plot it

pdp.pdp_plot(pdp_goals, 'Medical_History_15')

plt.show()


pdp_goals = pdp.pdp_isolate(model=model_xgb, dataset=X_test, model_features=X_test.columns.tolist(), feature='Medical_Keyword_15')



# plot it

pdp.pdp_plot(pdp_goals, 'Medical_Keyword_15')

plt.show()
pdp_goals = pdp.pdp_isolate(model=model_xgb, dataset=X_test, model_features=X_test.columns.tolist(), feature='Product_Info_4')



# plot it

pdp.pdp_plot(pdp_goals, 'Product_Info_4')

plt.show()
pdp_goals = pdp.pdp_isolate(model=model_xgb, dataset=X_test, model_features=X_test.columns.tolist(), feature='Medical_History_4')



# plot it

pdp.pdp_plot(pdp_goals, 'Medical_History_4')

plt.show()
pdp_goals = pdp.pdp_isolate(model=model_xgb, dataset=X_test, model_features=X_test.columns.tolist(), feature='Medical_History_23')



# plot it

pdp.pdp_plot(pdp_goals, 'Medical_History_23')

plt.show()
import shap



explainer = shap.TreeExplainer(model_xgb)

shap_values = explainer.shap_values(X_test)



shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)
def policy_acceptance_factors(model, policyholder):



    explainer = shap.TreeExplainer(model_xgb)

    shap_values = explainer.shap_values(policyholder)

    shap.initjs()

    return shap.force_plot(explainer.expected_value, shap_values, policyholder)
data_for_prediction = X_test.iloc[1,:].astype(float)

policy_acceptance_factors(model_xgb, data_for_prediction)
data_for_prediction = X_test.iloc[6,:].astype(float)

policy_acceptance_factors(model_xgb, data_for_prediction)
data_for_prediction = X_test.iloc[12,:].astype(float)

policy_acceptance_factors(model_xgb, data_for_prediction)
shap_values = shap.TreeExplainer(model_xgb).shap_values(X_test)

shap.dependence_plot("BMI", shap_values, X_test)
shap.dependence_plot("Medical_History_15", shap_values, X_test)
shap.dependence_plot("Medical_Keyword_15", shap_values, X_test)
shap.dependence_plot("Medical_History_23", shap_values, X_test)
shap.dependence_plot("Medical_History_4", shap_values, X_test)
shap.dependence_plot("bmi_mk15", shap_values, X_test)
shap.dependence_plot("Product_Info_4", shap_values, X_test)
shap_values = explainer.shap_values(X_train.iloc[:100])

shap.force_plot(explainer.expected_value, shap_values, X_test.iloc[:])
valid_sz = 5000

valid_idx = range(len(df_sel_feat)-valid_sz, len(df_sel_feat))



data = (TabularList.from_df(df_sel_feat, procs = procs, cont_names=cont_names, cat_names=cat_names)

        .split_by_rand_pct(0.1, seed=42)

        .label_from_df(cols=dep_var)

        .databunch(bs=1024*4)) 
from fastai.callbacks import *



auroc = AUROC()



learn = tabular_learner(data, layers=[200, 100], metrics=[auroc], 

                        ps=[0.3, 0.3], emb_drop=0.3)
learn.loss_func = LabelSmoothingCrossEntropy()
learn.lr_find()

learn.recorder.plot(suggestion=True)
lr = 1e-2

learn.fit_one_cycle(7, max_lr=lr,  pct_start=0.5, wd = 0.75)
learn.lr_find()

learn.recorder.plot(suggestion=True)
lr=1e-4

learn.fit_one_cycle(7, max_lr=lr,  pct_start=0.5, wd = 1)
learn.lr_find()

learn.recorder.plot(suggestion=True)
lr=1e-5

learn.fit_one_cycle(5, max_lr=lr,  pct_start=0.5, wd = 1.)
lr=1e-7

learn.fit_one_cycle(5, max_lr=lr,  pct_start=0.5, wd = 1.)
learn.recorder.plot_losses()
learn.save('1st-round')

learn.load('1st-round')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(8,8), dpi=60)
gc.collect()
data_init = (TabularList.from_df(df_sel_feat, procs = procs, cont_names=cont_names, cat_names=cat_names)

        .split_by_rand_pct(0.1, seed=42)

        .label_from_df(cols=dep_var)

        .databunch(bs=1024))
x = int(len(df_sel_feat)*.9)
train_df = df_sel_feat.iloc[:x]

test_df = df_sel_feat.iloc[x:]
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
val_pct = []

test_pct = []

roc_auc = AUROC()



for train_index, val_index in skf.split(train_df.index, train_df[dep_var]):

    data_fold = (TabularList.from_df(train_df, cat_names=cat_names.copy(),

                                  cont_names=cont_names.copy(), procs=procs,

                                  processor=data_init.processor) # Very important

              .split_by_idxs(train_index, val_index)

              .label_from_df(cols=dep_var)

              .databunch())

    

    data_test = (TabularList.from_df(test_df, cat_names=cat_names.copy(),

                                  cont_names=cont_names.copy(), procs=procs,

                                  processor=data_init.processor) # Very important

              .split_none()

              .label_from_df(cols=dep_var))

    

    data_test.valid = data_test.train

    data_test = data_test.databunch()

    

    learn_f = tabular_learner(data_fold, layers=[200, 100], metrics=[auroc], 

                        ps=[0.3, 0.3], emb_drop=0.3)

    

    learn_f.fit_one_cycle(5, max_lr=1e-3,  pct_start=0.5, wd = 1)

    

    _, val = learn_f.validate()

    

    learn_f.data.valid_dl = data_test.valid_dl

    

    _, test = learn_f.validate()

    

    val_pct.append(val.numpy())

    test_pct.append(test.numpy())
print(f'Validation\nmean: {np.mean(val_pct)}\nstd: {np.std(val_pct)}')


print(f'Test\nmean: {np.mean(test_pct)}\nstd: {np.std(test_pct)}')
class SaveFeatures():

    features=None

    def __init__(self, m): 

        self.hook = m.register_forward_hook(self.hook_fn)

        self.features = None

    def hook_fn(self, module, input, output): 

        out = output.detach().cpu().numpy()

        if isinstance(self.features, type(None)):

            self.features = out

        else:

            self.features = np.row_stack((self.features, out))

    def remove(self): 

        self.hook.remove()
sf = SaveFeatures(learn.model.layers[4])

_= learn.get_preds(data.train_ds)



label = [data.classes[x] for x in (list(data.train_ds.y.items))]

df_new = pd.DataFrame({'label': label})

array = np.array(sf.features)

x=array.tolist()

df_new['img_repr'] = x



d2 = pd.DataFrame(df_new.img_repr.values.tolist(), index = df_new.index).rename(columns = lambda x: 'img_repr{}'.format(x+1))

df_new_2 = df_new.join(d2)

df_new_2.drop(['img_repr'], axis=1, inplace=True)



sample_size=500

df_grp = df_new_2.groupby('label').apply(lambda x: x.sample(sample_size))

X = df_grp.drop(['label'], axis = 1).values

Y = df_grp["label"].values

tsne_plot(X, Y, "original.png")