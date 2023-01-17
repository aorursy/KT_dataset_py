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
# import the necessary libraries

import numpy as np 

import pandas as pd 

import os



# Visualisation libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()



from sklearn.ensemble import VotingClassifier,RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier

from sklearn.preprocessing import LabelEncoder



# Disable warnings 

import warnings

warnings.filterwarnings('ignore')

train_df = pd.read_csv('/kaggle/input/janatahack-healthcare-analytics-ii/Train_hMYJ020/train.csv')

test_df = pd.read_csv('/kaggle/input/janatahack-healthcare-analytics-ii/Test_ND2Q3bm/test.csv')

sub_df = pd.read_csv('/kaggle/input/janatahack-healthcare-analytics-ii/sample_submission_lfbv3c3.csv')





#Training data

print('Training data shape: ', train_df.shape)

train_df.head(5)
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train_df = reduce_mem_usage(train_df)

test_df = reduce_mem_usage(test_df)
print(train_df.shape, test_df.shape)
# * join the datasets

train_df['is_train']  = 1

test_df['Stay'] = -1

test_df['is_train'] = 0
full_df = train_df.append(test_df)

full_df.head()
full_df.isnull().sum()
full_df.fillna('-999',inplace=True)
full_df.columns
full_df.dtypes
cols = [ 'Hospital_type_code',

       'Hospital_region_code', 

       'Department', 'Ward_Type', 'Ward_Facility_Code','City_Code_Patient',

        'Type of Admission',

       'Severity of Illness',  'Age', 'Bed Grade'

       ]

for col in cols:

    if full_df[col].dtype==object:

        print(col)

        lbl = LabelEncoder()

        lbl.fit(list(full_df[col].values.astype('str')))

        full_df[col] = lbl.transform(list(full_df[col].values.astype('str')))
train = full_df[full_df['is_train']==1]

test = full_df[full_df['is_train']==0]

print(train.shape, test.shape)



train_df = train.copy()

test_df = test.copy()

del train, test
#define X and y

X = train_df.drop(['Stay', 'is_train', 'case_id', 'patientid'],axis = 1)

y = train_df.Stay

test_X = test_df.drop(['Stay', 'is_train', 'case_id', 'patientid'],axis = 1)



print(X.columns)
X.head()
clf1 = LGBMClassifier(boosting_type='gbdt', learning_rate=0.1, n_estimators=500, 

                      min_child_samples=20, random_state=1994,  n_jobs=-1, silent=False)

clf1.fit(X,y)

pred=clf1.predict(test_X)
pred
sub_df.head()
# Read the submission file

sub_df['Stay']=pred

sub_df.to_csv('lgb_submission.csv', index=False)
sub_df.head()
sub_df['Stay'].value_counts()
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



name = "lgbm_submission.csv"



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = name):  

    csv = sub_df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)





# create a link to download the dataframe

create_download_link(sub_df)