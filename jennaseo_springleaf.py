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
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from eli5.sklearn import PermutationImportance

from catboost import CatBoostClassifier,Pool

from sklearn.metrics import roc_curve, auc

from IPython.display import display

import matplotlib.patches as patch

import matplotlib.pyplot as plt

from sklearn.svm import NuSVR

from scipy.stats import norm

from sklearn import svm

import lightgbm as lgb

import xgboost as xgb

import seaborn as sns

import pandas as pd

import numpy as np

import warnings

import eli5

import time

import glob

import sys

import os

import gc
print('pandas: {}'.format(pd.__version__))

print('numpy: {}'.format(np.__version__))

print('Python: {}'.format(sys.version))
print(os.listdir("../input/springleaf-marketing-response/"))
#Zipfile 기능사용



import zipfile

#test.csv

with zipfile.ZipFile("../input/springleaf-marketing-response/test.csv.zip","r") as zf:

    zf.extractall(".")

#train.csv

with zipfile.ZipFile("../input/springleaf-marketing-response/train.csv.zip","r") as zf:

    zf.extractall(".")

#sample_submission.csv 

with zipfile.ZipFile("../input/springleaf-marketing-response/sample_submission.csv.zip","r") as zf:

    zf.extractall(".")
from subprocess import check_output

print(check_output(["ls", "test.csv"]).decode("utf8"))

print(check_output(["ls", "train.csv"]).decode("utf8"))

print(check_output(["ls", "sample_submission.csv"]).decode("utf8"))
test=pd.read_csv("./test.csv")

train=pd.read_csv("./train.csv")

sample_Sub=pd.read_csv("./sample_submission.csv")
columns = list(train.columns)



for i in range(1, 1935):

    # i ~ 1, 2, ... 1934

    varname = 'VAR_{:04d}'.format(i)

    if not (varname in columns):

        print(varname)
print(train.shape)

print(train.info())

train.head()
print(test.shape)

print(test.info())

test.head()
%%time 

train.describe()
%%time

test.describe()
def missing_check(data):

    tf=data.isna().sum().any()

    if tf==True:

        total = data.isnull().sum()

        percent = (data.isnull().sum()/data.isnull().count()*100)

        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

        data_type = []

        for col in data.columns:

            dtype = str(data[col].dtype)

            data_type.append(dtype)

        output['Types'] = data_type

        return(np.transpose(output))

    else:

        return(False)
missing_check(train)
missing_check(test)
sns.countplot(train.target)

plt.title("")
#수치

print(train['target'].value_counts())

#비율

print(round(train.target.value_counts() *100/ train.target.count(),2))
#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65

def reduce_mem_usage(df):

    start_mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in df.columns:

        if df[col].dtype != object:  # Exclude strings

            

            # Print current column type

            print("******************************")

            print("Column: ",col)

            print("dtype before: ",df[col].dtype)

            

            # make variables for Int, max and min

            IsInt = False

            mx = df[col].max()

            mn = df[col].min()

            

            # Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(df[col]).all(): 

                NAlist.append(col)

                df[col].fillna(mn-1,inplace=True)  

                   

            # test if column can be converted to an integer

            asint = df[col].fillna(0).astype(np.int64)

            result = (df[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True



            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        df[col] = df[col].astype(np.uint8)

                    elif mx < 65535:

                        df[col] = df[col].astype(np.uint16)

                    elif mx < 4294967295:

                        df[col] = df[col].astype(np.uint32)

                    else:

                        df[col] = df[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        df[col] = df[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        df[col] = df[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        df[col] = df[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        df[col] = df[col].astype(np.int64)    

            

            # Make float datatypes 32 bit

            else:

                df[col] = df[col].astype(np.float32)

            

            # Print new column type

            print("dtype after: ",df[col].dtype)

            print("******************************")

    

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    return df, NAlist
test, NAlist = reduce_mem_usage(test)

print("_________________")

print("")

print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")

print("_________________")

print("")

print(NAlist)
train, NAlist = reduce_mem_usage(train)

print("_________________")

print("")

print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")

print("_________________")

print("")

print(NAlist)
train.info()
test.info()
train_dropna=X_train.dropna(axis=1)

train_dropna=X_train.dropna()

test_dropna=X_train.dropna(axis=1)

test_dropna=X_train.dropna()
text
columns=["target","ID"]

X = train.drop(columns,axis=1)

y = train["target"]
X_test  = test.drop("ID",axis=1)
#테스트용으로 사이즈 추가로 축소

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,test_size=0.5, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,test_size=0.5, random_state=1)
# X_train.info(), X_test.info() -> 73.1+MB
#5-2. Eli5

perm_imp = PermutationImportance(rfc, random_state=1).fit(X_test, y_test)