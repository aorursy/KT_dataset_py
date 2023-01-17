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
#
# Hi, I am Practising Data Science Concepts and below is the Program that performs 
# Preprocessing on house-prices-advanced-regression-techniques, a dataset from Kaggle
#
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
df_train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df_test=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df_train
df_train.shape,df_test.shape
df_train.isna().sum()
null_value_treatment=(df_train.isna().sum()/df_train.shape[0])*100
null_value_treatment
#null_values_threshold=int(input("Please enter the acceptable threshold limit of null values::"))
null_values_threshold=30
dropped_columns=null_value_treatment[null_value_treatment>null_values_threshold].index
dropped_columns
retained_columns=null_value_treatment[null_value_treatment<null_values_threshold].index
retained_columns
len(dropped_columns),len(retained_columns),df_train.shape[1]
numerical_columns=df_train[retained_columns].describe().columns
numerical_columns
cat_columns=df_train[retained_columns].describe(include='object').columns
cat_columns
len(numerical_columns),len(cat_columns),len(retained_columns)
#
# Below Code Snipet Identifies the Numerical Continous and Class Columns 
# & Categorical More Uniques and Class Columns
#
numerical_cont=[]
numerical_class=[]
for i in numerical_columns:
    if df_train[i].nunique()>8:
        numerical_cont.append(i)
    else:
        numerical_class.append(i)
cat_moreuniques=[]
cat_class=[]
for j in cat_columns:
    if df_train[j].nunique()>8:
        cat_moreuniques.append(j)
    else:
        cat_class.append(j)
len(numerical_cont),len(numerical_class),len(cat_moreuniques),len(cat_class)
#
# NULL Values Treatment
#
for i in numerical_cont:
    df_train[i].fillna(df_train[i].median(),inplace=True)
    print("numerical_cont ::",i,"-- median -->",df_train[i].median())
for j in numerical_class:
    df_train[j].fillna(df_train[j].mode().values[0],inplace=True)
    print("numerical_class ::",j,"-- mode -->",df_train[j].mode().values[0])    
for k in cat_moreuniques:
    df_train[k].fillna(df_train[k].mode().values[0],inplace=True)
    print("categorical more uniques ::",k,"-- mode -->",df_train[j].mode().values[0])    
for l in cat_class:
    df_train[l].fillna(df_train[l].mode().values[0],inplace=True)
    print("categorical class ::",l,"-- mode -->",df_train[j].mode().values[0])
# To ensure that all the Null values were replaced
df_train[retained_columns].isna().sum().sum()
# Outlier Values treatment
for i in numerical_cont:
    x=df_train[i].values
    qrt_1=np.quantile(x,0.25)
    qrt_3=np.quantile(x,0.75)
    iqr=qrt_3-qrt_1
    utv=qrt_3 + 1.5 * iqr
    ltv=qrt_1 - 1.5 * iqr
    if np.median(x) == 0:
        print(i,":: qrt_1-->",qrt_1,":: qrt_3-->",qrt_3,":: utv-->",utv,":: ltv-->",ltv,":: median -->",np.median(x))
    outlier_treated_values=[]
    for j in df_train[i]:
        if j > utv or j < ltv:
            outlier_treated_values.append(np.median(x))
        else:
            outlier_treated_values.append(j)
    df_train[i]=outlier_treated_values
#
# Label Encoding and Scaling Process
#
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
# Label Encoding

for i in cat_columns.values:
    le=LabelEncoder()
    le.fit(df_train[i])
    x=le.transform(df_train[i])
    df_train[i]=x
# Actual Numerical Values Description
df_train[numerical_columns].describe()
df_minmaxscalar=df_train[numerical_columns]
df_stdscalar=df_train[numerical_columns]
# MinMax Scaling applied on a dummy numerical dataframe
for i in numerical_columns.values:
    le=MinMaxScaler()
    le.fit(pd.DataFrame(df_minmaxscalar[i]))
    x=le.transform(pd.DataFrame(df_minmaxscalar[i]))
    df_minmaxscalar[i]=x
# MinMaxScaling Numerical Values Description
df_minmaxscalar[numerical_columns].describe()
# Standard Scaling applied on a dummy numerical dataframe

for i in numerical_columns.values:
    le=StandardScaler()
    le.fit(pd.DataFrame(df_stdscalar[i]))
    x=le.transform(pd.DataFrame(df_stdscalar[i]))
    df_stdscalar[i]=x
# StandardScaling Numerical Values Description
df_stdscalar[numerical_columns].describe()