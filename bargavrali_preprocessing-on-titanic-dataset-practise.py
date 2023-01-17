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
# Hi, I am practising Data Science Concepts and the below Program gives the basic idea on
# the Preprocessing Steps involved during the Machine Learning. It works on Titanic Dataset
# and finally shows the Numerical data copies Scaled with MinMaxScaling and StandardScaling.
#
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sb
df_titanic = sb.load_dataset('titanic')
df_titanic.head()
#
#-------------------------SOME OBSERVATIONS ON DATA REDUNDANCIES------------------------------
# 1.survived ---> alive
# 2.pclass ---> class
# 3.embark ---> emark_town
# ---> 1,2 and 3 are the columns with exact duplications of data
#---------------------------------------------------------------------------------------------
# sex, age, who, adult_male are closely related
# sibsp, parch,alone are also related as the person will be alone only if sibsp and parch = 0
#---------------------------------------------------------------------------------------------
df_titanic[df_titanic['alone'] == True]
df_titanic['survived'].value_counts()
df_titanic['alive'].value_counts()
df_titanic['pclass'].value_counts()
df_titanic['class'].value_counts()
df_titanic['sex'].value_counts()
df_titanic['who'].value_counts()
df_titanic['adult_male'].value_counts()
df_titanic['embarked'].value_counts()
df_titanic['embark_town'].value_counts()
df_titanic.isna().sum()
null_value_treatment=(df_titanic.isna().sum()/df_titanic.shape[0])*100
null_value_treatment
null_value_threshold=30
retained_columns=null_value_treatment[null_value_treatment<null_value_threshold].index
retained_columns
discarded_columns=null_value_treatment[null_value_treatment>null_value_threshold].index
discarded_columns
retained_columns
numerical_columns=df_titanic[retained_columns].describe().columns
numerical_columns
categorical_columns=[]
for i in retained_columns:
    if i not in numerical_columns:
        categorical_columns.append(i)
categorical_columns, len(categorical_columns)
len(numerical_columns),len(categorical_columns),len(retained_columns)
numerical_cont=[]
numerical_class=[]
for i in numerical_columns:
    if df_titanic[i].nunique()>8:
        numerical_cont.append(i)
    else:
        numerical_class.append(i)
cat_moreuniques=[]
cat_class=[]
for i in categorical_columns:
    if df_titanic[i].nunique()>8:
        cat_moreuniques.append(i)
    else:
        cat_class.append(i)
len(numerical_cont),len(numerical_class),len(cat_moreuniques),len(cat_class)
        
for i in numerical_cont:
    df_titanic[i].fillna(df_titanic[i].median(),inplace=True)
for i in numerical_class:
    df_titanic[i].fillna(df_titanic[i].mode().values[0],inplace=True)
for i in cat_class:
    df_titanic[i].fillna(df_titanic[i].mode().values[0],inplace=True)
for i in cat_moreuniques:
    df_titanic[i].fillna(df_titanic[i].mode().values[0],inplace=True)
df_titanic[retained_columns].isna().sum()
outlier_treated_values=[]
for i in numerical_cont:
    x=df_titanic[i].values
    qrt_1=np.quantile(x,0.25)
    qrt_3=np.quantile(x,0.75)
    rtv=qrt_3 - qrt_1
    ltv= qrt_1 - 1.5*rtv
    utv=qrt_3 + 1.5*rtv
    outlier_treated_values=[]
    for j in x:
        if (j > utv) or (j < ltv):
            outlier_treated_values.append(np.median(x))
            print(i,"::",j,'replaced with ',np.median(x))
            print(i,"qrt_1-->",qrt_1,"qrt_3-->",qrt_3,"utv-->",utv,"ltv-->",ltv,"median is ",np.median(x))
            
        else:
            outlier_treated_values.append(j)
    df_titanic[i]=outlier_treated_values.copy()
            
df_titanic[numerical_cont].describe()
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
for i in categorical_columns:
    le=LabelEncoder()
    le.fit(df_titanic[i])
    x=le.transform(df_titanic[i])
    df_titanic[i]=x
df_minmaxscaling=df_titanic[numerical_columns]
for i in numerical_columns:
    le=MinMaxScaler()
    le.fit(pd.DataFrame(df_minmaxscaling[i]))
    x=le.transform(pd.DataFrame(df_minmaxscaling[i]))
    df_minmaxscaling[i]=x
df_stdscaling=df_titanic[numerical_columns]
for i in numerical_columns:
    le=StandardScaler()
    le.fit(pd.DataFrame(df_stdscaling[i]))
    x=le.transform(pd.DataFrame(df_stdscaling[i]))
    df_stdscaling[i]=x
df_minmaxscaling.describe()
df_stdscaling.describe()
