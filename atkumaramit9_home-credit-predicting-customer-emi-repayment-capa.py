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
#/kaggle/input/home-credit/application_train.csv
#/kaggle/input/home-credit/application_test.csv

### load required libraries

import pandas as pd
import numpy as np
import seaborn  as sns ###visuallization
### remove jupyter warnings
import warnings
warnings.filterwarnings('ignore')
##print multiple lines in jupyter notebook
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import matplotlib.pyplot as plt
import datetime as dt
from datetime import date, timedelta
import glob
df=pd.read_csv("/kaggle/input/application_train.csv")
df
files_location=pd.read_csv(r"/kaggle/input/application_train.csv")
files_location

### application train is inpt files for data set
application_train=files_location
## Rows
application_train.shape
## columns name list
application_train.columns
#Display first 10 rows

application_train.head(10)

## display last 10 rows
application_train.tail(10)

application_train

df = application_train.copy()
df
#Calculate age 
days_in_year=365.2425 
df['AGE']=(abs(df['DAYS_BIRTH']/days_in_year)).apply(lambda val: int(val))
# convert the days birth value to date
date_of_birth=[]
for i in df['DAYS_BIRTH']:
    date_of_birth.append(pd.to_datetime((date.today() + timedelta(i)).isoformat()))
df['DAYS_BIRTH']=date_of_birth
############################### Function to null treatment ##########################################

# '''This function is to perform null value treatment on given data using given threshold'''
def null_treated (df, threshold=30):
    import pandas as pd
    import numpy as np
    
    #Identifying percentage of null values
    null_value_treatment=(df.isna().sum()/df.shape[0])*100
    print(null_value_treatment)
    
    #Threshold for null treatment. If user doesn't pass any value, default is 30%
    null_values_threshold = threshold
    
    # Dropping columns based on threshold
    dropped_columns=null_value_treatment[null_value_treatment>null_values_threshold].index
    print('Dropped Columns:\n', dropped_columns)
    
    # Retained columns based on threashold
    retained_columns=null_value_treatment[null_value_treatment<null_values_threshold].index
    print('Retained Columns:\n', retained_columns)
    
    #Numerical columns
    numerical_columns=df[retained_columns].describe().columns
    print('Numerical Columns:\n', numerical_columns)
    
    # Charecter Columns
    char_columns=df[retained_columns].describe(include='object').columns
    print('Charecter Columns:\n', char_columns)
    
    print('Length of retained columns: ', len(retained_columns))
    print('Length of dropped columns: ', len(dropped_columns))
    print('Length of numerical columns: ', len(numerical_columns))
    print('Length of charecter columns: ', len(char_columns))
    
    # Tranforming in panadas tabular data
    
    import pandas as pd
    table_information_numerical=[]
    for i in df[numerical_columns]:
        table_information_numerical.append([i,df[i].nunique()])
    table_information_char=[]
    for i in df[char_columns]:
        table_information_char.append([i,df[i].nunique()])
    
    table_information_numerical=pd.DataFrame(table_information_numerical)
    table_information_char=pd.DataFrame(table_information_char)

    
    # Sperarating numerical continuous columns
    numerical_cont=table_information_numerical[table_information_numerical[1]>33][0].values
    print('Total numerical continuous columns: ', len(numerical_cont))
    print(numerical_cont)
    
    # Separating numerical class columns
    numerical_class=table_information_numerical[table_information_numerical[1]<=33][0].values
    print('Total numerical class columns: ', len(numerical_class))
    print(numerical_class)
    
    # Filling null values for retained columns
    df.drop(dropped_columns,axis=1,inplace=True)
    
    for i in char_columns:
        df[i].fillna(df[i].mode().values[0],inplace=True)
    for i in numerical_cont:
        df[i].fillna(df[i].median(),inplace=True)
    for i in numerical_class:
        df[i].fillna(df[i].mode().values[0],inplace=True)
        
    print("Null columns after treatment: ", df.isna().sum().sum())
    
    return df, numerical_cont, numerical_class, char_columns
df, numerical_cont, numerical_class, char_columns =null_treated(df)
numerical_cont
## Columns provided for analysis , merging it all to a new table train_table

train_table =pd.DataFrame()
train_table['TARGET']=application_train['TARGET']
train_table['FLAG_DOCUMENT_5'] = application_train['FLAG_DOCUMENT_5']
train_table['FLAG_DOCUMENT_4'] = application_train['FLAG_DOCUMENT_4']
train_table['AMT_REQ_CREDIT_BUREAU_QRT'] = application_train['AMT_REQ_CREDIT_BUREAU_QRT']
train_table['FLAG_DOCUMENT_16'] = application_train['FLAG_DOCUMENT_16']
train_table['FLAG_DOCUMENT_6'] = application_train['FLAG_DOCUMENT_6']
cols_to_analyse =pd.DataFrame( {'COLUMNS' : ['TARGET', 'AMT_REQ_CREDIT_BUREAU_QRT' ,'FLAG_DOCUMENT_5','FLAG_DOCUMENT_4','AGE',
 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_6']})
cols_to_analyse

variable_type= []

for i in cols_to_analyse['COLUMNS']:
    if i in numerical_cont:
        variable_type.append('NUM_CONT')
    elif i in numerical_class:
        variable_type.append('NUM_CLASS')
    else:
        variable_type.append('DISC_CHAR')
cols_to_analyse['COLUMN_TYPE']= variable_type
    
cols_to_analyse_cont=cols_to_analyse[cols_to_analyse['COLUMN_TYPE']=='NUM_CONT']
cols_to_analyse_class=cols_to_analyse[cols_to_analyse['COLUMN_TYPE']=='NUM_CLASS']
cols_to_analyse_char=cols_to_analyse[cols_to_analyse['COLUMN_TYPE']=='DISC_CHAR']
cols_to_analyse_cont
cols_to_analyse_class
cols_to_analyse_char
df[cols_to_analyse_class['COLUMNS']]
for i in cols_to_analyse_cont['COLUMNS']:
    plt.figure(figsize=(20,5))
    plt.hist(df[i])
    plt.ylabel("Range")
    plt.xlabel("AGE")
df['AGE'].describe()

new_df=pd.DataFrame(df['DAYS_BIRTH'].groupby(df['DAYS_BIRTH']).count()) 
new_df
birth_quarters=pd.DataFrame(new_df.DAYS_BIRTH.resample('Q').count())
birth_quarters
plt.figure(figsize=(20,5))
plt.plot(birth_quarters['DAYS_BIRTH'])
plt.xlabel('Quarter')
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.subplot(3,1,1)
sns.barplot(df['AMT_REQ_CREDIT_BUREAU_QRT'])
### Insight

##find the imp columns from cols_to_analyse and analyse using pairplot graph
sns.pairplot(df[['TARGET','AMT_REQ_CREDIT_BUREAU_QRT']])






















import matplotlib.pyplot as plt

