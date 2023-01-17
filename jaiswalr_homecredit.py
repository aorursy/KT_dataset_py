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
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity="all"
import matplotlib.pyplot as plt
import datetime as dt
from datetime import date, timedelta
df=pd.read_csv("/kaggle/input/home-credit/application_train.csv")
def MemoryManagement(df):
    '''This function saves memory by converting & allocating datatype required for each column data.'''
    print("Shape of data set", df.shape)
    Initial_Memory= df.memory_usage().sum() /(1024*1024)
    print ("Intial memory of the dataset in MB",df.memory_usage().sum() /(1024*1024))

    for i in df.describe().columns:
        if str(df[i].dtypes)[0:3]=='int':
            if df[i].max()<np.iinfo('int8').max:
                df[i]= df[i].astype('int8') ### one form of datatype to other
            elif df[i].max()<np.iinfo('int16').max:
                df[i]= df[i].astype('int16')
            elif df[i].max()<np.iinfo('int32').max:
                df[i]= df[i].astype('int32')
        else:
            if df[i].max()<np.finfo('float16').max:
                df[i]= df[i].astype('float16') ### one form of datatype to other
            elif df[i].max()<np.finfo('float32').max:
                df[i]= df[i].astype('float32')
            elif df[i].max()<np.finfo('float64').max:
                df[i]= df[i].astype('float64')

    print ("After processing memory of the dataset in MB",df.memory_usage().sum() /(1024*1024))
    new_memory =df.memory_usage().sum() /(1024*1024)
    print("Memory Saved: ",((Initial_Memory-new_memory)/Initial_Memory)*100)

    
def GarbageValueRemoval(df):
    '''This function to remove garbage value from dataset.'''
    for i in df.columns:
        df[i].replace(regex=True, to_replace=r'[^0-9a-zA-Z.\-]',value=r'', inplace=True) #remove special Char
        df[i].replace(regex=True,to_replace=r'^\s*$', value=np.nan, inplace=True)        #replace empty with NAN

def ColumnTypeIdentification(df,threshold=33):
    '''This function to segregate numeric & char columns and furher segregate into continuous and descrete based upon given theshold.'''
    import pandas as pd
    #Numerical columns
    numerical_columns=df.describe().columns
    print('Numerical Columns:\n', numerical_columns)
    
    # Charecter Columns
    char_columns=df.describe(include='object').columns
    print('Charecter Columns:\n', char_columns)
    
    print('Length of numerical columns: ', len(numerical_columns))
    print('Length of charecter columns: ', len(char_columns))
    
    # Making all the values as pandas table
    table_information_numerical=[]
    for i in df[numerical_columns]:
        table_information_numerical.append([i,df[i].nunique()])
    table_information_char=[]
    for i in df[char_columns]:
        table_information_char.append([i,df[i].nunique()])
    
    table_information_numerical=pd.DataFrame(table_information_numerical)
    table_information_char=pd.DataFrame(table_information_char)

    
    # Sperarating numerical continuous columns
    numerical_cont=table_information_numerical[table_information_numerical[1]>threshold][0].values
    print('Total numerical continuous columns: ', len(numerical_cont))
    print(numerical_cont)
    
    # Separating numerical class columns
    numerical_class=table_information_numerical[table_information_numerical[1]<=threshold][0].values
    print('Total numerical class columns: ', len(numerical_class))
    print(numerical_class)
    
    # Sperarating categorical continuous columns
    categorical_cont=table_information_char[table_information_char[1]>threshold][0].values
    print('Total categorical continuous columns: ', len(categorical_cont))
    print(categorical_cont)
    
    # Separating categorical descrete columns
    categorical_class=table_information_char[table_information_char[1]<=threshold][0].values
    print('Total categorical class columns: ', len(categorical_class))
    print(categorical_class)
    
    return numerical_cont, numerical_class, categorical_cont, categorical_class



def NullValueTreatment(df,threshold=30):
    '''This function is to revmoe or null values from data set'''
    print("Null columns before treatment: ", df.isna().sum().sum())
    
    #Count the null value for each column
    null_value_cnt=df.isna().sum()
    
    #Seperate the non null value columns and null value columns
    null_columns=null_value_cnt[null_value_cnt>0]
    non_null_columns=null_value_cnt[null_value_cnt==0]
    
    null_value_percent=(null_columns/df.shape[0])*100
    
    retain_null_col=null_value_percent[null_value_percent<threshold].index
    drop_null_col=null_value_percent[null_value_percent>=threshold].index

    print('Retainable Columns',retain_null_col.shape[0])
    print('Dropable Columns',drop_null_col.shape[0])
    
    df.drop(drop_null_col,axis=1,inplace=True)
    
    # Seperate the Discrete and Continuous columns within numeric and char
    numerical_cont, numerical_class, categorical_cont, categorical_class = ColumnTypeIdentification(df,33)
    
    for i in categorical_class:
        df[i].fillna(df[i].mode().values[0],inplace=True)
    for i in numerical_cont:
        df[i].fillna(df[i].median(),inplace=True)
    for i in numerical_class:
        df[i].fillna(df[i].mode().values[0],inplace=True)
    for i in categorical_cont:
        df[i].fillna(df[i].mode().values[0],inplace=True)
        
    print("Null columns after treatment: ", df.isna().sum().sum())


def OutliersTreatment(df):
    '''This function is to remove outliers from the data'''
    for i in df.describe().columns:
        Q1=df.describe().at['25%',i]
        Q3=df.describe().at['75%',i]
        IQR=Q3 - Q1
        LTV=Q1 - 1.5 * IQR
        UTV=Q3 + 1.5 * IQR
        x=np.array(df[i])
        p=[]
        for j in x:
            if j < LTV or j>UTV:
                p.append(df[i].median())
            else:
                p.append(j)
        df[i]=p
    

def StandardScaling(df):
    '''This function is to apply Standard Scalar on the given data and return scaled dataset'''
    from sklearn.preprocessing import StandardScaler
    for i in df.columns:
        le=StandardScaler()
        le.fit(df[i].values.reshape(-1, 1))
        x=le.transform(df[i].values.reshape(-1, 1))
        df[i]=x
    return df


# In[50]:


def MinMaxScaling(df):
    '''This function is to apply Standard Scalar on the given data and return scaled dataset'''
    from sklearn.preprocessing import MinMaxScaler
    for i in df.columns:
        le=MinMaxScaler()
        le.fit(df[i].values.reshape(-1, 1))
        x=le.transform(df[i].values.reshape(-1, 1))
        df[i]=x
    return df

def UnivariateVisual(df,cont):
    '''This function is to plot histogram for contenuous column'''
    import seaborn as sns
    import matplotlib.pyplot as plt
    for i in cont:
        plt.figure(figsize=(20,5))
        sns.distplot(df[i],kde=False)
        #plt.savefig(i+'plot.png') 
    return plt





def ContenuousBivariateVisual(df,cont):
    '''This function is plot scatter plot between two contenuous columns'''
    import seaborn as sns
    import matplotlib.pyplot as plt
    for i in cont:
        for j in cont:
            plt.figure(figsize=(20,5))
            sns.scatterplot(df[i], df[j])
            plt.xlabel(i)
            plt.ylabel(j)
    return plt
       




def ContDescBivariateVisual(df,cont,desc):
    '''This function is plot box plot between one and one descrete contenuous columns'''
    import seaborn as sns
    import matplotlib.pyplot as plt
    for i in cont:
        for j in desc:
            plt.figure(figsize=(20,5))
            sns.boxplot(df[i], df[j])
            plt.xlabel(i)
            plt.ylabel(j)
    return plt
       

df.head()
MemoryManagement(df)
df.head()
numerical_cont, numerical_class, categorical_cont, categorical_class = ColumnTypeIdentification(df)
numerical_cont
numerical_class
categorical_cont
categorical_class
df['ORGANIZATION_TYPE'].unique()


UnivariateVisual(df,numerical_cont)
d