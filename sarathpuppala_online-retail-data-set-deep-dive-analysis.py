from IPython.display import Image

Image("/kaggle/input/online-retail-socgen/workflow.PNG")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import plotly.express as px

import pandas_profiling as pp

import plotly.graph_objs as go

import matplotlib.pyplot as plt

from plotly.offline import iplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
Data=pd.read_csv("/kaggle/input/online-retail-socgen/OnlineRetail.csv", encoding='iso-8859-1' )

print("Count of Rows, Columns: ",Data.shape)
report = pp.ProfileReport(Data)

report.to_file("report.html")



report
Data.isnull().sum()
Data['Date']=[item[0] for item in Data['InvoiceDate'].str.split()]

Data['Time']=[item[1] for item in Data['InvoiceDate'].str.split()]

Data['Month']=[item[1] for item in Data['Date'].str.split('-')]

Data['Year']=[item[2] for item in Data['Date'].str.split('-')]

Data['TotalCost']=Data['Quantity']*Data['UnitPrice']
Month={'1':'Jan' , '2':'Feb' , '3':'Mar', '4':'Apr' ,'5':'May' , '6':'Jun' ,

       '7':'Jul' , '8':'Aug' , '9':'Sep' , '10':'Oct', '11':'Nov' ,'12':'Dec',

       '01':'Jan' , '02':'Feb' , '03':'Mar', '04':'Apr' ,'05':'May' , '06':'Jun' ,

       '07':'Jul' , '08':'Aug' , '09':'Sep' }



Data=Data.replace({"Month": Month})

Data.head()
Data['Description']=Data.groupby(["Country","UnitPrice","Date"])['Description'].transform(lambda x: x.fillna(x.mode()))

Data['Description']=Data['Description'].transform(lambda x: x.fillna("Others"))
df_Non_Null=Data[Data['CustomerID'].isnull()==False].copy()

df_Null=Data[Data['CustomerID'].isnull()==True].copy()



print("df_Non_Null Shape : ",df_Non_Null.shape,"\ndf_Null Shape : ",df_Null.shape)
Data_Join=df_Non_Null.merge(df_Null,on='InvoiceNo',how='inner')

Data_Join.head()
df_Null.head()
print("min : ",min(df_Non_Null['CustomerID']))

print("max : ",max(df_Non_Null['CustomerID']))
df_Null['InvoiceNo'].nunique()
np.random.seed( 30 )



df_Null['CustomerID']=df_Null.groupby(["InvoiceNo"])['CustomerID'].transform(lambda x: x.fillna(np.random.randint(18288,21998,1)[0]))

df_Null.head()
frames = [df_Non_Null, df_Null]

 

df = pd.concat(frames)

df.isnull().sum()
df.head()
def reduce_mem_usage(df):

    start_mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in df.columns:

        if df[col].dtype != object:  # Exclude strings

            

            # make variables for Int, max and min

            IsInt = False

            mx = df[col].max()

            mn = df[col].min()

            

            #Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(df[col]).all(): 

               NAlist.append(col)

               df[col].fillna(-999,inplace=True)  

                   

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

                        df[col] =df[col].astype(np.int64)    

            

            # Make float datatypes 32 bit

            else:

                df[col] = df[col].astype(np.float32)

            

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    return df,NAlist
df,df_Na=reduce_mem_usage(Data)
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
df = reduce_mem_usage(Data)
print("Around ",np.around(len(df[df["Country"] == 'United Kingdom'])/len(df)*100),"% of the data is from United Kingdom")
print("Around ",np.around(sum(n < 0 for n in df.TotalCost.values.flatten())/len(df)*100),"% of data is with -ve values")
df[df["TotalCost"] < 0]
df_list = ['Data', 'df_Non_Null', 'df_Null', 'Data_join', 'frames', 'df', 'df_Na']

del df_list



import gc 

gc.collect()