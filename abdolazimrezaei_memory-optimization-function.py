import pandas as pd
import os
import re
import numpy as np


df_sales_train_validation = pd.read_csv(r'../input/m5-forecasting-accuracy/sales_train_validation.csv')

df_sales_train_validation.head(3)
cols = df_sales_train_validation.filter(regex='d_').columns
max = 0
for c in cols:
    if df_sales_train_validation[c].max() > max:
        max = df_sales_train_validation[c].max()
print('The maximum value for columns d_1 to d_1913 is: ', max)        
def ReduceSize(df_,  fl = 1):
    intValues = ['int_', 'intc', 'intp', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']
    floatValues = ['float_', 'float16', 'float32', 'float64']
    minn, maxx = 0, 0 
    stype = ''
    for c in df_.columns:
        try:
            if df_[c].dtypes == 'object':
                df_[c] = df_[c].astype('int64')
                print('Successful conversion Object to Integer for COlumn: ', c)
        except:
            print('Not Possible Casting Object to INT64 for column: ', c)
        stype = df_[c].dtypes
        #Cast to INT
        if stype in intValues:
            minn , maxx = 1, -1
            maxx = df_[c].max()
            minn = df_[c].min()
            if (minn >= -128) &  (maxx <= 128):
                df_[c] = df_[c].astype('int8')                   
            else:
                if (minn >= -32767) &  (maxx <= 32767):
                    df_[c] = df_[c].astype('int16')
                else:
                    if (minn >= -2147483647) &  (maxx <= 2147483647):
                        df_[c] = df_[c].astype('int32')
                    else:
                        df_[c] = df_[c].astype('int64')
            #Cast to UINT
            if (fl == 2):
                if (minn >= 0) &  (maxx <= 255):
                    df_[c] = df_[c].astype('uint8')                   
                else:
                    if (minn >= 0) &  (maxx <= 65535):
                        df_[c] = df_[c].astype('uint16')                   
                    else:
                        if (minn >= 0) &  (maxx <= 4294967295):
                            df_[c] = df_[c].astype('uint32')                   
                        else:
                            if (minn >= 0) &  (maxx <= 18446744073709551615):
                                df_[c] = df_[c].astype('uint64')                   
        
        if stype in floatValues:
            try:
                df_[c] = df_[c].astype('float16')
            except:
                try:
                    df_[c] = df_[c].astype('float32')
                except:
                    df_[c] = df_[c].astype('float64')            
        print(c)
    
    return df_
df_sales_train_validation.memory_usage(index=False)

np.sum(df_sales_train_validation.memory_usage(index=False))

ReduceSize(df_sales_train_validation)

np.sum(df_sales_train_validation.memory_usage(index=False))
