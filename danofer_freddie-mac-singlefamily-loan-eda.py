

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from pandas.api.types import is_numeric_dtype

    
def reduce_mem_usage(props):

    start_mem_usg = props.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in props.columns:

        if is_numeric_dtype(df[c]):

#             if ((props[col].dtype != object) & (props[col].dtype !=np.datetime64)):  # Exclude strings

            if ((props[col].dtype != object) & (props[col].dtype !=np.datetime64)):  # Exclude strings



                # Print current column type

                print("******************************")

                print("Column: ",col)

                print("dtype before: ",props[col].dtype)



                # make variables for Int, max and min

                IsInt = False

                mx = props[col].max()

                mn = props[col].min()



                # Integer does not support NA, therefore, NA needs to be filled

                if not np.isfinite(props[col]).all(): 

                    NAlist.append(col)

                    props[col].fillna(mn-1,inplace=True)  



                # test if column can be converted to an integer

                asint = props[col].fillna(0).astype(np.int64)

                result = (props[col] - asint)

                result = result.sum()

                if result > -0.01 and result < 0.01:

                    IsInt = True





                # Make Integer/unsigned Integer datatypes

                if IsInt:

                    if mn >= 0:

                        if mx < 255:

                            props[col] = props[col].astype(np.uint8)

                        elif mx < 65535:

                            props[col] = props[col].astype(np.uint16)

                        elif mx < 4294967295:

                            props[col] = props[col].astype(np.uint32)

                        else:

                            props[col] = props[col].astype(np.uint64)

                    else:

                        if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                            props[col] = props[col].astype(np.int8)

                        elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                            props[col] = props[col].astype(np.int16)

                        elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                            props[col] = props[col].astype(np.int32)

                        elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                            props[col] = props[col].astype(np.int64)    



                # Make float datatypes 32 bit

                else:

                    props[col] = props[col].astype(np.float32)



                # Print new column type

                print("dtype after: ",props[col].dtype)

                print("******************************")

                

                return df

df = pd.read_csv("/kaggle/input/freddie-mac-singlefamily-loanlevel-dataset/loan_level_500k.csv").dropna(subset=["POSTAL_CODE","CREDIT_SCORE"])

df["DELINQUENT"] = df["DELINQUENT"].astype(int) #instead of bool



print(df.shape)

df.head()
df["FIRST_PAYMENT_DATE"] = pd.to_datetime(df["FIRST_PAYMENT_DATE"],format="%Y%m")

df["MATURITY_DATE"] = pd.to_datetime(df["MATURITY_DATE"],format="%Y%m")
df = reduce_mem_usage(df) # ewrrors
df.dtypes
# from pandas.api.types import is_numeric_dtype



# for c in df.columns:

#     if is_numeric_dtype(df[c]):

#         df[c] = pd.to_numeric(df[c])

    

# df.dtypes
100*df.isna().mean() # there were very few missing values/nans for zipcodes, we dropped them
df.describe()
df["DELINQUENT"].describe()
df.sample(frac=1).to_csv("freddie-mac-singlefamily-loans_500k.csv.gz",index=False,compression="gzip")