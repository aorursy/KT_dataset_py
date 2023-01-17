import os

import gc

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm.auto import tqdm
def reduce_mem_usage(props):

    start_mem_usg = props.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in props.columns:

        if props[col].dtype != object:  # Exclude strings

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

    

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = props.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size\n")

    return props, NAlist
%%time



chunk_train = pd.read_csv('../input/train.csv', chunksize=1000000)



chunk_list = []



for chunk in tqdm(chunk_train):

    probs, NAlist = reduce_mem_usage(chunk)

    chunk_list.append(probs)

    del chunk, probs, NAlist



gc.collect()

train = pd.concat(chunk_list, axis=0)

train.head()
train.head()
train.shape
train.to_feather('./train.ftr')
%%time 



new_train = pd.read_feather('./train.ftr', use_threads=4)
new_train.head()