# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm_notebook as tqdm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df_test = pd.read_csv('../input/test.csv')
df_test.drop(['ID_code'], axis=1, inplace=True)
df_test.head()
df_test = df_test.values #convert to numpy array
unique_samples = []
unique_count = np.zeros_like(df_test) #create a same shape (200000,200) of zero np
unique_count
for feature in tqdm(range(df_test.shape[1])):
    # loop thru each column, find which rows has value that never repeat in the same column and +1 in unique_count
    
    # return the indexes and counts on each unique value found in the column 
    _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)
    
    ## uncomment to see what np.unique return.
    #print (index_[:20])
    #print (count_[:20])
    #break
    
    ## [ 14030 119481 194633 103891  23396 141831 111894  64543  32828 186765  188883 109095  17040 112895   2680 152511  14952  50619   7246  71156]
    ## [2 2 1 1 2 1 1 1 2 1 1 2 4 1 3 1 1 2 2 2]
    
    # meaning at index 14030 , the value 0.1887 repeated twice. (another 1 is index 19561)
    
    
    ### the following line  only +1 those row with 1 occurrence indexes
    unique_count[index_[count_ == 1], feature] += 1
unique_count.shape
unique_count[0:100,0:10]
# Samples which have unique values are real the others are fake
real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]
real_samples_indexes
synthetic_samples_indexes

# df_test_real = df_test[real_samples_indexes].copy()

# generator_for_each_synthetic_sample = []
# # Using 20,000 samples should be enough. 
# # You can use all of the 100,000 and get the same results (but 5 times slower)
# for cur_sample_index in tqdm(synthetic_samples_indexes[:20000]):
#     cur_synthetic_sample = df_test[cur_sample_index]
#     potential_generators = df_test_real == cur_synthetic_sample
    
#     #print (cur_synthetic_sample)
#     #print (potential_generators)
    
#     # A verified generator for a synthetic sample is achieved
#     # only if the value of a feature appears only once in the
#     # entire real samples set
#     features_mask = np.sum(potential_generators, axis=0) == 1
#     verified_generators_mask = np.any(potential_generators[:, features_mask], axis=1)
#     verified_generators_for_sample = real_samples_indexes[np.argwhere(verified_generators_mask)[:, 0]]
#     generator_for_each_synthetic_sample.append(set(verified_generators_for_sample))
# public_LB = generator_for_each_synthetic_sample[0]
# for x in tqdm(generator_for_each_synthetic_sample):
#     if public_LB.intersection(x):
#         public_LB = public_LB.union(x)

# private_LB = generator_for_each_synthetic_sample[1]
# for x in tqdm(generator_for_each_synthetic_sample):
#     if private_LB.intersection(x):
#         private_LB = private_LB.union(x)
        
# print(len(public_LB))
# print(len(private_LB))