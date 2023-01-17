import numpy as np 

import pandas as pd

import time





bld_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv', index_col = 'building_id')

bld_df.head()
t_point_1 = time.time()



usage_columns = {i:(s[:3], s) for i,s in enumerate(set(bld_df['primary_use']))}

# {0: ('Off', 'Office'), 1: ('War', 'Warehouse/storage')...



usage_columns_dict = {v[1]:i for i, v in usage_columns.items()}

# {'Office': 0, 'Warehouse/storage': 1, 'Religious worship':2,...



shape_usage = (bld_df.shape[0],len(set(bld_df['primary_use'])))

# = (length of bld_df, num of categories) = (1449, 16)



usage_matrix = np.zeros(shape_usage, dtype = np.int8)



j = range(bld_df.shape[0])

i = list(bld_df['primary_use'].map(usage_columns_dict))

usage_matrix[j, i] = 1



primari_use_df = pd.DataFrame(usage_matrix, index = bld_df.index, columns = [v[0] for k,v in usage_columns.items()])





print(f'     -----> done: {time.time()-t_point_1:0.3f} s')



bld_df.merge(primari_use_df, left_index = True, right_index = True)