%reset
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # pip install seaborn

sns.set(style="whitegrid", color_codes=True)
pd.set_option('display.max_columns', None)
df = pd.read_csv('../input/train.csv', header=0,sep=',')
#different place_id
len(df['place_id'].unique())
df_cl = df.copy()
ax = sns.distplot(df_cl["time"], kde=False)
ax = sns.distplot(df_cl["accuracy"], kde=False)
aggregation = {
    'x' : {
        'median_x' : 'median',
        'mean_x' : 'mean',
        'variance_x' : 'var' 
    },
    'y' : {
        'median_y' : 'median',
        'mean_y' : 'mean',
        'var_y' : 'var'
    }
}
df_cl_agg = df_cl.groupby("place_id").agg(aggregation)
df_cl_agg.columns = df_cl_agg.columns.droplevel()
df_cl_agg.head()
result = pd.merge(df_cl, df_cl_agg, left_on='place_id', right_index=True, how='inner', suffixes=('_left', '_right'))
result.head()
result["distance_median"] = np.sqrt( np.power((result["median_y"]-result["y"]), 2) + np.power((result["median_x"]-result["x"]), 2) )
result.head()
