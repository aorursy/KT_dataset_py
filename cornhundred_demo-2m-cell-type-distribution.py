from clustergrammer2 import net

from sc_data_lake import SC_Data_Lake

# base_dir = '../data/primary_data/big_data/cao_2million-cell_2019_parquet_files/'

base_dir = '../input/cao-2millioncell-2019-parquet-files/cao_2million-cell_2019_parquet_files/cao_2million-cell_2019_parquet_files/'

lake = SC_Data_Lake(base_dir)

import warnings

warnings.filterwarnings('ignore')
df_dist = lake.get_cat_distribution_across_samples('Main_cell_type')

print('total cells', df_dist.sum(axis=1).sum())
df_dist.sum(axis=1).sort_values(ascending=False)
net.load_df(df_dist)

net.swap_nan_for_zero()

net.widget()