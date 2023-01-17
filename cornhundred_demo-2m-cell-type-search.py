from clustergrammer2 import net

from sc_data_lake import SC_Data_Lake

# base_dir = '../data/primary_data/big_data/cao_2million-cell_2019_parquet_files/'

base_dir = '../input/cao-2millioncell-2019-parquet-files/cao_2million-cell_2019_parquet_files/cao_2million-cell_2019_parquet_files/'

lake = SC_Data_Lake(base_dir)

import warnings

warnings.filterwarnings('ignore')

from dask.distributed import Client, progress

client = Client()

client
lake.reset_filters()

lake.add_filter('detected_doublet', False)

lake.add_filter('Main_cell_type', ['Megakaryocytes', 'Melanocytes'])
lake.add_cats(['Main_cell_type', 'Main_trajectory', 'development_stage'])
%%time

df_gex = lake.dask_run_cell_search()

print('\nCell Search Results')

print('------------------------')

print(df_gex.shape)
df_gex.shape
client.close()
net.load_df(df_gex)

if df_gex.shape[1] > 5000:

    net.random_sample(axis='col', num_samples=5000, random_state=99)

net.filter_N_top(inst_rc='row', N_top=100, rank_type='var')

net.normalize(axis='row', norm_type='zscore')

net.clip(-5,5)

net.load_df(net.export_df().round(2))

net.widget()
import umap

import matplotlib.pyplot as plt

%matplotlib inline 

import pandas as pd

def make_umap_plot(df, cat_index, colors_dict, title, min_dist=0.0, n_neighbors=5, s=0.5, alpha=0.5, 

                   figsize=(10,10)):

    cols = df.columns.tolist()

    cats = [x[cat_index] for x in cols]

    list_colors = [colors_dict[x.split(': ')[1]] for x in cats]

    

    

    

    embedding = umap.UMAP(n_neighbors=n_neighbors, random_state=99,

                          min_dist=min_dist,

                          metric='correlation').fit_transform(df.transpose())

    df_umap = pd.DataFrame(data=embedding, columns=['x', 'y'])

    df_umap.plot(kind='scatter', x='x', y='y',  c=list_colors, alpha=alpha, s=s, figsize=figsize, 

                 title=title)
ini_colors = {}

ini_colors['Cell Type'] = net.viz['cat_colors']['col']['cat-0']

ini_colors['Trajectory'] = net.viz['cat_colors']['col']['cat-1']

colors_dict = {}

for inst_type in ini_colors:

    inst_dict = ini_colors[inst_type]

    for inst_cat in inst_dict:

        new_key = inst_cat.split(': ')[1]

        new_color = inst_dict[inst_cat]

        colors_dict[new_key] = new_color
df_tmp = net.export_df()

df_tmp.shape
make_umap_plot(df_tmp, cat_index=1, colors_dict=colors_dict, title='Main Cell Type', s=2)
make_umap_plot(df_tmp, cat_index=2, colors_dict=colors_dict, title='Main Trajectory', s=2)
1+1