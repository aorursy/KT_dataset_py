# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
show_widget = False
from copy import deepcopy
from clustergrammer2 import net as net_ini
from copy import deepcopy
net = deepcopy(net_ini)
df = {}
if show_widget == False:
    print('\n-----------------------------------------------------')
    print('>>>                                               <<<')    
    print('>>> Please set show_widget to True to see widgets <<<')
    print('>>>                                               <<<')    
    print('-----------------------------------------------------\n')    
    delattr(net, 'widget_class') 
df['ini'] = net.load_gene_exp_to_df('../input/pbmc3k-filtered-gene-bc-matrices/pbmc3k_filtered_gene_bc_matrices/pbmc3k_filtered_gene_bc_matrices/hg19/')
df['ini'].shape
all_genes = df['ini'].index.tolist()
print(len(all_genes))
keep_genes = [x for x in all_genes if 'RPL' not in x]
keep_genes = [x for x in keep_genes if 'RPS' not in x]
print(len(keep_genes))

df['filt'] = df['ini'].loc[keep_genes]
print(df['filt'].shape)

# Removing Mitochondrial Genes
list_mito_genes = ['MTRNR2L11', 'MTRF1', 'MTRNR2L12', 'MTRNR2L13', 'MTRF1L', 'MTRNR2L6', 'MTRNR2L7',
                'MTRNR2L10', 'MTRNR2L8', 'MTRNR2L5', 'MTRNR2L1', 'MTRNR2L3', 'MTRNR2L4']


all_genes = df['filt'].index.tolist()
mito_genes = [x for x in all_genes if 'MT-' == x[:3] or 
             x.split('_')[0] in list_mito_genes]
print(mito_genes)

keep_genes = [x for x in all_genes if x not in mito_genes]
df['filt'] = df['filt'].loc[keep_genes]
df['filt'].shape
df['ash'] = np.arcsinh(df['filt']/5)
df['ash-umi'] = net.umi_norm(df['ash'])

num_keep_sum = 10000
ser_mean = df['ash-umi'].sum(axis=1)
keep_sum = ser_mean.sort_values(ascending=False).index.tolist()[:num_keep_sum]
df['ash-umi'] = df['ash-umi'].loc[keep_sum]
df['ash-umi'].shape
net.load_df(df['ash-umi'])
net.normalize(axis='row', norm_type='zscore')
df['ash-umi-z'] = net.export_df()
net.load_df(df['ash-umi'])
net.filter_N_top(inst_rc='row', N_top=250, rank_type='var')
net.normalize(axis='row', norm_type='zscore')
net.clip(lower=-5, upper=5)
net.widget()
net.load_file('../input/cibersort-signatures/cell_type_signatures/cell_type_signatures/nm3337_narrow_cell_type_sigs.txt')
net.normalize(axis='row', norm_type='zscore')
df_sig = net.export_df()
print(df_sig.shape)

rows = df_sig.index.tolist()
new_rows = [x.split('_')[0] for x in rows]
df_sig.index = new_rows
ct_color = {}
ct_color['T cells CD8'] = 'red'
ct_color['T cells CD4 naive'] = 'blue'
ct_color['T cells CD4 memory activated'] = 'blue'
ct_color['T cells CD4 memory resting'] = '#87cefa' # sky blue
ct_color['B cells naive'] = 'purple'
ct_color['B cells memory'] = '#DA70D6' # orchid
ct_color['NK cells activated'] = 'yellow'
ct_color['NK cells resting'] = '#FCD116' # sign yellow
ct_color['Monocytes'] = '#98ff98' # mint green
ct_color['Macrophages M0'] = '#D3D3D3' # light grey
ct_color['Macrophages M1'] = '#C0C0C0' # silver
ct_color['Macrophages M2'] = '#A9A9A9' # dark grey
ct_color['N.A.'] = 'white'

net.set_cat_colors(ct_color, 'col', 1)
gene_sig = df_sig.idxmax(axis=1)
gs_dict = {}
for inst_gene in gene_sig.index.tolist():
    gs_dict[inst_gene] = gene_sig[inst_gene][0]
df_sig_cat = deepcopy(df_sig)
rows = df_sig_cat.index.tolist()
new_rows = [(x, 'Cell Type: ' + gs_dict[x]) if x in gs_dict else (x, 'N.A.') for x in rows ]
df_sig_cat.index = new_rows

net.load_df(df_sig_cat)
net.set_cat_colors(ct_color, 'row', 1, 'Cell Type')
net.load_df(df_sig_cat)
net.clip(lower=-5, upper=5)
net.widget()
df_pred_cat, df_sig_sim, y_info = net.predict_cats_from_sigs(df['ash-umi-z'], df_sig, 
                                                                   predict_level='Cell Type', unknown_thresh=0.05)
df['ash-umi-cat'] = deepcopy(df['ash-umi'])
df['ash-umi-cat'].columns = df_pred_cat.columns.tolist()
print(df_pred_cat.shape)
df_sig_sim = df_sig_sim.round(2)
net.load_df(df_sig_sim)
net.set_cat_colors(ct_color, 'col', 1, cat_title='Cell Type')
net.set_cat_colors(ct_color, 'row', 1)
df_sig_sim.columns = df_pred_cat.columns.tolist()
net.load_df(df_sig_sim)
net.widget()
rows = df_pred_cat.index.tolist()
new_rows = [(x, 'Cell Type: ' + gs_dict[x]) if x in gs_dict else (x, 'N.A.') for x in rows ]
df_pred_cat.index = new_rows
net.load_df(df_pred_cat)
net.clip(lower=-5, upper=5)
net.widget()
rows = df['ash-umi-cat'].index.tolist()
new_rows = [(x, 'Cell Type: ' + gs_dict[x]) if x in gs_dict else (x, 'N.A.') for x in rows]
df['ash-umi-cat'].index = new_rows
net.load_df(df['ash-umi-cat'])
net.filter_N_top(inst_rc='row', N_top=250, rank_type='var')
net.normalize(axis='row', norm_type='zscore')
net.clip(lower=-5, upper=5)
net.widget()
df['ash-umi-cat'].to_csv('pbmc_2700_cell_types.txt', sep='\t')
# Somethihg