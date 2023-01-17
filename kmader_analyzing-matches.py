import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pydicom import read_file
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
from IPython.display import display
import networkx as nx
def create_graph_from_df(in_df, ):
    g = nx.Graph()
    for _, c_row in in_df.iterrows():
        g.add_edge(c_row['annotator_x'], c_row['annotator_y'], 
                   weight = c_row['iou'], 
                   label = '{:2.0f}%'.format(100*c_row['iou']))
    return g
def show_graph(g, figsize = (20, 20)):
    fig, ax1 = plt.subplots(1, 1, figsize = figsize)
    ax1.axis('off')
    edge_labels = nx.get_edge_attributes(g,'label')
    n_pos = nx.spring_layout(g)
    nx.draw_networkx(g, 
                     pos=n_pos, 
                     ax = ax1);
    nx.draw_networkx_edge_labels(g, 
                     pos=n_pos, 
                     edge_labels = edge_labels,
                     ax = ax1);
    return fig
base_csv_dir = '../input/annotator-quality-2d/'
base_dcm_dir = '../input/crowds-cure-cancer-2017/annotated_dicoms/'
annot_df = pd.read_csv(os.path.join(base_csv_dir, 'matching_results.csv'), index_col = 0)
# remove duplicates
single_annot_df = annot_df[annot_df['annotator_x']<annot_df['annotator_y']].copy()
print(single_annot_df.shape[0], annot_df.shape[0])
annot_df.sample(3)
grp_df = single_annot_df.groupby(['anatomy', 'seriesUID'])
for _, (c_group, c_df) in zip(range(1), grp_df):
    # get the first group
    print(c_group)
    display(c_df)
    pass
g = create_graph_from_df(c_df)
show_graph(g);
g = create_graph_from_df(single_annot_df.query('anatomy == "Liver"'))
show_graph(g, figsize = (40, 40));
g = create_graph_from_df(single_annot_df.query('anatomy == "Ovarian"'))
show_graph(g, figsize = (40, 40));
avg_annot_df = single_annot_df.groupby(['annotator_x', 'annotator_y']).agg({'iou': 'mean', 'seriesUID': 'count'}).reset_index()
avg_annot_df.sample(2)
g = create_graph_from_df(avg_annot_df)
show_graph(g, figsize = (40, 40));
worst_annot_df = annot_df.groupby('annotator_x').agg({'iou': 'max', 'seriesUID': 'count'}).reset_index().sort_values('iou').query('iou<1e-3')
worst_annot = worst_annot_df['annotator_x'].values
print(worst_annot.shape[0])
worst_annot_df
clean_annot_df = single_annot_df[~(single_annot_df['annotator_x'].isin(worst_annot) | (single_annot_df['annotator_x'].isin(worst_annot)))]
clean_full_annot_df = annot_df[~(annot_df['annotator_x'].isin(worst_annot) | (annot_df['annotator_x'].isin(worst_annot)))]
print(single_annot_df.shape[0], '->', clean_annot_df.shape[0])
clean_full_annot_df.groupby(
    ['anatomy', 'seriesUID']).agg(
    {'iou': 'mean', 'annotator_x': 'count'}).reset_index().sort_values(
    'annotator_x', ascending = False)['annotator_x'].hist()
filt_annot = lambda cut_off: clean_full_annot_df.query('iou>{}'.format(cut_off)).groupby(['anatomy', 'seriesUID']).agg({'iou': 'mean', 'annotator_x': 'count'}).reset_index().sort_values('annotator_x', ascending = False)
cut_off = np.linspace(15, 100, 10, dtype = int)[:-1]
out_df_list = []
for n_cut_off in cut_off:
    c_df = filt_annot(n_cut_off/100.0)
    c_df['cutoff'] = n_cut_off
    out_df_list += [c_df]
out_df = pd.concat(out_df_list)
import seaborn as sns
sns.factorplot(x = 'cutoff', 
               y = 'annotator_x', 
               hue = 'anatomy', 
               data = out_df,
               kind = 'swarm',
              size = 6)
sns.factorplot(x = 'cutoff', 
               y = 'annotator_x', 
               hue = 'anatomy', 
               data = out_df,
               kind = 'violin',
               col = 'anatomy',
               col_wrap = 2,
              size = 8)
all_df_out = []
for annot_per_case in range(2, 10):
    c_df = out_df.query('annotator_x>={}'.format(annot_per_case)).groupby(['cutoff','anatomy']).apply(lambda c_rows: pd.Series({'total_annotations': np.sum(c_rows['annotator_x']),
                                                                     'total_cases': np.unique(c_rows['seriesUID']).shape[0]
                                                                                   })).reset_index()
    c_df['annot_per_case_cutoff'] = annot_per_case
    all_df_out += [c_df]
all_df = pd.concat(all_df_out)
sns.factorplot(x = 'annot_per_case_cutoff', 
               y = 'total_cases', 
               hue = 'cutoff', 
               data = all_df,
               col = 'anatomy',
               col_wrap = 2,
               sharey = False,
              size = 8)
sns.factorplot(x = 'annot_per_case_cutoff', 
               y = 'total_cases', 
               hue = 'anatomy', 
               data = all_df,
               col = 'cutoff',
               col_wrap = 3,
               sharey = False,
              size = 8)
total_annotation_count = {k['anatomy']: (k['count'], k['cases'])
                                         for k in pd.read_csv('../input/crowds-cure-cancer-2017/CrowdsCureCancer2017Annotations.csv').groupby(
    'anatomy').apply(lambda x: pd.Series(
    {'count': x.shape[0], 'cases': np.unique(x['seriesUID']).shape[0]})).reset_index().T.to_dict().values()}
total_annotation_count
fig, m_axs = plt.subplots(2, 2, figsize = (20, 20))
for (c_organ, c_df), ax1 in zip(all_df.groupby('anatomy'), m_axs.flatten()):
    X, x_vals = c_df['annot_per_case_cutoff'].factorize()
    Y, y_vals = c_df['cutoff'].factorize()
    c_mat = np.zeros((X.max()+1, Y.max()+1), dtype = 'int')
    c_mat[X, Y] = c_df['total_cases'].values.astype(int)
    ax1 = sns.heatmap(c_mat.T, annot = True, fmt = 'd', ax = ax1)
    ax1.set_xticklabels(x_vals)
    ax1.set_yticklabels(y_vals)
    ax1.set_xlabel('Annotations Per Case')
    ax1.set_ylabel('Minimum Agreement (DICE %)')
    ax1.set_title('{0}: Patients: {2}, Annotations: {1}'.format(c_organ, *total_annotation_count[c_organ]))
