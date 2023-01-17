import os
data_dir = '../input'
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pydicom import read_file
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
base_dcm_dir = '../input/annotated_dicoms/'
ann_df = pd.read_csv(os.path.join(data_dir, 'CrowdsCureCancer2017Annotations.csv'))
ann_df.sample(2).T
ann_df.groupby(['annotator', 'radiologist_status']).agg(dict(order = 'count')
                                                       ).reset_index().sort_values('order', ascending = False).head(10)
ann_3d_df = ann_df.copy()
ann_3d_df['x0'] = ann_3d_df.apply(lambda x: min(x['start_x'], x['end_x']), 1)
ann_3d_df['x1'] = ann_3d_df.apply(lambda x: max(x['start_x'], x['end_x']), 1)
ann_3d_df['y0'] = ann_3d_df.apply(lambda x: min(x['start_y'], x['end_y']), 1)
ann_3d_df['y1'] = ann_3d_df.apply(lambda x: max(x['start_y'], x['end_y']), 1)
ann_3d_df.drop(['start_x', 'start_y', 'end_x', 'end_y'], axis = 1, inplace = True)
ann_3d_df['bbox'] = ann_3d_df.apply(lambda x: tuple(int(y) for y in [x['x0'], x['x1'], x['y0'], x['y1'], x['sliceIndex']]), 1)
ann_3d_df.sample(2)
annotator_status = {k['annotator']: k['radiologist_status'] for k in 
 ann_3d_df[['annotator', 'radiologist_status']].drop_duplicates().T.to_dict().values()} # keep track of the annotators status
# build base grid for all images
step_size = .5
base_x = np.arange(ann_3d_df['x0'].min(), ann_3d_df['x1'].max(), step_size)
base_y = np.arange(ann_3d_df['y0'].min(), ann_3d_df['y1'].max(), step_size)
xx, yy = np.meshgrid(base_x, base_y, indexing = 'ij')
print('x', ann_3d_df['x0'].min(), ann_3d_df['x1'].max())
print('y', ann_3d_df['y0'].min(), ann_3d_df['y1'].max())
bb_func = lambda bcube: (xx>=bcube[0]) & (xx<=bcube[1]) & (yy>=bcube[2]) & (yy<=bcube[3])
def calc_iou(cube_a, cube_b):
    bcube_a = bb_func(cube_a)
    bcube_b = bb_func(cube_b)
    iou_val = np.sum(bcube_a & bcube_b)/(1+np.sum(bcube_a+bcube_b))
    return (0.75**np.abs(cube_a[4]-cube_b[4]))*iou_val
print(calc_iou((0, 100, 0, 100, 100),
        (0, 100, 0, 100, 100))) # should be 1 since they overlap 100%
print(calc_iou((100, 200, 100, 200, 100),
        (100, 200, 150, 200, 100))) # should be 0.5 since they overlap 50%
print(calc_iou((0, 100, 0, 100, 100),
        (0, 100, 0, 100, 105))) # should be 0.24 since they overlap 100 but are off by 5 slices
_, j_row = next(ann_3d_df.iterrows())
_, k_row = next(ann_3d_df.sample(1).iterrows())
print('j->j IOU:', calc_iou(j_row['bbox'], j_row['bbox']))
print('j->k IOU:', calc_iou(j_row['bbox'], k_row['bbox']))
%%time
test_series_id = ann_3d_df.groupby('seriesUID').agg({'order':'count'}).reset_index().sort_values('order', ascending = False).head(1)['seriesUID'].values
cur_ann_df = list(ann_3d_df[ann_3d_df['seriesUID'].isin(test_series_id)].iterrows())
conf_mat = np.eye(len(cur_ann_df))
for i, (_, c_row) in enumerate(cur_ann_df):
    for j, (_, d_row) in enumerate(cur_ann_df[i+1:], i+1):
        c_iou = calc_iou(c_row['bbox'], d_row['bbox'])
        conf_mat[i,j] = c_iou
        conf_mat[j,i] = c_iou
sns.heatmap(conf_mat, 
            annot = True, 
            fmt = '2.2f')
simple_lesions = ann_3d_df[['anatomy', 'seriesUID', 'annotator', 'bbox']]
lesion_product_df = pd.merge(simple_lesions, simple_lesions, on = ['anatomy', 'seriesUID'])
lesion_product_df['is_first'] =  lesion_product_df.apply(lambda x: x['annotator_x']<x['annotator_y'], 1)
print(lesion_product_df.shape[0])
lesion_product_df = lesion_product_df[lesion_product_df['is_first']].drop(['is_first'],1)
lesion_product_df.sample(2)
%%time
lesion_product_df['iou'] = lesion_product_df.apply(lambda c_row: calc_iou(c_row['bbox_x'], c_row['bbox_y']), 1)
# add back the duplicated rows by swapping annotator_x and annotator_y
out_iou_df = lesion_product_df[['anatomy', 'seriesUID', 'annotator_x', 'annotator_y', 'iou']].fillna(0.0).copy()
out_iou_swapped = out_iou_df.copy()
out_iou_swapped['annotator_x'] = out_iou_df['annotator_y']
out_iou_swapped['annotator_y'] = out_iou_df['annotator_x']
out_iou_df = pd.concat([out_iou_df, out_iou_swapped]).sort_values(['seriesUID', 'annotator_x'])

out_iou_df.to_csv('matching_results.csv')
out_iou_df.sample(2)
from sklearn.preprocessing import LabelEncoder

ix_df = out_iou_df[['iou']].copy()
ix_df['ix'] = out_iou_df['annotator_x'].map(lambda x: '{}:{}'.format(annotator_status[x][0].upper(), x))
ix_df['iy'] = out_iou_df['annotator_y'].map(lambda x: '{}:{}'.format(annotator_status[x][0].upper(), x))

all_labe = LabelEncoder()
all_labe.fit(ix_df['ix'])
ix_df['ix'] = all_labe.transform(ix_df['ix']).astype(int)
ix_df['iy'] = all_labe.transform(ix_df['iy']).astype(int)
ix_df.sample(3)
sns.heatmap(ix_df.pivot_table(values = 'iou', columns = 'iy', index = 'ix', aggfunc = 'mean').values)
dist_map = 1/(ix_df.pivot_table(values = 'iou', columns = 'iy', index = 'ix', aggfunc = 'mean', fill_value=0).values+1e-3)
from scipy.cluster.hierarchy import dendrogram, linkage
plt.figure(figsize=(30, 14))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('annotator name')
plt.ylabel('distance')
dendrogram(
    linkage(dist_map),
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=10.,  # font size for the x axis labels
    leaf_label_func = lambda idx: all_labe.classes_[idx]
)
plt.show()
def pct_75(x): 
    return np.percentile(x, 75)
iou_results_df = out_iou_df.groupby(['anatomy', 'seriesUID', 'annotator_x']).agg({'iou': ['count', 'mean', 'max', pct_75, 'median']}).reset_index()
iou_results_df.columns = ['_'.join([y for y in col if len(y)>0]).strip() for col in iou_results_df.columns.values]
iou_results_df.sample(5)
iou_results_df['radiologist_status'] = iou_results_df['annotator_x'].map(annotator_status.get)
sns.pairplot(iou_results_df.drop('iou_count',1), 
             kind = 'reg', diag_kind="kde",
             hue = 'radiologist_status', size = 5)
ax = sns.swarmplot(x = 'radiologist_status', 
              y = 'iou_pct_75', 
              hue = 'annotator_x', 
                   size = 2.5,
              data = iou_results_df)
ax.legend_.remove()
sns.pairplot(iou_results_df.drop('iou_count',1), 
             diag_kind="kde",
             hue = 'anatomy', size = 5)
ax = sns.swarmplot(x = 'anatomy', 
              y = 'iou_pct_75', 
              hue = 'annotator_x', 
                   size = 3,
              data = iou_results_df)
ax.legend_.remove()
black_list = []
rem_annot_results = []
for ii in range(150):
    c_subset = out_iou_df[~out_iou_df['annotator_x'].isin(black_list)]
    c_subset = c_subset[~c_subset['annotator_y'].isin(black_list)]
    black_list += c_subset.groupby(['annotator_x']).agg({'iou': 'max'}).reset_index().sort_values('iou').head(1)['annotator_x'].values.tolist()
    c_subset = out_iou_df[~out_iou_df['annotator_x'].isin(black_list)]
    c_subset = c_subset[~c_subset['annotator_y'].isin(black_list)]
    if ii % 10==0:
        print('%2.2f%% mean agreement on %d annotations from %d annotators' %(c_subset['iou'].mean()*100, 
                                                       c_subset.shape[0],
                                                      c_subset['annotator_x'].drop_duplicates().shape[0]))
    rem_annot_results += [dict(annotations = c_subset.shape[0], 
                               annotators = c_subset['annotator_x'].drop_duplicates().shape[0],
                              mean_iou = c_subset['iou'].mean()*100)]
print(len(black_list), ', '.join(black_list[:10]))
out_iou_df['annot_id'] = out_iou_df.apply(lambda x: '{annotator_x}:{seriesUID}'.format(**x), 1)
black_list = []
rem_single_results = []
for ii in range(700):
    c_subset = out_iou_df[~out_iou_df['annot_id'].isin(black_list)]
    black_list += c_subset.groupby(['annot_id']).agg({'iou': 'max'}).reset_index().sort_values('iou').head(3)['annot_id'].values.tolist()
    c_subset = out_iou_df[~out_iou_df['annot_id'].isin(black_list)]
    
    if ii % 30 == 0:
        print('%2.2f%% mean agreement on %d annotations from %d annotators' %(c_subset['iou'].mean()*100, 
                                                       c_subset.shape[0],
                                                      c_subset['annotator_x'].drop_duplicates().shape[0]))
    rem_single_results += [dict(annotations = c_subset.shape[0], 
                               annotators = c_subset['annotator_x'].drop_duplicates().shape[0],
                              mean_iou = c_subset['iou'].mean()*100)]
ranked_ann_df = out_iou_df.groupby(['annotator_x']).apply(lambda x: pct_75(x['iou'])).reset_index().sort_values(0, ascending = False)
ranked_ann_df.columns = ['annotator', 'score']
ranked_ann_df['is_rad'] = ranked_ann_df['annotator'].map(annotator_status.get)
keep_rads = ranked_ann_df.query('is_rad=="radiologist"').head(2)
keep_rads
c_matchset = out_iou_df[out_iou_df['annotator_y'].isin(white_list)]
c_matchset = c_matchset[~c_matchset['annotator_x'].isin(white_list)]
match_agg = c_matchset.groupby(['annotator_x']).agg({'iou': pct_75}).reset_index()
match_agg = match_agg[~match_agg['annotator_x'].isin(white_list)]
match_agg.sort_values('iou', ascending = False)
white_list = keep_rads['annotator'].values.tolist()[:1]
add_annot_results = []
for ii in range(150):
    if ii>0:
        c_matchset = out_iou_df[out_iou_df['annotator_y'].isin(white_list)]
        c_matchset = c_matchset[~c_matchset['annotator_x'].isin(white_list)]
        match_agg = c_matchset.groupby(['annotator_x']).agg({'iou': pct_75}).reset_index()
        match_agg = match_agg[~match_agg['annotator_x'].isin(white_list)]
        white_list += match_agg.sort_values('iou', ascending = False).head(1)['annotator_x'].values.tolist()
    
    c_subset = out_iou_df[out_iou_df['annotator_x'].isin(white_list)]
    c_subset = c_subset[c_subset['annotator_y'].isin(white_list)]
    
    if ii % 30 == 0:
        print('%2.2f%% mean agreement on %d annotations from %d annotators' %(c_subset['iou'].mean()*100, 
                                                       c_subset.shape[0],
                                                      c_subset['annotator_x'].drop_duplicates().shape[0]))
    add_annot_results += [dict(annotations = c_subset.shape[0], 
                               annotators = c_subset['annotator_x'].drop_duplicates().shape[0],
                              mean_iou = c_subset['iou'].mean()*100)]
rem_annot = pd.DataFrame(rem_annot_results)
rem_single = pd.DataFrame(rem_single_results)
add_annot = pd.DataFrame(add_annot_results)
fig, ax1 = plt.subplots(1,1, figsize = (8, 8))
ax1.plot(out_iou_df.shape[0]-rem_annot['annotations'], rem_annot['mean_iou'], 'b.-', label = 'Removing Annotators')
ax1.plot(out_iou_df.shape[0]-rem_single['annotations'], rem_single['mean_iou'], 'g.-', label = 'Removing Annotations')
ax1.plot(out_iou_df.shape[0]-add_annot['annotations'], add_annot['mean_iou'], 'r.-', label = 'Adding Annotators')
ax1.legend()
ax1.set_title('Removed Annotations')
ax1.set_ylabel('Inter-reader DICE (%)');
from tqdm import tqdm_notebook
all_annot_results = []
for start_annot in tqdm_notebook(out_iou_df['annotator_x'].drop_duplicates()):
    white_list = [start_annot]
    for ii in range(50):
        if ii>0:
            c_matchset = out_iou_df[out_iou_df['annotator_y'].isin(white_list)]
            c_matchset = c_matchset[~c_matchset['annotator_x'].isin(white_list)]
            match_agg = c_matchset.groupby(['annotator_x']).agg({'iou': pct_75}).reset_index()
            match_agg = match_agg[~match_agg['annotator_x'].isin(white_list)]
            white_list += match_agg.sort_values('iou', ascending = False).head(1)['annotator_x'].values.tolist()

        c_subset = out_iou_df[out_iou_df['annotator_x'].isin(white_list)]
        c_subset = c_subset[c_subset['annotator_y'].isin(white_list)]

        all_annot_results += [dict(
            starting_annotator = start_annot,
            annotations = c_subset.shape[0], 
            annotators = c_subset['annotator_x'].drop_duplicates().shape[0],
            mean_iou = c_subset['iou'].mean()*100)]
all_annot = pd.DataFrame(all_annot_results).dropna()
ax = sns.factorplot(x = 'annotators', 
               y = 'mean_iou', 
               hue = 'starting_annotator',
              data = all_annot
              )
all_annot['ann_status'] = all_annot['starting_annotator'].map(annotator_status.get)
fig, ax1 = plt.subplots(1,1, figsize = (10, 10))
sns.swarmplot(x = 'annotators', 
              y = 'mean_iou', 
              hue = 'ann_status', 
              data = all_annot.query('annotators<=5 & annotators>0'),
              ax = ax1
             )
fig, ax1 = plt.subplots(1,1, figsize = (20, 10))
sns.violinplot(x = 'annotators', 
              y = 'mean_iou', 
              hue = 'ann_status', 
              data = all_annot.query('annotators<=10 & annotators>0'),
               ax = ax1
             )
