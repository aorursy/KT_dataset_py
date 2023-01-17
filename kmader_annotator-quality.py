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
# isotropic-ish assumption
ann_3d_df['z0'] = ann_3d_df.apply(lambda x: x['sliceIndex']-0.25*((x['x1']-x['x0'])/2+(x['y1']-x['y0'])/2), 1)
ann_3d_df['z1'] = ann_3d_df.apply(lambda x: x['sliceIndex']+0.25*((x['x1']-x['x0'])/2+(x['y1']-x['y0'])/2), 1)
ann_3d_df['bcube'] = ann_3d_df.apply(lambda x: tuple(int(y) for y in [x['x0'], x['x1'], x['y0'], x['y1'], x['z0'], x['z1']]), 1)
ann_3d_df.sample(2)
# build base grid for all images
step_size = 2
base_x = np.arange(ann_3d_df['x0'].min(), ann_3d_df['x1'].max(), step_size)
base_y = np.arange(ann_3d_df['y0'].min(), ann_3d_df['y1'].max(), step_size)
base_z = np.arange(ann_3d_df['z0'].min(), ann_3d_df['z1'].max(), step_size)
xx, yy, zz = np.meshgrid(base_x, base_y, base_z, indexing = 'ij')
bb_func = lambda c_row: (xx>=c_row['x0']) & (xx<=c_row['x1']) & (yy>=c_row['y0']) & (yy<=c_row['y1']) & (zz>=c_row['z0']) & (zz<=c_row['z1'])
bb_func = lambda bcube: (xx>=bcube[0]) & (xx<=bcube[1]) & (yy>=bcube[2]) & (yy<=bcube[3]) & (zz>=bcube[4]) & (zz<=bcube[5])
def calc_iou(cube_a, cube_b):
    bcube_a = bb_func(cube_a)
    bcube_b = bb_func(cube_b)
    return np.sum(bcube_a & bcube_b)/(1+np.sum(bcube_a+bcube_b))
print(calc_iou((0, 100, 0, 100, 0, 100),
        (0, 100, 0, 100, 0, 100))) # should be 1 since they overlap 100%
print(calc_iou((0, 100, 0, 100, 0, 100),
        (0, 100, 0, 100, 50, 100))) # should be 0.5 since they overlap 50%
_, j_row = next(ann_3d_df.iterrows())
_, k_row = next(ann_3d_df.sample(1).iterrows())
print('j->j IOU:', calc_iou(j_row['bcube'], j_row['bcube']))
print('j->k IOU:', calc_iou(j_row['bcube'], k_row['bcube']))
%%time
test_series_id = ann_3d_df.groupby('seriesUID').agg({'order':'count'}).reset_index().sort_values('order', ascending = False).head(1)['seriesUID'].values
cur_ann_df = list(ann_3d_df[ann_3d_df['seriesUID'].isin(test_series_id)].iterrows())
conf_mat = np.eye(len(cur_ann_df))
for i, (_, c_row) in enumerate(cur_ann_df):
    for j, (_, d_row) in enumerate(cur_ann_df[i+1:], i+1):
        c_iou = calc_iou(c_row['bcube'], d_row['bcube'])
        conf_mat[i,j] = c_iou
        conf_mat[j,i] = c_iou
sns.heatmap(conf_mat, 
            annot = True, 
            fmt = '2.2f')
simple_lesions = ann_3d_df[['anatomy', 'seriesUID', 'annotator', 'bcube']]
lesion_product_df = pd.merge(simple_lesions, simple_lesions, on = ['anatomy', 'seriesUID'])
lesion_product_df['is_first'] =  lesion_product_df.apply(lambda x: x['annotator_x']<x['annotator_y'], 1)
print(lesion_product_df.shape[0])
lesion_product_df = lesion_product_df[lesion_product_df['is_first']].drop(['is_first'],1)
lesion_product_df.sample(2)
# only process a single subgroup
if False:
    lesion_product_df = lesion_product_df[lesion_product_df['anatomy'].isin(['Lung'])]
print(lesion_product_df.shape[0])
import dask
import dask.bag as dbag
import dask.diagnostics as diag
from multiprocessing.pool import ThreadPool
def process_row(in_rowpair):
    # avoid duplicate comptuation by using the annotator name to order the query
    i, c_row = in_rowpair
    return i, calc_iou(c_row['bcube_x'], c_row['bcube_y'])
all_results = dbag.from_sequence(list(lesion_product_df.sort_values('seriesUID').iterrows()), 
                                 partition_size = 10).map(process_row)
all_results
with diag.ProgressBar(), dask.set_options(pool = ThreadPool(4)):
    iou_results = all_results.compute()
# reinsert the results into the dataframe
out_iou = lesion_product_df['anatomy'].map(lambda x: 0.0)
for i, score in iou_results:
    out_iou[i] = score
lesion_product_df['iou'] = out_iou
# add back the duplicated rows by swapping annotator_x and annotator_y
out_iou_df = lesion_product_df[['anatomy', 'seriesUID', 'annotator_x', 'annotator_y', 'iou']].copy()
out_iou_swapped = out_iou_df.copy()
out_iou_swapped['annotator_x'] = out_iou_df['annotator_y']
out_iou_swapped['annotator_y'] = out_iou_df['annotator_x']
out_iou_df = pd.concat([out_iou_df, out_iou_swapped]).sort_values(['seriesUID', 'annotator_x'])
out_iou_df = out_iou_df.fillna(0.0)
out_iou_df.to_csv('matching_results.csv')
out_iou_df.sample(2)
def pct_90(x): 
    return np.percentile(x, 90)
iou_results_df = out_iou_df.groupby(['anatomy', 'seriesUID', 'annotator_x']).agg({'iou': ['count', 'mean', 'max', pct_90, 'median']}).reset_index()
iou_results_df.columns = ['_'.join([y for y in col if len(y)>0]).strip() for col in iou_results_df.columns.values]
iou_results_df.sample(5)
annotator_status = {k['annotator']: k['radiologist_status'] for k in 
 ann_3d_df[['annotator', 'radiologist_status']].drop_duplicates().T.to_dict().values()}
iou_results_df['radiologist_status'] = iou_results_df['annotator_x'].map(annotator_status.get)
sns.pairplot(iou_results_df.drop('iou_count',1), 
             hue = 'radiologist_status', size = 5)
ax = sns.swarmplot(x = 'radiologist_status', 
              y = 'iou_pct_90', 
              hue = 'annotator_x', 
              data = iou_results_df)
ax.legend_.remove()
sns.pairplot(iou_results_df.drop('iou_count',1), 
             hue = 'anatomy', size = 5)
ax = sns.swarmplot(x = 'anatomy', 
              y = 'iou_pct_90', 
              hue = 'annotator_x', 
              data = iou_results_df)
ax.legend_.remove()
black_list = []
rem_annot_results = []
for ii in range(180):
    c_subset = out_iou_df[~out_iou_df['annotator_x'].isin(black_list)]
    black_list += c_subset.groupby(['annotator_x']).agg({'iou': 'max'}).reset_index().sort_values('iou').head(1)['annotator_x'].values.tolist()
    c_subset = out_iou_df[~out_iou_df['annotator_x'].isin(black_list)]
    if ii % 10==0:
        print('%2.2f%% mean agreement on %d annotations from %d annotators' %(c_subset['iou'].mean()*100, 
                                                       c_subset.shape[0],
                                                      c_subset['annotator_x'].drop_duplicates().shape[0]))
    rem_annot_results += [dict(annotations = c_subset.shape[0], 
                               annotators = c_subset['annotator_x'].drop_duplicates().shape[0],
                              mean_iou = c_subset['iou'].mean()*100)]
iou_results_df = c_subset.groupby(['anatomy', 'seriesUID', 'annotator_x']).agg({'iou': ['count', 'mean', 'max', pct_90, 'median']}).reset_index()
iou_results_df.columns = ['_'.join([y for y in col if len(y)>0]).strip() for col in iou_results_df.columns.values]
iou_results_df['radiologist_status'] = iou_results_df['annotator_x'].map(annotator_status.get)
sns.pairplot(iou_results_df.drop('iou_count',1), 
             hue = 'radiologist_status', 
             size = 5)
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
iou_results_df = c_subset.groupby(['anatomy', 'seriesUID', 'annotator_x']).agg({'iou': ['count', 'mean', 'max', pct_90, 'median']}).reset_index()
iou_results_df.columns = ['_'.join([y for y in col if len(y)>0]).strip() for col in iou_results_df.columns.values]
iou_results_df['radiologist_status'] = iou_results_df['annotator_x'].map(annotator_status.get)
sns.pairplot(iou_results_df.drop('iou_count',1), 
             hue = 'radiologist_status', 
             size = 5)
rem_annot = pd.DataFrame(rem_annot_results)
rem_single = pd.DataFrame(rem_single_results)
fig, ax1 = plt.subplots(1,1, figsize = (8, 8))
ax1.plot(out_iou_df.shape[0]-rem_annot['annotations'], rem_annot['mean_iou'], 'b.-', label = 'Removing Annotators')
ax1.plot(out_iou_df.shape[0]-rem_single['annotations'], rem_single['mean_iou'], 'g.-', label = 'Removing Annotations')
ax1.legend()
ax1.set_title('Removed Annotations')
ax1.set_ylabel('Inter-reader DICE (%)');
