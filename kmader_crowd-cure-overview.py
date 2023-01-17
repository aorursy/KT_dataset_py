import os
data_dir = '../input'
!ls -R {data_dir} | head
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pydicom import read_file
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
base_dcm_dir = '../input/annotated_dicoms/'
ann_df = pd.read_csv(os.path.join(data_dir, 'CrowdsCureCancer2017Annotations.csv'))
test_rows = ann_df.sort_values(['anatomy','seriesUID']).head(501).tail(6)
test_rows.T
fig, m_axs = plt.subplots(2,3, figsize = (20, 12))
for (_, c_row), c_ax in zip(test_rows.iterrows(), m_axs.flatten()):
    t_dicom = read_file(os.path.join(base_dcm_dir, 
                                     c_row['patientID'], # format for dicom files
                                     c_row['seriesUID'], 
                                     str(c_row['sliceIndex'])))
    c_ax.imshow(t_dicom.pixel_array, cmap = 'bone')
    rect = Rectangle((min(c_row['start_x'], c_row['end_x']), 
                      min(c_row['start_y'], c_row['end_y'])), 
                     np.abs(c_row['end_x']-c_row['start_x']),
                     np.abs(c_row['end_y']-c_row['start_y'])
                     )
    c_ax.add_collection(PatchCollection([rect], alpha = 0.25, facecolor = 'red'))
    c_ax.set_title('{patientID} {anatomy}\n{annotator}\n{radiologist_status}'.format(**c_row))
    c_ax.axis('off')
test_rows = ann_df.sort_values(['anatomy','seriesUID']).head(11).tail(6)
fig, m_axs = plt.subplots(2,3, figsize = (20, 12))
for (_, c_row), c_ax in zip(test_rows.iterrows(), m_axs.flatten()):
    t_dicom = read_file(os.path.join(base_dcm_dir, 
                                     c_row['patientID'], # format for dicom files
                                     c_row['seriesUID'], 
                                     str(c_row['sliceIndex'])))
    c_ax.imshow(t_dicom.pixel_array, cmap = 'bone')
    circle = Circle((c_row['start_x'], c_row['start_y']), 
                     radius = c_row['length']/2
                     )
    c_ax.add_collection(PatchCollection([circle], alpha = 0.25, facecolor = 'red'))
    c_ax.set_title('{patientID} {anatomy}\n{annotator}\n{radiologist_status}'.format(**c_row))
    c_ax.axis('off')
from IPython.display import display, Markdown
# view annotators
dmark = lambda x: display(Markdown(x))
sum_view = lambda x, rows = 10: ann_df.groupby(x).count()[['order']].reset_index().sort_values('order', ascending = False).head(rows)
dmark('# Annotators')
display(sum_view(['annotator', 'radiologist_status']))
dmark('# Anatomy')
display(sum_view('anatomy'))
dmark('# Patient')
display(sum_view('patientID'))
sns.violinplot(x='anatomy', y = 'length', data = ann_df)
sns.violinplot(x='anatomy', y = 'sliceIndex', data = ann_df)
top_annotations = ann_df.groupby('anatomy').apply(
    lambda x: x.groupby('seriesUID').count()[['order']].reset_index().sort_values('order', ascending = False).head(5)).reset_index(drop = True)
#fig, ax1 = plt.subplots(1,1, figsize = (25, 8))
g = sns.factorplot(x = 'patientID', 
              y = 'length', 
              hue = 'radiologist_status',
               col = 'anatomy',
               kind = 'swarm',
                   sharex = False,
                   sharey = False,
              data = ann_df[ann_df['seriesUID'].isin(top_annotations['seriesUID'])])
g.set_xticklabels(rotation=90)
length_summary_df = ann_df.pivot_table(values = 'length', 
                   columns='radiologist_status', 
                   index = ['anatomy', 'seriesUID'],
                  aggfunc='mean').reset_index().dropna()
display(length_summary_df.groupby('anatomy').agg('mean').reset_index())
length_summary_df['mean'] = length_summary_df.apply(lambda x: 0.5*x['not_radiologist']+0.5*x['radiologist'], 1)
length_summary_df['not_radiologist'] = length_summary_df['not_radiologist'] / length_summary_df['mean']
length_summary_df['radiologist'] = length_summary_df['radiologist'] / length_summary_df['mean']
length_summary_df.sample(3)
sns.factorplot(
    x = 'anatomy',
    y = 'value',
    hue = 'radiologist_status',
    kind = 'swarm',
    data = pd.melt(length_summary_df, 
        id_vars = ['anatomy'], 
        value_vars = ['radiologist', 'not_radiologist']))
sns.factorplot(
    x = 'anatomy',
    y = 'value',
    hue = 'radiologist_status',
    kind = 'violin',
    data = pd.melt(length_summary_df, 
        id_vars = ['anatomy'], 
        value_vars = ['radiologist', 'not_radiologist']))
