import numpy as np
import pandas as pd
from glob import glob
import cv2
from matplotlib import pyplot as plt
import seaborn as sns
DATA_ROOT_PATH = '../input/data-engine-analysis'
predictions = pd.read_csv(f'{DATA_ROOT_PATH}/all_train_eval_images.csv')
ground_truth = pd.read_csv(f'{DATA_ROOT_PATH}/df_final.csv')
predictions.drop(['Unnamed: 0'],axis=1,inplace=True)
predictions.head()
false_negatives = ground_truth[ground_truth['results']=='FN']
false_negatives.head()
print(f'percentage of false negatives out of all predictions: {len(false_positives)/len(predictions)}')
image_ids = false_negatives['image_id'].unique()
fn_stats = pd.DataFrame()
for i, image_id in enumerate(image_ids):
    fn_stats.loc[i, 'image_id']= image_id
    fn_stats.loc[i, 'no_of_fn'] = len(false_negatives[false_negatives['image_id']==image_id]['results']=='FN')
over_5_fn_stats = fn_stats[fn_stats['no_of_fn']>5.]
over_5_fn_stats.to_csv('over_5_fn_stats.csv',index=False)
fn_stats['no_of_fn'] = np.uint8(fn_stats['no_of_fn'])
fn_stats.head()
predictions.head()
predictions['bbox_area'] = predictions['bbox_eval_width'] * predictions['bbox_eval_height']

# Define small, medium and large according to Wheat EDA
AREA_SMALL = 56 * 56
AREA_MEDIUM = 96 * 96
predictions['bbox_size'] = 'large'
predictions.loc[predictions['bbox_area'] < AREA_MEDIUM, 'bbox_size'] = 'medium'
predictions.loc[predictions['bbox_area'] < AREA_SMALL, 'bbox_size'] = 'small'
predictions.to_csv('train_set_inference.csv',index=False)
predictions.head()
