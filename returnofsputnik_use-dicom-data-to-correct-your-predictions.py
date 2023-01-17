from functools import partial
from collections import defaultdict
import pydicom
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

np.warnings.filterwarnings('ignore')
labels = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_1_train_labels.csv')
details = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_1_detailed_class_info.csv')
# duplicates in details just have the same class so can be safely dropped
details = details.drop_duplicates('patientId').reset_index(drop=True)
labels_w_class = labels.merge(details, how='inner', on='patientId')
# get lists of all train/test dicom filepaths
train_dcm_fps = glob.glob('../input/rsna-pneumonia-detection-challenge/stage_1_train_images/*.dcm')
test_dcm_fps = glob.glob('../input/rsna-pneumonia-detection-challenge/stage_1_test_images/*.dcm')

train_dcms = [pydicom.read_file(x, stop_before_pixels=True) for x in train_dcm_fps]
test_dcms = [pydicom.read_file(x, stop_before_pixels=True) for x in test_dcm_fps]
def parse_dcm_metadata(dcm):
    unpacked_data = {}
    group_elem_to_keywords = {}
    # iterating here to force conversion from lazy RawDataElement to DataElement
    for d in dcm:
        pass
    # keys are pydicom.tag.BaseTag, values are pydicom.dataelem.DataElement
    for tag, elem in dcm.items():
        tag_group = tag.group
        tag_elem = tag.elem
        keyword = elem.keyword
        group_elem_to_keywords[(tag_group, tag_elem)] = keyword
        value = elem.value
        unpacked_data[keyword] = value
    return unpacked_data, group_elem_to_keywords

train_meta_dicts, tag_to_keyword_train = zip(*[parse_dcm_metadata(x) for x in train_dcms])
test_meta_dicts, tag_to_keyword_test = zip(*[parse_dcm_metadata(x) for x in test_dcms])
# join all the dicts
unified_tag_to_key_train = {k:v for dict_ in tag_to_keyword_train for k,v in dict_.items()}
unified_tag_to_key_test = {k:v for dict_ in tag_to_keyword_test for k,v in dict_.items()}

# quick check to make sure there are no different keys between test/train
assert len(set(unified_tag_to_key_test.keys()).symmetric_difference(set(unified_tag_to_key_train.keys()))) == 0

tag_to_key = {**unified_tag_to_key_test, **unified_tag_to_key_train}
tag_to_key
# using from_records here since some values in the dicts will be iterables and some are constants
train_df = pd.DataFrame.from_records(data=train_meta_dicts)
test_df = pd.DataFrame.from_records(data=test_meta_dicts)
train_df['dataset'] = 'train'
test_df['dataset'] = 'test'
df = pd.concat([train_df, test_df])
df.head(1)
# separating PixelSpacing list to single values
df['PixelSpacing_x'] = df['PixelSpacing'].apply(lambda x: x[0])
df['PixelSpacing_y'] = df['PixelSpacing'].apply(lambda x: x[1])
df = df.drop(['PixelSpacing'], axis='columns')

# x and y are always the same
assert sum(df['PixelSpacing_x'] != df['PixelSpacing_y']) == 0
# ReferringPhysicianName appears to just be empty strings
assert sum(df['ReferringPhysicianName'] != '') == 0

# SeriesDescription appears to be 'view: {}'.format(ViewPosition)
set(df['SeriesDescription'].unique())

# so these two columns don't have any useful info and can be safely dropped
nunique_all = df.aggregate('nunique')
nunique_all
# drop constant cols and other two from above
#ReferringPhysicianName is all ''
#PatientName is the same as PatientID
#PixelSpacing_y is the same as PixelSpacing_x
#The series and SOP UID's are just random numbers / id's, so I'm deleting them too
df = df.drop(nunique_all[nunique_all == 1].index.tolist() + ['SeriesDescription', 'ReferringPhysicianName', 'PatientName', 'PixelSpacing_y', 'SOPInstanceUID','SeriesInstanceUID','StudyInstanceUID'], axis='columns')

# now that we have a clean metadata dataframe we can merge back to our initial tabular data with target and class info
df = df.merge(labels_w_class, how='left', left_on='PatientID', right_on='patientId')

df['PatientAge'] = df['PatientAge'].astype(int)
# df now has multiple rows for some patients (those with multiple bounding boxes in label_w_class)
# so creating one with no duplicates for patients
df_deduped = df.drop_duplicates('PatientID', keep='first')
df_deduped.head()
#Correct ages that are mistyped
df_deduped.loc[df_deduped['PatientAge'] > 140, 'PatientAge'] = df_deduped.loc[df_deduped['PatientAge'] > 140, 'PatientAge'] - 100
#Convert binary features from categorical to 0/1
# Categorical features with Binary encode (0 or 1; two categories)
for bin_feature in ['PatientSex', 'ViewPosition']:
    df_deduped[bin_feature], uniques = pd.factorize(df_deduped[bin_feature])
#Drop the duplicated column patientID
del df_deduped['patientId']

#Drop columns that are going to be repetitive
del df_deduped['dataset']
df_deduped.head()
jonneoofs = pd.read_csv("../input/jonneoofs/jonne_oofs.csv")
jonneoofs = jonneoofs.sort_values('patientID').reset_index(drop=True)
andyharless_sub = pd.read_csv("../input/andyharless/submission (7).csv")
labels.head() #The real train
jonneoofs.head() #The oofs from Jonne's kernel
andyharless_sub.head() # The submission from Andy Harless, which is a fork from Jonne
jonneoofs['i_am_train'] = 1
andyharless_sub['i_am_train'] = 0
tr_te = jonneoofs.append(andyharless_sub)
del tr_te['confidence'] #Not used in grading
tr_te.columns = ['PatientID','x_guess','y_guess','width_guess','height_guess','i_am_train']
tr_te.head()
df_deduped.head()
merged_df = tr_te.merge(df_deduped, how='left', on='PatientID')
merged_df.head()
filledmerged_df = merged_df.fillna(-1) #Fill in missings
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold

train_df = filledmerged_df[filledmerged_df['i_am_train']==1]
test_df = filledmerged_df[filledmerged_df['i_am_train']==0]
             
#Cross validate with K Fold, 5 splits
folds = KFold(n_splits= 5, shuffle=True, random_state=2222)

# Create arrays and dataframes to store results
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
             
feats = [f for f in train_df.columns if f not in ['PatientID', 'i_am_train', 'x','y','width','height','Target','class']]

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats])):
    dtrain = lgb.Dataset(data=train_df[feats].iloc[train_idx], 
                         label=train_df['x'].iloc[train_idx], 
                         free_raw_data=False, silent=True)
    dvalid = lgb.Dataset(data=train_df[feats].iloc[valid_idx], 
                         label=train_df['x'].iloc[valid_idx], 
                         free_raw_data=False, silent=True)

    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'nthread': 4,
        'learning_rate': 0.10, 
        'max_depth': 2,
        #'reg_alpha': 0,
        #'reg_lambda': 0,
        #'min_split_gain': 0.0222415,
        'seed': 15000,
        'verbose': 50,
        'metric': 'l2',
    }

    clf = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dvalid],
        early_stopping_rounds=50,
        verbose_eval=True
    )

    oof_preds[valid_idx] = clf.predict(dvalid.data)
    sub_preds += clf.predict(test_df[feats]) / folds.n_splits

xpreds_oof = oof_preds.copy()
xpreds_sub = sub_preds.copy()
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold

train_df = filledmerged_df[filledmerged_df['i_am_train']==1]
test_df = filledmerged_df[filledmerged_df['i_am_train']==0]
             
#Cross validate with K Fold, 5 splits
folds = KFold(n_splits= 5, shuffle=True, random_state=2222)

# Create arrays and dataframes to store results
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
             
feats = [f for f in train_df.columns if f not in ['PatientID', 'i_am_train', 'x','y','width','height','Target','class']]

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats])):
    dtrain = lgb.Dataset(data=train_df[feats].iloc[train_idx], 
                         label=train_df['y'].iloc[train_idx], 
                         free_raw_data=False, silent=True)
    dvalid = lgb.Dataset(data=train_df[feats].iloc[valid_idx], 
                         label=train_df['y'].iloc[valid_idx], 
                         free_raw_data=False, silent=True)

    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'nthread': 4,
        'learning_rate': 0.10, 
        'max_depth': 2,
        #'reg_alpha': 0,
        #'reg_lambda': 0,
        #'min_split_gain': 0.0222415,
        'seed': 15000,
        'verbose': 50,
        'metric': 'l2',
    }

    clf = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dvalid],
        early_stopping_rounds=50,
        verbose_eval=True
    )

    oof_preds[valid_idx] = clf.predict(dvalid.data)
    sub_preds += clf.predict(test_df[feats]) / folds.n_splits

ypreds_oof = oof_preds.copy()
ypreds_sub = sub_preds.copy()
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold

train_df = filledmerged_df[filledmerged_df['i_am_train']==1]
test_df = filledmerged_df[filledmerged_df['i_am_train']==0]
             
#Cross validate with K Fold, 5 splits
folds = KFold(n_splits= 5, shuffle=True, random_state=2222)

# Create arrays and dataframes to store results
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
             
feats = [f for f in train_df.columns if f not in ['PatientID', 'i_am_train', 'x','y','width','height','Target','class']]

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats])):
    dtrain = lgb.Dataset(data=train_df[feats].iloc[train_idx], 
                         label=train_df['width'].iloc[train_idx], 
                         free_raw_data=False, silent=True)
    dvalid = lgb.Dataset(data=train_df[feats].iloc[valid_idx], 
                         label=train_df['width'].iloc[valid_idx], 
                         free_raw_data=False, silent=True)

    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'nthread': 4,
        'learning_rate': 0.10, 
        'max_depth': 2,
        #'reg_alpha': 0,
        #'reg_lambda': 0,
        #'min_split_gain': 0.0222415,
        'seed': 15000,
        'verbose': 50,
        'metric': 'l2',
    }

    clf = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dvalid],
        early_stopping_rounds=50,
        verbose_eval=True
    )

    oof_preds[valid_idx] = clf.predict(dvalid.data)
    sub_preds += clf.predict(test_df[feats]) / folds.n_splits

widthpreds_oof = oof_preds.copy()
widthpreds_sub = sub_preds.copy()
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold

train_df = filledmerged_df[filledmerged_df['i_am_train']==1]
test_df = filledmerged_df[filledmerged_df['i_am_train']==0]
             
#Cross validate with K Fold, 5 splits
folds = KFold(n_splits= 5, shuffle=True, random_state=2222)

# Create arrays and dataframes to store results
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
             
feats = [f for f in train_df.columns if f not in ['PatientID', 'i_am_train', 'x','y','width','height','Target','class']]

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats])):
    dtrain = lgb.Dataset(data=train_df[feats].iloc[train_idx], 
                         label=train_df['height'].iloc[train_idx], 
                         free_raw_data=False, silent=True)
    dvalid = lgb.Dataset(data=train_df[feats].iloc[valid_idx], 
                         label=train_df['height'].iloc[valid_idx], 
                         free_raw_data=False, silent=True)

    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'nthread': 4,
        'learning_rate': 0.10, 
        'max_depth': 2,
        #'reg_alpha': 0,
        #'reg_lambda': 0,
        #'min_split_gain': 0.0222415,
        'seed': 15000,
        'verbose': 50,
        'metric': 'l2',
    }

    clf = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dvalid],
        early_stopping_rounds=50,
        verbose_eval=True
    )

    oof_preds[valid_idx] = clf.predict(dvalid.data)
    sub_preds += clf.predict(test_df[feats]) / folds.n_splits

heightpreds_oof = oof_preds.copy()
heightpreds_sub = sub_preds.copy()
# What is the number of rows where we have a box?
train_df.loc[train_df['x'] > -1]['x'].shape[0] / train_df.shape[0]
train_df['xpredsoof'] = xpreds_oof
train_df['ypredsoof'] = ypreds_oof
train_df['widthpredsoof'] = widthpreds_oof
train_df['heightpredsoof'] = heightpreds_oof
train_df.loc[train_df['widthpredsoof'] <= 100]
#train_df.loc[(train_df['xpredsoof'] > 130) & (train_df['ypredsoof'] > 134)].shape[0] / train_df.shape[0]
train_df.loc[(train_df['widthpredsoof'] > 100)].shape[0] / train_df.shape[0]
andyharless_sub['xpred'] = xpreds_sub
andyharless_sub['ypred'] = ypreds_sub
andyharless_sub['widthpred'] = widthpreds_sub
andyharless_sub['heightpred'] = heightpreds_sub

andyharless_sub['xpred'] = andyharless_sub['xpred'].round()
andyharless_sub['ypred'] = andyharless_sub['ypred'].round()
andyharless_sub['widthpred'] = andyharless_sub['widthpred'].round()
andyharless_sub['heightpred'] = andyharless_sub['heightpred'].round()
#andyharless_sub.loc[andyharless_sub['widthpred'] <= 100, 'xpred'] = ''
#andyharless_sub.loc[andyharless_sub['widthpred'] <= 100, 'ypred'] = ''
#andyharless_sub.loc[andyharless_sub['widthpred'] <= 100, 'heightpred'] = ''
#andyharless_sub.loc[andyharless_sub['widthpred'] <= 100, 'widthpred'] = ''
andyharless_sub['confidence'] = '1'
andyharless_sub.head()
#del andyharless_sub['x']
#del andyharless_sub['y']
#del andyharless_sub['width']
#del andyharless_sub['height']
#del andyharless_sub['i_am_train']
andyharless_sub['PredictionString'] = andyharless_sub['confidence'].map(str)+' '+andyharless_sub['xpred'].map(str)+' '+andyharless_sub['ypred'].map(str)+' '+andyharless_sub['widthpred'].map(str)+' '+andyharless_sub['heightpred'].map(str)
andyharless_sub.loc[andyharless_sub['PredictionString']=='1    ', 'PredictionString'] = '' #Correct empties
andyharless_sub.loc[andyharless_sub['x'].isnull(), 'PredictionString'] = '' #Remove boxes if we predicted there were none
andyharless_sub[['patientID','PredictionString']].to_csv('dicom_corrections.csv', index=False)
