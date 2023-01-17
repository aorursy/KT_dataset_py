import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
import gc
import warnings
warnings.simplefilter(action = 'ignore')
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.model_selection import KFold
detailed_class_info = pd.read_csv('../input/stage_1_detailed_class_info.csv')
train_labels = pd.read_csv('../input/stage_1_train_labels.csv')

df = pd.merge(left = detailed_class_info, right = train_labels, how = 'left', on = 'patientId')

del detailed_class_info, train_labels
gc.collect()

df.info(null_counts = True)
df.head()
df = df.drop_duplicates()
df.info()
df['patientId'].value_counts().head(10)
df[df['patientId'] == '32408669-c137-4e8d-bd62-fe8345b40e73']
df['patientId'].value_counts().value_counts()
df[df['Target'] == 0]['patientId'].value_counts().value_counts()
sns.countplot(x = 'class', hue = 'Target', data = df);
df[df['class'] == 'Lung Opacity']['Target'].value_counts(dropna = False)
df[df['class'] == 'No Lung Opacity / Not Normal']['Target'].value_counts(dropna = False)
df[df['class'] == 'Normal']['Target'].value_counts(dropna = False)
print('Patients can have {} different classes'.format(df.groupby('patientId')['class'].nunique().nunique()))
df_areas = df.dropna()[['x', 'y', 'width', 'height']].copy()
df_areas['x_2'] = df_areas['x'] + df_areas['width']
df_areas['y_2'] = df_areas['y'] + df_areas['height']
df_areas['x_center'] = df_areas['x'] + df_areas['width'] / 2
df_areas['y_center'] = df_areas['y'] + df_areas['height'] / 2
df_areas['area'] = df_areas['width'] * df_areas['height']

df_areas.head()
sns.jointplot(x = 'x', y = 'y', data = df_areas, kind = 'hex', gridsize = 20);
sns.jointplot(x = 'x_center', y = 'y_center', data = df_areas, kind = 'hex', gridsize = 20);
sns.jointplot(x = 'x_2', y = 'y_2', data = df_areas, kind = 'hex', gridsize = 20);
sns.jointplot(x = 'width', y = 'height', data = df_areas, kind = 'hex', gridsize = 20);
n_columns = 3
n_rows = 3
_, axes = plt.subplots(n_rows, n_columns, figsize=(8 * n_columns, 5 * n_rows))
for i, c in enumerate(df_areas.columns):
    sns.boxplot(y = c, data = df_areas, ax = axes[i // n_columns, i % n_columns])
plt.tight_layout()
plt.show()
df_areas[df_areas['width'] > 500]
pid_width = list(df[df['width'] > 500]['patientId'].values)
df[df['patientId'].isin(pid_width)]
df_areas[df_areas['height'] > 900].shape[0]
pid_height = list(df[df['height'] > 900]['patientId'].values)
df[df['patientId'].isin(pid_height)]
df = df[~df['patientId'].isin(pid_width + pid_height)]
df.shape
df_meta = df.drop('class', axis = 1).copy()
dcm_columns = None

for n, pid in enumerate(df_meta['patientId'].unique()):
    dcm_file = '../input/stage_1_train_images/%s.dcm' % pid
    dcm_data = pydicom.read_file(dcm_file)
    
    if not dcm_columns:
        dcm_columns = dcm_data.dir()
        dcm_columns.remove('PixelSpacing')
        dcm_columns.remove('PixelData')
    
    for col in dcm_columns:
        if not (col in df_meta.columns):
            df_meta[col] = np.nan
        index = df_meta[df_meta['patientId'] == pid].index
        df_meta.loc[index, col] = dcm_data.data_element(col).value
        
    del dcm_data
    
gc.collect()

df_meta.head()
to_drop = df_meta.nunique()
to_drop = to_drop[(to_drop <= 1) | (to_drop == to_drop['patientId'])].index
to_drop = to_drop.drop('patientId')
to_drop
df_meta.drop(to_drop, axis = 1, inplace = True)
df_meta.head()
print('Dropped {} useless features'.format(len(to_drop)))
df_meta.nunique()
sum(df_meta['ReferringPhysicianName'].unique() != '')
df_meta.drop('ReferringPhysicianName', axis = 1, inplace = True)
df_meta['PatientAge'] = df_meta['PatientAge'].astype(int)
df_meta['SeriesDescription'] = df_meta['SeriesDescription'].map({'view: AP': 'AP', 'view: PA': 'PA'})
df_meta.head()
print('There are {} equal elements between SeriesDescription and ViewPosition from {}.' \
      .format(sum(df_meta['SeriesDescription'] == df_meta['ViewPosition']), df_meta.shape[0]))
df_meta.drop('SeriesDescription', axis = 1, inplace = True)
plt.figure(figsize = (25, 5))
sns.countplot(x = 'PatientAge', hue = 'Target', data = df_meta);
sns.countplot(x = 'PatientSex', hue = 'Target', data = df_meta);
sns.countplot(x = 'ViewPosition', hue = 'Target', data = df_meta);
df_meta['PatientSex'] = df_meta['PatientSex'].map({'F': 0, 'M': 1})
df_meta['ViewPosition'] = df_meta['ViewPosition'].map({'PA': 0, 'AP': 1})
df_meta.head()
df_meta.corr()
def fast_lgbm_cv_scores(df, target, task, rs = 0):
    warnings.simplefilter('ignore')
    
    if task == 'classification':
        clf = LGBMClassifier(n_estimators = 10000, nthread = 4, random_state = rs)
        metric = 'auc'
    else:
        clf = LGBMRegressor(n_estimators = 10000, nthread = 4, random_state = rs)
        metric = 'mean_absolute_error'

    # Cross validation model
    folds = KFold(n_splits = 2, shuffle = True, random_state = rs)
        
    # Create arrays and dataframes to store results
    pred = np.zeros(df.shape[0])
    
    feats = df.columns.drop(target)
    
    feature_importance_df = pd.DataFrame(index = feats)
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df[feats], df[target])):
        train_x, train_y = df[feats].iloc[train_idx], df[target].iloc[train_idx]
        valid_x, valid_y = df[feats].iloc[valid_idx], df[target].iloc[valid_idx]

        clf.fit(train_x, train_y, 
                eval_set = [(valid_x, valid_y)], eval_metric = metric, 
                verbose = -1, early_stopping_rounds = 100)

        if task == 'classification':
            pred[valid_idx] = clf.predict_proba(valid_x, num_iteration = clf.best_iteration_)[:, 1]
        else:
            pred[valid_idx] = clf.predict(valid_x, num_iteration = clf.best_iteration_)
        
        feature_importance_df[n_fold] = pd.Series(clf.feature_importances_, index = feats)
        
        del train_x, train_y, valid_x, valid_y
        gc.collect()

    if task == 'classification':    
        return feature_importance_df, pred, roc_auc_score(df[target], pred)
    else:
        return feature_importance_df, pred, mean_absolute_error(df[target], pred)
f_imp, _, score = fast_lgbm_cv_scores(df_meta.drop(['patientId', 'x', 'y', 'width', 'height'], axis = 1), 
                                      target = 'Target', task = 'classification')
print('ROC-AUC for Target = {}'.format(score))
f_imp
for c in ['x', 'y', 'width', 'height']:
    df_meta[c] = df_meta[c].fillna(-1)
df_meta.head()
f_imp, pred, score = fast_lgbm_cv_scores(df_meta[['x', 'PatientAge', 'PatientSex', 'ViewPosition']], 
                                   target = 'x', task = 'regression')
print('MAE for x = {}'.format(score))
val = df_meta[['x']]
val['pred'] = pred
val['error'] = abs(val['x'] - val['pred'])
val[['pred', 'error', 'x']].sort_values('x').reset_index(drop = True).plot();
f_imp
f_imp, pred, score = fast_lgbm_cv_scores(df_meta[['y', 'PatientAge', 'PatientSex', 'ViewPosition']], 
                                   target = 'y', task = 'regression')
print('MAE for y = {}'.format(score))
val = df_meta[['y']]
val['pred'] = pred
val['error'] = abs(val['y'] - val['pred'])
val[['pred', 'error', 'y']].sort_values('y').reset_index(drop = True).plot();
f_imp
f_imp, pred, score = fast_lgbm_cv_scores(df_meta[['width', 'PatientAge', 'PatientSex', 'ViewPosition']], 
                                   target = 'width', task = 'regression')
print('MAE for width = {}'.format(score))
val = df_meta[['width']]
val['pred'] = pred
val['error'] = abs(val['width'] - val['pred'])
val[['pred', 'error', 'width']].sort_values('width').reset_index(drop = True).plot();
f_imp
f_imp, pred, score = fast_lgbm_cv_scores(df_meta[['height', 'PatientAge', 'PatientSex', 'ViewPosition']], 
                                   target = 'height', task = 'regression')
print('MAE for height = {}'.format(score))
val = df_meta[['height']]
val['pred'] = pred
val['error'] = abs(val['height'] - val['pred'])
val[['pred', 'error', 'height']].sort_values('height').reset_index(drop = True).plot();
f_imp
