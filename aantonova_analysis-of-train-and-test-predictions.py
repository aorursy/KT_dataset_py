import numpy as np
import pandas as pd
import time
import warnings
warnings.simplefilter(action = 'ignore')
from sklearn.metrics import accuracy_score, precision_score, recall_score
from matplotlib import pyplot as plt
import seaborn as sns
# transform prediction from string to table
def transform_prediction_to_table(df):
    prediction = pd.DataFrame(index = ['patientId', 'confidence', 'x', 'y', 'width', 'height'])

    for _, row in df.iterrows():
        # all except 'NAN'
        if len(str(row['PredictionString'])) > 3:
            row_array = row['PredictionString'].strip().split(' ')
            for i in range(int(len(row_array) / 5)):
                prediction[prediction.shape[1]] = [row['patientId'], row_array[i * 5]] + \
                                                  [b for b in row_array[i * 5 + 1 : i * 5 + 5]]
        else:
            prediction[prediction.shape[1]] = [row['patientId'], -1, -1, -1, -1, -1]

    prediction = prediction.T
    prediction['confidence'] = prediction['confidence'].astype(float)
    prediction['x'] = prediction['x'].astype(int)
    prediction['y'] = prediction['y'].astype(int)
    prediction['width'] = prediction['width'].astype(int)
    prediction['height'] = prediction['height'].astype(int)
    
    prediction.replace(-1, np.nan, inplace = True)
    return prediction
# helper function to calculate IoU
# based on kernel https://www.kaggle.com/chenyc15/mean-average-precision-metric
def iou(box1, box2):
    x11, y11, w1, h1 = list(map(int, box1))
    x21, y21, w2, h2 = list(map(int, box2))
    assert w1 * h1 > 0
    assert w2 * h2 > 0
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2 - xi1) * (yi2 - yi1)
        union = area1 + area2 - intersect
        return intersect / union
# calculate a single IoU metric, based on masks, for two table's rows with boxes, associated with one patientId
def mask_iou(boxes_true, boxes_pred):
    mask_true = np.zeros((1024, 1024))
    mask_pred = np.zeros((1024, 1024))
    
    for _, box in boxes_true.iterrows():
        x1 = int(box['x'])
        y1 = int(box['y'])
        x2 = x1 + int(box['width'])
        y2 = y1 + int(box['height'])
        mask_true[y1 : y2, x1 : x2] = 1

    for _, box in boxes_pred.iterrows():
        x1 = int(box['x'])
        y1 = int(box['y'])
        x2 = x1 + int(box['width'])
        y2 = y1 + int(box['height'])
        mask_pred[y1 : y2, x1 : x2] = 1

    mask_i = mask_true * mask_pred
    mask_u = mask_true + mask_pred - mask_i
    
    return float(sum(sum(mask_i))) / sum(sum(mask_u))
# calculate Mean Average Precision IoU metric for two table's rows with boxes, associated with one patientId
# based on kernel https://www.kaggle.com/chenyc15/mean-average-precision-metric
def map_iou(boxes_true, boxes_pred, thresholds = [.4, .45, .5, .55, .6, .65, .7, .75]):
    
    # According to the introduction, images with no ground truth bboxes will not be 
    # included in the map score unless there is a false positive detection (?)
        
    # return None if both are empty, don't count the image in final evaluation (?)
    if (boxes_true.shape[0] == 0) and (boxes_pred.shape[0] == 0):
        return None
    
    # [x, y, w, h] for boxes_true
    # [confidence, x, y, w, h] for boxes_pred
    assert (boxes_true.shape[1] == 4 and boxes_pred.shape[1] == 5), 'Boxes shape error'
    
    # sort boxes_pred by scores in decreasing order
    boxes_pred = boxes_pred.sort_values('confidence', ascending = False)
    
    map_total = 0
    
    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in boxes_true.iterrows():
            matched = False
            for j, bp in boxes_pred.iterrows():
                miou = iou(bt, bp[1:])
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1 # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1 # bt has no match, count as FN
                
        fp = boxes_pred.shape[0] - len(matched_bt) # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        map_total += m
    
    return map_total / len(thresholds)
# calculate different metrics and aggregated values for one prediction
def get_table_for_one_prediction(df_pred, labels = None):
    # get values for one patient's boxes
    def get_evals(boxes):
        if boxes.shape[0] > 0:
            area_sum = 0
            area_min = 1e7
            area_max = 0
            area_cnt = 0
            for j, row in boxes.iterrows():
                area = row['width'] * row['height']
                area_sum += area
                area_cnt += 1
                if area < area_min:
                    area_min = area
                if area > area_max:
                    area_max = area
            return [int(area_cnt > 0), area_cnt, area_sum, area_min, area_max, float(area_sum) / area_cnt]
        else:
            return [0] * 6
    
    if labels is None:
        # create table for test prediction
        res = pd.DataFrame(index = ['pred_target', 'pred_cnt', 'pred_area_sum', 'pred_area_min', 
                                    'pred_area_max', 'pred_area_mean'])
    
        for patientId in df_pred['patientId'].unique():
            pred_boxes = df_pred[df_pred['patientId'] == patientId][['confidence', 'x', 'y', 'width', 'height']].dropna()
            res[patientId] = get_evals(pred_boxes)
    else:
        # create table for train prediction
        res = pd.DataFrame(index = ['mask_iou', 'map_iou', 
                'true_target', 'true_cnt', 'true_area_sum', 'true_area_min', 'true_area_max', 'true_area_mean',
                'pred_target', 'pred_cnt', 'pred_area_sum', 'pred_area_min', 'pred_area_max', 'pred_area_mean'])
    
        for patientId in labels['patientId'].unique():
            pred_boxes = df_pred[df_pred['patientId'] == patientId][['confidence', 'x', 'y', 'width', 'height']].dropna()
            true_boxes = labels[labels['patientId'] == patientId][['x', 'y', 'width', 'height']].dropna()
            res[patientId] = [mask_iou(true_boxes, pred_boxes), map_iou(true_boxes, pred_boxes)] + \
                             get_evals(true_boxes) + get_evals(pred_boxes)
    
    return res.T
# calculate different metrics and aggregated values for all predictions
def get_table_for_all_predictions(prediction_files, labels = None):
    if labels is None:
        res = pd.DataFrame(index = ['LB_score_for_test', 
                              'max_cnt', 'mean_area_sum', 'mean_area_min', 'mean_area_max', 'mean_area_mean'])
    else:
        res = pd.DataFrame(index = ['LB_score_for_test', 'mean_map_iou', 'mean_mask_iou', 
                              'accuracy', 'precision_0', 'precision_1', 'recall_0', 'recall_1',
                              'max_cnt', 'mean_area_sum', 'mean_area_min', 'mean_area_max', 'mean_area_mean'])

    for key in prediction_files.keys():
        print(key, time.ctime())
        prediction_string = pd.read_csv(PREDICTIONS_FOLDER + key)
        prediction = transform_prediction_to_table(prediction_string)
        eval_table = get_table_for_one_prediction(prediction, labels)
        
        if labels is None:
            # create table for test prediction
            res[key] = [prediction_files[key], 
                 eval_table['pred_cnt'].max(), eval_table['pred_area_sum'].mean(), 
                 eval_table['pred_area_min'].mean(), eval_table['pred_area_max'].mean(), 
                 eval_table['pred_area_mean'].mean()]
        else:
            # create table for train prediction
            res[key] = [prediction_files[key], 
                 eval_table['map_iou'].mean(), eval_table['mask_iou'].mean(), 
                 accuracy_score(eval_table['true_target'], eval_table['pred_target'])] + \
                 list(precision_score(eval_table['true_target'], eval_table['pred_target'], average = None)) + \
                 list(recall_score(eval_table['true_target'], eval_table['pred_target'], average = None)) + \
                [eval_table['pred_cnt'].max(), eval_table['pred_area_sum'].mean(), 
                 eval_table['pred_area_min'].mean(), eval_table['pred_area_max'].mean(), 
                 eval_table['pred_area_mean'].mean()]
            
    return res.T
PREDICTIONS_FOLDER = '../input/rsna-predictions/'
labels = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_1_train_labels.csv')
labels.head()
labels.shape
# path_to_file: LB_score for test data, associated with this train data
train_prediction_files = {
    '1_prediction_train.csv': .093,
    '2_prediction_train.csv': .106,
    '3_prediction_train.csv': .113,
    '4_prediction_train.csv': .119,
    '5_prediction_train.csv': .124,
    '6_prediction_train.csv': .129
}
train_current_prediction = '6_prediction_train.csv'
prediction_string = pd.read_csv(PREDICTIONS_FOLDER + train_current_prediction)
prediction_string.head()
prediction_string.shape
prediction = transform_prediction_to_table(prediction_string)
prediction.head()
prediction.shape
prediction['confidence'].min()
train_current_table = get_table_for_one_prediction(prediction, labels)
train_current_table.head()
train_current_table.describe()
plt.figure(figsize = (25, 5))
plt.subplot(121)
sns.boxplot(x = 'mask_iou', data = train_current_table);
plt.subplot(122)
sns.boxplot(x = 'map_iou', data = train_current_table);
_, axes = plt.subplots(1, 2, sharey = True, figsize = (25, 5))
sns.boxplot(x = 'true_target', y = 'pred_cnt', data = train_current_table, ax = axes[0]);
sns.boxplot(y = 'true_cnt', data = train_current_table[train_current_table['true_target'] == 1], ax = axes[1]);
_, axes = plt.subplots(1, 2, sharey = True, figsize = (25, 5))
sns.boxplot(x = 'true_target', y = 'pred_area_sum', data = train_current_table, ax = axes[0]);
sns.boxplot(y = 'true_area_sum', data = train_current_table[train_current_table['true_target'] == 1], ax = axes[1]);
_, axes = plt.subplots(1, 2, sharey = True, figsize = (25, 5))
sns.boxplot(x = 'true_target', y = 'pred_area_min', data = train_current_table, ax = axes[0]);
sns.boxplot(y = 'true_area_min', data = train_current_table[train_current_table['true_target'] == 1], ax = axes[1]);
_, axes = plt.subplots(1, 2, sharey = True, figsize = (25, 5))
sns.boxplot(x = 'true_target', y = 'pred_area_max', data = train_current_table, ax = axes[0]);
sns.boxplot(y = 'true_area_max', data = train_current_table[train_current_table['true_target'] == 1], ax = axes[1]);
_, axes = plt.subplots(1, 2, sharey = True, figsize = (25, 5))
sns.boxplot(x = 'true_target', y = 'pred_area_mean', data = train_current_table, ax = axes[0]);
sns.boxplot(y = 'true_area_mean', data = train_current_table[train_current_table['true_target'] == 1], ax = axes[1]);
all_train_predictions = get_table_for_all_predictions(train_prediction_files, labels)
all_train_predictions
all_train_predictions.to_csv('all_train_predictions.csv')
all_train_predictions[['LB_score_for_test', 'mean_map_iou', 'mean_mask_iou']].plot(figsize = (10, 5));
all_train_predictions.corr()
test_prediction_files = {
    '1_prediction_test.csv': .093,
    '2_prediction_test.csv': .106,
    '3_prediction_test.csv': .113,
    '4_prediction_test.csv': .119,
    '5_prediction_test.csv': .124,
    '6_prediction_test.csv': .129
}
test_current_prediction = '6_prediction_test.csv'
prediction_string = pd.read_csv(PREDICTIONS_FOLDER + test_current_prediction)
prediction_string.head()
prediction_string.shape
prediction = transform_prediction_to_table(prediction_string)
prediction.head()
prediction.shape
prediction['confidence'].min()
test_current_table = get_table_for_one_prediction(prediction)
test_current_table.head()
_, axes = plt.subplots(1, 3, sharey = True, figsize = (25, 5))
sns.boxplot(y = 'pred_cnt', data = test_current_table, ax = axes[0]);
sns.boxplot(y = 'pred_cnt', data = train_current_table, ax = axes[1]);
sns.boxplot(y = 'true_cnt', data = train_current_table, ax = axes[2]);
_, axes = plt.subplots(1, 3, sharey = True, figsize = (25, 5))
sns.boxplot(y = 'pred_area_sum', data = test_current_table[test_current_table['pred_target'] == 1], ax = axes[0]);
sns.boxplot(y = 'pred_area_sum', data = train_current_table[train_current_table['pred_target'] == 1], ax = axes[1]);
sns.boxplot(y = 'true_area_sum', data = train_current_table[train_current_table['true_target'] == 1], ax = axes[2]);
_, axes = plt.subplots(1, 3, sharey = True, figsize = (25, 5))
sns.boxplot(y = 'pred_area_min', data = test_current_table[test_current_table['pred_target'] == 1], ax = axes[0]);
sns.boxplot(y = 'pred_area_min', data = train_current_table[train_current_table['pred_target'] == 1], ax = axes[1]);
sns.boxplot(y = 'true_area_min', data = train_current_table[train_current_table['true_target'] == 1], ax = axes[2]);
_, axes = plt.subplots(1, 3, sharey = True, figsize = (25, 5))
sns.boxplot(y = 'pred_area_max', data = test_current_table[test_current_table['pred_target'] == 1], ax = axes[0]);
sns.boxplot(y = 'pred_area_max', data = train_current_table[train_current_table['pred_target'] == 1], ax = axes[1]);
sns.boxplot(y = 'true_area_max', data = train_current_table[train_current_table['true_target'] == 1], ax = axes[2]);
_, axes = plt.subplots(1, 3, sharey = True, figsize = (25, 5))
sns.boxplot(y = 'pred_area_mean', data = test_current_table[test_current_table['pred_target'] == 1], ax = axes[0]);
sns.boxplot(y = 'pred_area_mean', data = train_current_table[train_current_table['pred_target'] == 1], ax = axes[1]);
sns.boxplot(y = 'true_area_mean', data = train_current_table[train_current_table['true_target'] == 1], ax = axes[2]);
all_test_predictions = get_table_for_all_predictions(test_prediction_files)
all_test_predictions
all_test_predictions.to_csv('all_test_predictions.csv')
all_test_predictions.corr()