import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
INPUT_DIR = '/kaggle/input'
DATA_DIR = os.path.join(INPUT_DIR,"rsna-pneumonia-detection-challenge")
MODEL_RESULTS = os.path.join(INPUT_DIR,"predict-on-validation","validation_predictions.csv")
anns = pd.read_csv(os.path.join(DATA_DIR,'stage_1_train_labels.csv'))
anns.head()
anns.shape
# I need to group by patient id so that it is easy to join with predictions:

def gather_gt(patient_df):
    if np.sum(patient_df['x'].isna()) > 0:
        return []
    else:
        gts = []
        for index, row in patient_df.iterrows():
            gts.append({
                'x':row['x'],
                'y':row['y'],
                'width':row['width'],
                'height':row['height']
            })
        return gts

gt_patient = anns.groupby('patientId').apply(gather_gt)
gt_patient = gt_patient.to_frame("gt").reset_index()
gt_patient['Target'] = gt_patient['gt'].apply(lambda x: 1 * (x != []))
len(gt_patient)
results = pd.read_csv(MODEL_RESULTS, header=None, names=['patientId','prediction'])
results = results[~results['prediction'].isna()]
# we will join ground truth only for patients in the results df
df = results.merge(gt_patient, on='patientId', how='left')
print(df.shape)
df.head(n=10)
def filter_predictions(min_confidence):
    def mapper(prediction):
        exploded = prediction.strip().split(" ")
        predictions = [exploded[x:x+5] for x in range(0, len(exploded),5)]
        filtered_predictions = [p for p in predictions if float(p[0]) >= min_confidence]
        return " ".join([" ".join(p) for p in filtered_predictions])
    return mapper
def minimal_confidence_predictions(bbox_predictions, min_confidence):
    return bbox_predictions.apply(filter_predictions(min_confidence))

def rsna_precision(tp, fp, fn):
    return tp/(tp+fp+fn)

def rsna_precision_for(min_conf):
    # In my results there is no dedicated column for classification prediction -
    # I will be inferring it from bounding box prediction for various confidence thresholds
    y_pred = 1 * (minimal_confidence_predictions(df['prediction'], min_conf) != '')
    tn, fp, fn, tp = confusion_matrix(df['Target'], y_pred).ravel()
    return rs

def metrics_for_confidences():
    for min_conf in [x/100.0 for x in range(70,100)]:
        y_pred = 1 * (minimal_confidence_predictions(df['prediction'], min_conf) != '')
        tn, fp, fn, tp = confusion_matrix(df['Target'], y_pred).ravel()
        prec = rsna_precision(tp,fp,fn)
        cnt = tn + fp + fn + tp
        yield {"confidence":min_conf,"tn":tn/cnt,"fp":fp/cnt,"fn":fn/cnt,"tp":tp/cnt,"prec":prec}
metrics = pd.DataFrame(list(metrics_for_confidences()))
melted = metrics.melt(id_vars='confidence',value_vars=['tn','fp','fn','tp','prec'])

plt.figure(figsize=(20,10))

sns.lineplot(x='confidence', y='value', hue='variable',data=melted)
metrics.iloc[np.argmax(metrics['prec'])]
def to_structure(prediction):
    exploded = prediction.strip().split(" ")
    predictions = [exploded[x:x+5] for x in range(0, len(exploded),5)]    
    return [{'x':float(p[1]), 'y':float(p[2]), 'width': float(p[3]), 'height':float(p[4]), 'confidence':float(p[0])} for p in predictions]
        
df['all_predictions'] = df['prediction'].apply(to_structure)
# source: https://www.kaggle.com/chenyc15/mean-average-precision-metric

# extended version of metrics per patient giving more information:

iouthresholds = np.linspace(0.4,0.75,num=8)

# helper function to calculate IoU
def iou(box1, box2):
    x11, y11, w1, h1 = box1
    x21, y21, w2, h2 = box2
    assert w1 * h1 > 0
    assert w2 * h2 > 0
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2-xi1) * (yi2-yi1)
        union = area1 + area2 - intersect
        return intersect / union

def map_iou(boxes_true, boxes_pred, scores, thresholds = iouthresholds):
    """
    Mean average precision at differnet intersection over union (IoU) threshold
    
    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output: 
        map: mean average precision of the image
    """
    
    # According to the introduction, images with no ground truth bboxes will not be 
    # included in the map score unless there is a false positive detection (?)
    result= {
        'tp':0,
        'tn':0,
        'fp':0,
        'fn':0,
        'skipped':0,
        'predicted_cnt':len(boxes_pred),
        'gt_cnt':len(boxes_true),
        'ious':{}
    }    
    # return None if both are empty, don't count the image in final evaluation (?)
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        result['skipped'] = 1
        result['tn'] = 1
        return result
    if len(boxes_true) > 0 and len(boxes_pred) == 0:
        result['prec'] = 0
        result['fn'] = len(iouthresholds) * (len(boxes_true) - len(boxes_pred))
        return result
    if len(boxes_true) == 0 and len(boxes_pred) > 0:
        result['prec'] = 0
        result['fp'] = len(iouthresholds) * (len(boxes_pred) - len(boxes_true))
        return result
    
    assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    
    # I am not doing any sorting just assume that predictions are sorted according to confidence, since I cannot find a way to so
    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(-1 * scores, kind='mergesort'), :]
    
    map_total = 0
    
    # loop over thresholds
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou(bt, bp)
                result['ious'][(i,j)] = miou
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1 # bt is matched for the first time, count as TP
                    matched_bt.add(j)                    
            if not matched:
                fn += 1 # bt has no match, count as FN
                
        fp = len(boxes_pred) - len(matched_bt) # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        map_total += m
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    result['prec'] = map_total / len(thresholds)
    result['tp'] = total_tp
    result['fn'] = total_fn
    result['fp'] = total_fp
    
    return result

def bbox_to_array(bbox_dict):
    return [
                    bbox_dict['x'],
                    bbox_dict['y'],
                    bbox_dict['width'],
                    bbox_dict['height'],
                ]

def patient_metrics_off(row):
    gtboxes = np.array([bbox_to_array(b) for b in row['gt']])
    predboxes = np.array([bbox_to_array(b) for b in row['predictions']])
    confidences = np.array([b['confidence'] for b in row['predictions']])
    return map_iou(gtboxes,predboxes, confidences)
def filter_by_min_confidence(min_conf):
    def f(predictions):
        return [p for p in predictions if p['confidence'] > min_conf]
    return f
    
def metrics_for_confidences_bbox(df):
    for min_conf in [x/100.0 for x in range(70,100)]:
        df['predictions'] = df['all_predictions'].apply(filter_by_min_confidence(min_conf))
        patient_metrics_df = df.apply(patient_metrics_off, axis=1,  result_type='expand')
        tp = np.sum(patient_metrics_df.tp)
        tn = np.sum(patient_metrics_df.tn)
        fp = np.sum(patient_metrics_df.fp)
        fn = np.sum(patient_metrics_df.fn)
        not_skipped = patient_metrics_df[patient_metrics_df.skipped != 1]
        prec = np.mean(not_skipped.prec)        
        cnt = tn + fp + fn + tp
        yield {"confidence":min_conf,"tn":tn/cnt,"fp":fp/cnt,"fn":fn/cnt,"tp":tp/cnt,"prec":prec}
plt.figure(figsize=(20,10))

metrics_bbox = pd.DataFrame(list(metrics_for_confidences_bbox(df)))
melted_bbox = metrics_bbox.melt(id_vars='confidence',value_vars=['tn','fp','fn','tp','prec'])

sns.lineplot(x='confidence', y='value', hue='variable',data=melted_bbox)
metrics_bbox.iloc[np.argmax(metrics_bbox['prec'])]
df['predictions'] = df['all_predictions']
patient_metrics_df = df.join(df.apply(patient_metrics_off, axis=1,  result_type='expand'))
# source: https://www.kaggle.com/meaninglesslives/dataset-visualization-using-opencv
def overlay_box(im, box, rgb, stroke=1):
    """
    Method to overlay single box on image

    """
    # --- Convert coordinates to integers
    box = [int(b) for b in box]
    
    # --- Extract coordinates
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width

    im[y1:y1 + stroke, x1:x2] = rgb
    im[y2:y2 + stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    return im


import cv2
import pydicom
from IPython.display import display, Image


def cvshow(image, format='.png', rate=255 ):
    decoded_bytes = cv2.imencode(format, image*rate)[1].tobytes()
    display(Image(data=decoded_bytes))
    return

def visualize(df, patientId):
    dcm_file = '../input/rsna-pneumonia-detection-challenge/stage_1_train_images/%s.dcm' % patientId
    dcm_data = pydicom.read_file(dcm_file)
    img = dcm_data.pixel_array
    img = np.stack([img] * 3, axis=2)
    
    def show_boxes(img,boxes,rgb):
        for box in boxes:
            #y,x,h,w
            box = [box['y'],box['x'],box['height'],box['width']]                
            overlay_box(img, box, rgb)
            
    gt_boxes = df[df['patientId'] == patientId]['gt'].values[0]
    gt_col = np.array([1,250,1])  
    pred_boxes = df[df['patientId'] == patientId]['predictions'].values[0]    
    pred_col = np.array([1,1,250])  
    
    show_boxes(img,gt_boxes,gt_col)
    show_boxes(img,pred_boxes,pred_col)
    
    ious = df[df['patientId'] == patientId]['ious'].values[0]
    offsets = dict([(i,2) for i in range(len(pred_boxes))])
    for box in pred_boxes:
        confidence = box['confidence']        
        img = cv2.putText(img=np.copy(img), text="conf = {:.2f}".format(confidence), org=(int(box['x'])+5,int(box['y']) + 15),
                          fontFace=1, fontScale=1, color=(0,0,255), thickness=1)
    for k,iou in ious.items():     
        if iou > 0.0:            
            box = pred_boxes[k[1]]            
            img = cv2.putText(img=np.copy(img), text="iou = {:.2f}".format(iou), org=(int(box['x'])+5,int(box['y']) + 15 * offsets[k[1]]),
                          fontFace=1, fontScale=1, color=(0,0,255), thickness=1)
            offsets[k[1]]+=1
    prec = df[df['patientId'] == patientId]['prec'].values[0]
    img = cv2.putText(img=np.copy(img), text="precision = {:.2f}".format(prec), org=(10,300),
                          fontFace=1, fontScale=1, color=(0,0,255), thickness=1)
    cvshow(img)
patient_metrics_df_3 = patient_metrics_df[patient_metrics_df['gt_cnt'] == 3]
patient_metrics_df_2 = patient_metrics_df[patient_metrics_df['gt_cnt'] == 2]
patient_metrics_df_1 = patient_metrics_df[patient_metrics_df['gt_cnt'] == 1]
patient_metrics_df_0 = patient_metrics_df[patient_metrics_df['gt_cnt'] == 0]
fns = np.hstack([patient_metrics_df_3.sort_values('fn', ascending=False).patientId.values[:2],
patient_metrics_df_2.sort_values('fn', ascending=False).patientId.values[:2],
patient_metrics_df_1.sort_values('fn', ascending=False).patientId.values[:2]])

for pid in fns:
    visualize(patient_metrics_df, pid)
fps = np.hstack([patient_metrics_df_3.sort_values('fp', ascending=False).patientId.values[:2],
patient_metrics_df_2.sort_values('fp', ascending=False).patientId.values[:2],
patient_metrics_df_1.sort_values('fp', ascending=False).patientId.values[:2]])

for pid in fps:
    visualize(patient_metrics_df, pid)

best_prec = np.hstack([patient_metrics_df_3.sort_values('prec', ascending=False).patientId.values[:2],
patient_metrics_df_2.sort_values('prec', ascending=False).patientId.values[:2],
patient_metrics_df_1.sort_values('prec', ascending=False).patientId.values[:2]])

for pid in best_prec:
    visualize(patient_metrics_df, pid)
high_conf_df = df.copy()
high_conf_df['predictions'] = df['all_predictions'].apply(filter_by_min_confidence(0.98))
high_conf = high_conf_df.join(high_conf_df.apply(patient_metrics_off, axis=1, result_type='expand'))
high_conf_low_prec = high_conf.sort_values('fp', ascending=False).patientId.values[:6]
# high_conf.sort_values('fp', ascending=False).head()
for pid in high_conf_low_prec:
    visualize(high_conf, pid)
def area(bbox):
    return bbox['width'] * bbox['height']    
    
def min_bbox_size(bboxes):
    if len(bboxes)==0:
        return 0
    else:
        return np.min([area(bbox) for bbox in bboxes])

def max_bbox_size(bboxes):
    if len(bboxes)==0:
        return 0
    else:
        return np.max([area(bbox) for bbox in bboxes])

patient_metrics_df['min_pred_size'] = patient_metrics_df['all_predictions'].apply(min_bbox_size)
patient_metrics_df['max_pred_size'] = patient_metrics_df['all_predictions'].apply(max_bbox_size)
patient_metrics_df['min_gt_size'] = patient_metrics_df['gt'].apply(min_bbox_size)
patient_metrics_df['max_gt_size'] = patient_metrics_df['gt'].apply(max_bbox_size)
smallest_pred = patient_metrics_df[patient_metrics_df['min_pred_size'] > 0].sort_values('min_pred_size', ascending=True).patientId.values[:4]

for pid in smallest_pred:
    visualize(patient_metrics_df, pid)
smallest_gt = patient_metrics_df[patient_metrics_df['min_gt_size'] > 0].sort_values('min_gt_size', ascending=True).patientId.values[:4]

for pid in smallest_gt:
    visualize(patient_metrics_df, pid)
largest_pred = patient_metrics_df.sort_values('max_pred_size', ascending=False).patientId.values[:4]

for pid in largest_pred:
    visualize(patient_metrics_df, pid)

largest_gt = patient_metrics_df.sort_values('max_gt_size', ascending=False).patientId.values[:4]

for pid in largest_gt:
    visualize(patient_metrics_df, pid)
