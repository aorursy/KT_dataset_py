import numpy as np
import tensorflow as tf
from skimage import measure
import keras.backend as K
# helper function to calculate IoU
def iou_bbox(box1, box2):
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
    
# simple test
box1 = [100, 100, 200, 200]
box2 = [100, 100, 300, 200]
print(iou_bbox(box1, box2))
def map_iou(boxes_true, boxes_pred, thresholds=(0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75)):
    """
    Mean average precision at differnet intersection over union (IoU) threshold

    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image.
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image.
                    bbox format: (x1, y1, w, h)
        thresholds: IoU shresholds to evaluate mean average precision on
    output:
        map: mean average precision of the image
    """

    # According to the introduction, images with no ground truth bboxes will not be
    # included in the map score unless there is a false positive detection (?)

    # return 0 if both are empty, don't count the image in final evaluation (?)
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return 0

    map_total = 0

    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou_bbox(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1 # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1 # bt has no match, count as FN

        fp = len(boxes_pred) - len(matched_bt) # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        map_total += m

    return map_total / len(thresholds)
def unet_mask_to_bbox_coords(mask, threshold=0.5, do_resize=False):
    '''
    :param mask: predicted mask, numpy array of shape (widht, height, 1) or (width, height)
    :param threshold: threshold for binarization of mask
    :return: bbox coordinates, in form [x, y, width, height] for each bbox coordinate
    :rtype: numpy array
    '''
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    # resize predicted mask
    if do_resize:
        mask = resize(mask, (1024, 1024), mode='constant')
    # threshold predicted mask, multiply by 255, since predictions were upscaled for memory performance
    comp = mask > threshold
    # apply connected components
    comp = measure.label(comp)

    bboxes = np.array([]).reshape((0, 4))

    for region in measure.regionprops(comp):
        # retrieve x, y, height and width
        y, x, y2, x2 = region.bbox
        height = y2 - y
        width = x2 - x
        bboxes = np.concatenate([bboxes, np.array([[x, y, width, height]])], axis=0)

    return bboxes
def competitionMetric(y_true, y_pred):
    '''
    Implementation of rsna pneumonia competition metric
    '''
    def np_competitionMetric(np_true, np_pred):
        '''
        Compute the mean map_iou for each sample of the batch
        '''
        return np.mean([map_iou(unet_mask_to_bbox_coords(true), unet_mask_to_bbox_coords(pred))
                               for true, pred in zip(np_true, np_pred)]).astype(np.float32)

    return tf.py_func(np_competitionMetric,
                      inp=[y_true, y_pred],
                      Tout=tf.float32,
                      stateful=False,
                      name='competitionMetric'
                      )
y_true_array = np.zeros((1, 128, 128, 1))
y_true_array[:, 20: 41, 23:45, :] = 1
y_true = tf.Variable(y_true_array, dtype='float32', name='y_true')

y_pred_array = np.zeros((1, 128, 128, 1))
y_pred_array[:, 20: 37, 18:37, :] = 1
y_pred = tf.Variable(y_pred_array, dtype='float32', name='y_pred')
sess = K.get_session()
sess.run(tf.global_variables_initializer())
sess.run(competitionMetric(y_true, y_pred))
box_true = [[20, 23, 21, 22]] #x, y, width, height
box_pred = [[20, 18, 17, 19]]
map_iou(box_true, box_pred)
sess = K.get_session()

y_true_array = np.zeros((24, 128, 128, 1))
y_true_array[:, 20: 35, 10:40, :] = 1
y_true = tf.Variable(y_true_array, dtype='float32', name='y_true')

y_pred_array = np.zeros((24, 128, 128, 1))
y_pred_array[5:, 20: 37, 19:37, :] = 1
y_pred_array[:10, 100:115, 105:115, :] = 1
y_pred = tf.Variable(y_pred_array, dtype='float32', name='y_pred')
sess.run(tf.global_variables_initializer())
sess.run(competitionMetric(y_true, y_pred))
batch_box_true = [ [[20, 10, 15, 30]] for _ in range(24)]
batch_box_pred = [ [[100, 105, 15, 10]] for _ in range(5)] +\
                 [ [[100, 105, 15, 10], [20, 19, 17, 18]] for _ in range(5)] +\
                 [ [[20, 19, 17, 18]] for _ in range(14)]
np.mean([map_iou(box_true, box_pred) for box_true, box_pred in zip(batch_box_true, batch_box_pred)])