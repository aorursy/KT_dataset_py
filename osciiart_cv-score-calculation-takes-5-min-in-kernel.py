# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# prepare functions for calculation

import skimage.io
import skimage.segmentation
from skimage.transform import resize
import os
from multiprocessing import Pool


def load_y_true(id, TRAIN_PATH):
    file = TRAIN_PATH + "{}/images/{}.png".format(id, id)
    mfile = TRAIN_PATH + "{}/masks/*.png".format(id)

    image = skimage.io.imread(file)

    masks = skimage.io.imread_collection(mfile).concatenate()
    height, width, _ = image.shape
    num_masks = masks.shape[0]
    y_true = np.zeros((num_masks, height, width), np.bool)
    y_true[:, :, :] = masks[:, :, :] // 255  # Change ground truth mask to zeros and ones
    return y_true


def load_y_trues(TRAIN_PATH='input/stage1_train/', ng=[]):
    train_ids = sorted(next(os.walk(TRAIN_PATH))[1])
    for item in ng:
        train_ids.remove(item)
    y_trues = []
    for i in range(len(train_ids)):
        id = train_ids[i]
        y_trues.append(load_y_true(id, TRAIN_PATH))
    return y_trues


def IOU(y_pred, y_true):
    """
    calcurate IOU of 1 image and its prediction
    :param y_pred: numpy array. shape (mask, height, width)
    :param y_true:
    :return: list of scores. len = number of true masks
    """
    num_true = len(y_true)
    num_pred = len(y_pred)
    iou = []
    for pr in range(num_pred):
        bol = 0  # best overlap
        bun = 1e-9  # corresponding best union
        for tr in range(num_true):
            olap = y_pred[pr] * y_true[tr]  # Intersection points
            osz = np.sum(olap)  # Add the intersection points to see size of overlap
            if osz > bol:  # Choose the match with the biggest overlap
                bol = osz
                bun = np.sum(np.maximum(y_pred[pr], y_true[tr]))  # Union formed with sum of maxima
        iou.append(bol / bun)
    return iou

def mAP2(args):
    """
    return np array with 43 values.
    [0:2] = number of prediction masks, number of true masks
    [1:6] = TP, FP, FN, Prec. with threshold 0.50
    [6:10] = TP, FP, FN, Prec. with threshold 0.55
    ...
    [38:42] = TP, FP, FN, Prec. with threshold 0.95
    [42] = mean Prec.
    :param y_pred:
    :param y_true:
    :return: np array shape = 41,
    """
    y_pred, train_id, train_path  = args
    y_true = load_y_true(train_id, train_path)
    num_true = len(y_true)
    num_pred = len(y_pred)
    # print(y_pred.dtype, y_true.dtype)
    iou = IOU(y_pred, y_true)
    output = np.zeros(43, np.float64)
    p_all = 0
    thresholds = np.arange(0.5, 1.0, 0.05)
    for i in range(thresholds.shape[0]):
        t = thresholds[i]
        matches = iou > t
        tp = np.count_nonzero(matches)  # True positives
        fp = num_pred - tp  # False positives
        fn = num_true - tp  # False negatives
        p = tp / (tp + fp + fn)
        p_all += p
        output[i*4+2:(i+1)*4+2] = np.array([tp, fp, fn, p])
    output[0] = num_pred
    output[1] = num_true
    output[42] = p_all/10
    return output

def mAP(args):
    """
    return np array with 43 values.
    [0:2] = number of prediction masks, number of true masks
    [1:6] = TP, FP, FN, Prec. with threshold 0.50
    [6:10] = TP, FP, FN, Prec. with threshold 0.55
    ...
    [38:42] = TP, FP, FN, Prec. with threshold 0.95
    [42] = mean Prec.
    :param y_pred:
    :param y_true:
    :return: np array shape = 41,
    """
    y_pred, y_true = args
    num_true = len(y_true)
    num_pred = len(y_pred)
    # print(y_pred.dtype, y_true.dtype)
    iou = IOU(y_pred, y_true)
    output = np.zeros(43, np.float64)
    p_all = 0
    thresholds = np.arange(0.5, 1.0, 0.05)
    for i in range(thresholds.shape[0]):
        t = thresholds[i]
        matches = iou > t
        tp = np.count_nonzero(matches)  # True positives
        fp = num_pred - tp  # False positives
        fn = num_true - tp  # False negatives
        p = tp / (tp + fp + fn)
        p_all += p
        output[i*4+2:(i+1)*4+2] = np.array([tp, fp, fn, p])
    output[0] = num_pred
    output[1] = num_true
    output[42] = p_all/10
    return output


def valid_score(y_preds, y_trues):
    """
    calculate a validation IOU score of predction and some related values
    :param y_preds: list of np array. len = number of images.
                    each np array's shape = (number of nuclei, height, width)
                    each height and width must be same with its original input image
    :return: pd.Dataframe with shape = (number of images, 43)
    """
    pool = Pool(1) # 16 107sec
    
    list_mAP = pool.map(mAP,
                       [(y_preds[i], y_trues[i])
                        for i in range(len(y_preds))])
    pool.terminate()
    scores = np.array(list_mAP)
    cols = ['pred_masks', 'true_masks',
            'TP_0.50', 'FP_0.50', 'FN_0.50', 'Prec_0.50',
            'TP_0.55', 'FP_0.55', 'FN_0.55', 'Prec_0.55',
            'TP_0.60', 'FP_0.60', 'FN_0.60', 'Prec_0.60',
            'TP_0.65', 'FP_0.65', 'FN_0.65', 'Prec_0.65',
            'TP_0.70', 'FP_0.70', 'FN_0.70', 'Prec_0.70',
            'TP_0.75', 'FP_0.75', 'FN_0.75', 'Prec_0.75',
            'TP_0.80', 'FP_0.80', 'FN_0.80', 'Prec_0.80',
            'TP_0.85', 'FP_0.85', 'FN_0.85', 'Prec_0.85',
            'TP_0.90', 'FP_0.90', 'FN_0.90', 'Prec_0.90',
            'TP_0.95', 'FP_0.95', 'FN_0.95', 'Prec_0.95',
            'mAP',
            ]
    scores = pd.DataFrame(scores, columns=cols)
    return scores


def valid_score2(y_preds, train_ids, train_path):
    """
    calculate a validation IOU score of predction and some related values
    :param y_preds: list of np array. len = number of images.
                    each np array's shape = (number of nuclei, height, width)
                    each height and width must be same with its original input image
    :return: pd.Dataframe with shape = (number of images, 43)
    """
    pool = Pool() # 16 107sec
    
    list_mAP = pool.map(mAP2,
                       [(y_preds[i], train_ids[i], train_path)
                        for i in range(len(y_preds))])
    pool.terminate()
    scores = np.array(list_mAP)
    cols = ['pred_masks', 'true_masks',
            'TP_0.50', 'FP_0.50', 'FN_0.50', 'Prec_0.50',
            'TP_0.55', 'FP_0.55', 'FN_0.55', 'Prec_0.55',
            'TP_0.60', 'FP_0.60', 'FN_0.60', 'Prec_0.60',
            'TP_0.65', 'FP_0.65', 'FN_0.65', 'Prec_0.65',
            'TP_0.70', 'FP_0.70', 'FN_0.70', 'Prec_0.70',
            'TP_0.75', 'FP_0.75', 'FN_0.75', 'Prec_0.75',
            'TP_0.80', 'FP_0.80', 'FN_0.80', 'Prec_0.80',
            'TP_0.85', 'FP_0.85', 'FN_0.85', 'Prec_0.85',
            'TP_0.90', 'FP_0.90', 'FN_0.90', 'Prec_0.90',
            'TP_0.95', 'FP_0.95', 'FN_0.95', 'Prec_0.95',
            'mAP',
            ]
    scores = pd.DataFrame(scores, columns=cols)
    return scores
# load sample cv prediction
from skimage.morphology import label
import time

start_time = time.time()

y_preds = np.load("../input/y-preds-samplenpy/y_preds_sample.npy")
print("y pred loading done. {:1.3f} sec".format(time.time() - start_time))
# resize prediction
# ignore bad train data
ng = ["12aeefb1b522b283819b12e4cfaf6b13c1264c0aadac3412b4edd2ace304cb40",
      "7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80",]
sizes_train = []
TRAIN_PATH = '../input/data-science-bowl-2018/stage1_train/'
train_ids = sorted(next(os.walk(TRAIN_PATH))[1])
for item in ng:
    train_ids.remove(item)

for i in range(len(train_ids)):
    id = train_ids[i]
    file = TRAIN_PATH + "{}/images/{}.png".format(id,id)
    image = skimage.io.imread(file)
    sizes_train.append((image.shape[0], image.shape[1]))

y_preds_resized = []
for i in range(len(y_preds)):
    y_preds_resized.append(resize(np.squeeze(y_preds[i]),
                           (sizes_train[i][0], sizes_train[i][1]),
                           mode='constant', preserve_range=True))

print("y pred resizing done. {:1.3f} sec".format(time.time() - start_time))
# split mask
y_preds_labeled = []
CUTOFF = 0.5
for i in range(len(y_preds)):
    y_label_base = label(y_preds_resized[i] > CUTOFF)
    y_pred_labeled = np.zeros([np.max(y_label_base) + 1, y_label_base.shape[0], y_label_base.shape[1]], dtype=np.bool)
    for k in range(y_pred_labeled.shape[0]):
        y_pred_labeled[k] = y_label_base == k
    y_preds_labeled.append(y_pred_labeled)
   
print("y pred mask splitting done. {:1.3f} sec".format(time.time() - start_time))
# load y true

# scoring

# faster but requires more RAM
# y_trues = load_y_true(TRAIN_PATH=TRAIN_PATH, ng=ng)
# score = valid_score(y_preds_labeled, y_trues)

score = valid_score2(y_preds_labeled, train_ids, TRAIN_PATH)

print("valid score", np.mean(score['mAP']))
print("scoring done.", time.time() - start_time)