from itertools import chain

from pathlib import Path

import random



import numpy as np

import pandas as pd





# Data path

DATA_DIR = Path('../input')
train_df = pd.read_csv(DATA_DIR / 'train.csv')

train_df['CategoryId'] = train_df['ClassId'].str.split('_').str[0]  # 10(classId)_1_2_3(attributeIds) => 10
# Ground truth

gt_columns = ['ImageId', 'EncodedPixels', 'Height', 'Width', 'CategoryId']

gt_df = train_df.sample(5000, random_state=777)[gt_columns]

gt_df.head()
# Prediction

pred_columns = ['ImageId', 'EncodedPixels', 'CategoryId']

pred_df_ = train_df.sample(3000, random_state=111)[pred_columns]



# Prediction uses ground truth data partialy for simulation.

# The prediction has the perfect masks and classIDs of the ground truth data at least 2000.

pred_df = pd.concat([gt_df.iloc[:2000][pred_columns], pred_df_], axis=0, sort=False)

pred_df.head()
print(f"Ground truth: {len(gt_df)}")

print(f"Predictions: {len(pred_df)}")
def drop_randomly(pixels):

    pixels_ = pixels.split()

    split_pixels = np.split(np.array(pixels_), len(pixels_)/2)



    # Drop pixels

    random.seed(7)

    remains = int(random.choice(np.arange(0.5, 1.1, 0.1)) * len(split_pixels))

    drop_pixels = random.sample(split_pixels, remains)



    # Fluctuate pixel length

    def fluc_pixel(arr, f):

        return np.array([arr[0], max(1, int(arr[1]) + f)])



    random.seed(7)

    fluc = np.random.randint(-10, 10, len(drop_pixels))

    dp_ = [fluc_pixel(arr, f) for arr, f in zip(drop_pixels, fluc)]



    dp = list(chain.from_iterable([dp.tolist() for dp in dp_]))



    return ' '.join(dp)



pred_df_pseudo = pred_df.copy()

pred_df_pseudo['EncodedPixels'] = pred_df['EncodedPixels'].apply(drop_randomly)

pred_df_pseudo.head()
def calc_IoU(A,B):

    AorB = np.logical_or(A,B).astype('int')

    AandB = np.logical_and(A,B).astype('int')

    IoU = AandB.sum() / AorB.sum()

    return IoU



def rle_to_mask(rle_list, SHAPE):

    tmp_flat = np.zeros(SHAPE[0]*SHAPE[1])

    if len(rle_list) == 1:

        mask = np.reshape(tmp_flat, SHAPE).T

    else:

        strt = rle_list[::2]

        length = rle_list[1::2]

        for i,v in zip(strt,length):

            tmp_flat[(int(i)-1):(int(i)-1)+int(v)] = 255

        mask = np.reshape(tmp_flat, SHAPE).T

    return mask



def calc_IoU_threshold(data):

    # Note: This rle_to_mask should be called before loop below for speed-up! We currently implement here to reduse memory usage.

    mask_gt = rle_to_mask(data['EncodedPixels_gt'].split(), (int(data['Height']), int(data['Width'])))

    mask_pred = rle_to_mask(data['EncodedPixels_pred'].split(), (int(data['Height']), int(data['Width'])))

    return calc_IoU(mask_gt, mask_pred)



def evaluation(gt_df, pred_df):

    eval_df = pd.merge(gt_df, pred_df, how='outer', on=['ImageId', 'CategoryId'], suffixes=['_gt', '_pred'])



    # IoU for True Positive

    idx_ = eval_df['EncodedPixels_gt'].notnull() & eval_df['EncodedPixels_pred'].notnull()

    IoU = eval_df[idx_].apply(calc_IoU_threshold, axis=1)



    # False Positive

    fp = (eval_df['EncodedPixels_gt'].isnull() & eval_df['EncodedPixels_pred'].notnull()).sum()



    # False Negative

    fn = (eval_df['EncodedPixels_gt'].notnull() & eval_df['EncodedPixels_pred'].isnull()).sum()



    threshold_IoU = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    scores = []

    for th in threshold_IoU:

        # True Positive

        tp = (IoU > th).sum()

        maybe_fp = (IoU <= th).sum()



        # False Positive (not Ground Truth) + False Positive (under IoU threshold)

        fp_IoU = fp + maybe_fp



        # Calculate evaluation score

        score = tp / (tp + fp_IoU + fn)

        scores.append(score)

        print(f"Threshold: {th}, Precision: {score}, TP: {tp}, FP: {fp_IoU}, FN: {fn}")



    mean_score = sum(scores) / len(threshold_IoU)

    print(f"Mean precision score: {mean_score}")
evaluation(gt_df, pred_df)
evaluation(gt_df, pred_df_pseudo)