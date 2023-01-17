%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skimage.io
import skimage.morphology

sns.set(font_scale=1.3, palette="colorblind", rc={"figure.figsize": (14,9) })
img_id = '0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9'
file = f"../input/stage1_train/{img_id}/images/{img_id}.png"
mfile = f"../input/stage1_train/{img_id}/masks/*.png"
image = skimage.io.imread(file)
masks = skimage.io.imread_collection(mfile).concatenate()
height, width, _ = image.shape

y_true = (masks == 255).astype(np.uint8)

# Remove item 20
y_preds = np.concatenate((y_true[:19], y_true[20:]), axis=0)
y_pred = np.empty_like(y_preds)

# Reduce areas
for i, y_p in enumerate(y_preds):
    skimage.morphology.binary_erosion(y_p, out=y_pred[i])

# Label areas
num_true = y_true.shape[0]
num_pred = y_pred.shape[0]
np.multiply(y_true, 
            np.arange(1, num_true+1, dtype=np.uint8)[:, np.newaxis, np.newaxis], 
            out=y_true)

np.multiply(y_pred, 
            np.arange(1, num_pred+1, dtype=np.uint8)[:, np.newaxis, np.newaxis], 
            out=y_pred)

# Combine labels into one image
y_true = np.max(y_true, axis=0)
y_pred = np.max(y_pred, axis=0)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(y_true)
ax1.set_title("True Labels")
ax1.grid(False)
ax2.imshow(y_pred)
ax2.set_title("Simulated Predictions")
ax2.grid(False);
def ap(y_true, y_pred):
    # remove one for background
    num_true = len(np.unique(y_true)) - 1
    num_pred = len(np.unique(y_pred)) - 1
    
    if num_true == 0 and num_pred == 0:
        return 1
    elif num_true == 0 or num_pred == 0:
        return 0
    
    # bin size + 1 for background
    intersect = np.histogram2d(
        y_true.flatten(), y_pred.flatten(), bins=(num_true+1, num_pred+1))[0]
    
    area_t = np.histogram(y_true, bins=(num_true+1))[0][:, np.newaxis]
    area_p = np.histogram(y_pred, bins=(num_pred+1))[0][np.newaxis, :]
    
    # get rid of background
    union = area_t + area_p - intersect
    intersect = intersect[1:, 1:]
    union = union[1:, 1:]
    iou = intersect / union 
    
    threshold = np.arange(0.5, 1.0, 0.05)[np.newaxis, np.newaxis, :]
    matches = iou[:,:, np.newaxis] > threshold
    
    tp = np.sum(matches, axis=(0,1))
    fp = num_true - tp
    fn = num_pred - tp
    
    return np.mean(tp/(tp+fp+fn))
ap(y_true, y_true)
ap(y_true, y_pred)
def ap_wc(y_true_in, y_pred_in):
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        prec.append(p)

    return np.mean(prec)
ap_wc(y_true, y_pred)
%%timeit
ap_wc(y_true, y_pred)
%%timeit
ap(y_true, y_pred)