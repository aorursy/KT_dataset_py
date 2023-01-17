import numpy as np
import pandas as pd

from random import randint

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from sklearn.model_selection import train_test_split

from skimage.transform import resize

from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout

from functools import partial
import keras.backend as K

from tqdm import tqdm_notebook
import os
print(os.listdir("../input"))
# train_path = "../input/train.csv"
# depths_path = "../input/depths.csv"

# images_path = "../input/train/images/{}.png"
# masks_path = "../input/train/masks/{}.png"
# test_path = "../input/test/images/{}.png"

print(os.listdir("../input"))

train_path = "../input/tgs-salt-identification-challenge/train.csv"
depths_path = "../input/tgs-salt-identification-challenge/depths.csv"

images_path = "../input/tgs-salt-identification-challenge/train/images/{}.png"
masks_path = "../input/tgs-salt-identification-challenge/train/masks/{}.png"
test_path = "../input/tgs-salt-identification-challenge/test/images/{}.png"

image_1 = "0b73b427d1"
image_2 = "0c02f95a08"

img_size_target = 128
img_size_original = 101
img_channels = 1

test_size = 0.2
train_size = 4000
random_state = 1337
def _resize(img, target_size):
    img_size = img.size
    if img_size == target_size:
        return img
    return resize(img, (target_size, target_size), mode='constant', preserve_range=True)

def upsample(img, target_size=img_size_target):
    return _resize(img, target_size)
    
def downsample(img, target_size=img_size_original):
    return _resize(img, target_size)

def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i
train_df = pd.read_csv(train_path, index_col="id", usecols=[0])
depths_df = pd.read_csv(depths_path, index_col="id")

train_df = train_df.join(depths_df)

test_df = depths_df[~depths_df.index.isin(train_df.index)]

train_df["images"] = [
    np.array(load_img(images_path.format(idx), grayscale=True)) / 255 
    for idx in tqdm_notebook(train_df.index)
]

train_df["masks"] = [
    np.array(load_img(masks_path.format(idx), grayscale=True)) / 255 
    for idx in tqdm_notebook(train_df.index)
]

train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_original, 2)        
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

x_test = np.array([
    upsample(np.array(load_img(test_path.format(idx), grayscale=True))) / 255 
    for idx in tqdm_notebook(test_df.index)
]).reshape(-1, img_size_target, img_size_target, 1)
id_train = train_df.index.values
x_train = np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
y_train = np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
coverage = train_df.coverage.values
coverage_class = train_df.coverage_class.values
depths = train_df.z.values

ID_train, ID_valid, X_train, X_valid, Y_train, Y_valid, C_train, C_valid, CC_train, CC_valid, Z_train, Z_valid = train_test_split(
    id_train,
    x_train,
    y_train,
    coverage,
    coverage_class,
    depths,
    test_size=test_size,
    stratify=train_df.coverage_class,
    random_state=random_state
)
model_path = "../input/u-net-dropout-augmentation-stratification/keras.model"

model = load_model(model_path)
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return K.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

model_path = "../input/u-net-bn-aug-strat-dice/keras.model"

clf = partial(bce_dice_loss)
clf.__name__ = "bce_dice_loss"

model = load_model(model_path, custom_objects={'bce_dice_loss': clf})
P_valid = model.predict(X_valid).reshape(-1, img_size_target, img_size_target)
P_valid = np.array([downsample(x) for x in P_valid])
y_valid_ori = np.array([train_df.loc[idx].masks for idx in ID_valid])
# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

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
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)
thresholds = np.linspace(0, 1, 50)
ious = np.array([iou_metric_batch(y_valid_ori, np.int32(P_valid > threshold)) for threshold in tqdm_notebook(thresholds)])

threshold_best_index = np.argmax(ious[9:-10]) + 9
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

PA_valid = np.array([
    np.round(pred > threshold_best) 
    for pred in tqdm_notebook(P_valid)
], dtype=np.float32)
plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
plt.legend()
images_valid = [downsample(X_valid.reshape(-1,128,128)[i]) for i in range(len(X_valid))]
masks_valid = [downsample(Y_valid.reshape(-1,128,128)[i]) for i in range(len(Y_valid))]

preds_valid = [P_valid[i] for i in range(len(P_valid))]
preds_adjusted_valid = [PA_valid[i] for i in range(len(PA_valid))]

coverage_valid = C_valid
coverage_class_valid = CC_valid
valid_df = pd.DataFrame({
    "images" : images_valid,
    "masks" : masks_valid,
    "depths" : Z_valid,
    "coverages" : coverage_valid,
    "coverage_classes" : coverage_class_valid,
    "predictions": preds_valid,
    "predictions_with_threshold": preds_adjusted_valid, 
}, index=ID_valid)
class_count = valid_df.groupby("coverage_classes").agg("count")["coverages"]

def jaccard_index(row):
    a_n_b = np.sum(row['predictions_with_threshold'] == row['masks'])
    index = float(a_n_b) / float((img_size_original ** 2) * 2 - a_n_b)
    return index

valid_df["jaccard_index"] = valid_df.apply(jaccard_index, axis=1)

jaccard_index_df = valid_df.groupby("coverage_classes").agg(["mean", "count"])["jaccard_index"]
jaccard_index_df
number_of_images_per_class = 12
number_of_classes = 10
df = valid_df.reset_index().set_index("coverage_classes").join(jaccard_index_df).reset_index().set_index("index")

number_of_images_per_data = 3
number_of_images = number_of_images_per_class * number_of_classes * number_of_images_per_data
grid_width = 12
grid_height = number_of_images // grid_width

fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width * 2, grid_height * 2))
img_count = 0
        
for coverage_class in jaccard_index_df.sort_values("mean").index[:number_of_classes]:
    temp = df[df["coverage_classes"] == coverage_class].head(number_of_images_per_class)
    for i in range(number_of_images_per_class):
        data = temp.iloc[i]
        
        img = data.images
        mask = data.masks
        pred = data.predictions_with_threshold
        depth = data.depths
        coverage = round(data.coverages, 2)
        
        # image
        ax = axs[img_count // grid_width, img_count % grid_width]
        img_count += 1
        ax.imshow(img, cmap="Greys")
        # mask
        ax = axs[img_count // grid_width, img_count % grid_width]
        img_count += 1
        ax.imshow(mask, alpha=0.3, cmap="Greens")
        # prediction
        ax = axs[img_count // grid_width, img_count % grid_width]
        img_count += 1
        ax.imshow(pred, alpha=0.3, cmap="OrRd")
        
        # details
        ax.text(1, img_size_original - 1, depth, color="black")
        ax.text(img_size_original - 1, 1, coverage, color="black", ha="right", va="top")
        ax.text(1, 1,                     coverage_class, color="black", ha="left", va="top")

plt.suptitle("Green: salt, Red: prediction. Top-left: coverage class, top-right: salt coverage, bottom-left: depth")