# Import Necessary Libraries
import warnings
warnings.filterwarnings('ignore')
import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random
import tensorflow as tf
import cv2
import sklearn

tf.compat.v1.disable_eager_execution()

from tqdm import tqdm
from skimage import io
from itertools import chain
from glob import glob
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, CSVLogger
from tensorflow.keras.metrics import AUC, Precision, Recall
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

%matplotlib inline
random.seed(2020)
# Define Path to the Chest-Xray Data Directory
DATA_DIR = 'G:/Downloads/archive/chest_xray'

# Training Data Directory
TRAIN_DIR = os.path.join(DATA_DIR,'train')

# Validation Data Directory
VALID_DIR = os.path.join(DATA_DIR,'val')

# Test Data Directory
TEST_DIR = os.path.join(DATA_DIR,'test')
# list of normal and pneumonia images

#train set
train_normal = os.listdir(TRAIN_DIR+'/NORMAL')
train_pneumonia = os.listdir(TRAIN_DIR+'/PNEUMONIA')

# validation set
val_normal = os.listdir(VALID_DIR+'/NORMAL')
val_pneumonia = os.listdir(VALID_DIR+'/PNEUMONIA')

# test set
test_normal = os.listdir(TEST_DIR+'/NORMAL')
test_pneumonia = os.listdir(TEST_DIR+'/PNEUMONIA')
# print counts of normal and pneumonia images
# train set
print(f"Number of Pneumonia cases in train set: {len(train_pneumonia)}")
print(f"Number of Normal cases in train set: {len(train_normal)}")
print()

# validation set
print(f"Number of Pneumonia cases in valid set: {len(val_pneumonia)}")
print(f"Number of Normal cases in valid set: {len(val_normal)}")
print()

# test set
print(f"Number of Pneumonia cases in test set: {len(test_pneumonia)}")
print(f"Number of Normal cases in test set: {len(test_normal)}")
labels = ['train', 'valid', 'test']
pneumonia = [len(train_pneumonia), len(val_pneumonia), len(test_pneumonia)]
normal = [len(train_normal), len(val_normal), len(test_normal)]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, pneumonia, width, label='Pneumonia')
rects2 = ax.bar(x + width/2, normal, width, label='Normal')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Count')
ax.set_title('Frequency of each dataset')
ax.set_xticks(x)
ax.set_ylim([0, 4200])
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height() 
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()
# Normal X-Ray images
num_samples = 3
train_normal_samples = random.sample(train_normal, num_samples)

fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
for path, ax in zip(train_normal_samples, axes):
    img = plt.imread(TRAIN_DIR+'/NORMAL/'+path)
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_aspect('auto')
    ax.title.set_text('NORMAL')
plt.show()
# Pneumonia X-Ray images
train_pneumonia_samples = random.sample(train_pneumonia, num_samples)

fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
for path, ax in zip(train_pneumonia_samples, axes):
    img = plt.imread(TRAIN_DIR+'/PNEUMONIA/'+path)
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_aspect('auto')
    ax.title.set_text('PNEUMONIA')
plt.show()
CLASS_NAMES = np.array([class_name.split(os.path.sep)[-1] for class_name in glob(os.path.join(TRAIN_DIR,"*"))])
CLASS_NAMES
# Pixel Intensity Distribution of Pneumonia Positive Cases
train_pneumonia_images = random.sample(train_pneumonia,100)
pneumonia_pixel = []
for img in train_pneumonia_images:
    img = io.imread(os.path.join(TRAIN_DIR,"PNEUMONIA",img)).ravel()
    # Remove Background Noise
    img_mask = [True if 35 < pixel < 255 else False for pixel in img]
    pneumonia_pixel.extend(img[img_mask])
print(f"Mean: {round(np.mean(pneumonia_pixel), 2)}, STD: {round(np.std(pneumonia_pixel), 2)}")
pneumonia_dist = plt.hist(pneumonia_pixel, bins=256);
# Pixel Intensity Distribution of Pneumonia Negative Cases
train_non_pneumonia_images = random.sample(train_normal,100)
non_pneumonia_pixel = []
for img in train_non_pneumonia_images:
    img = io.imread(os.path.join(TRAIN_DIR,"NORMAL",img)).ravel()
    # Remove Background Noise
    img_mask = [True if 35 < pixel < 255 else False for pixel in img]
    non_pneumonia_pixel.extend(img[img_mask])
print(f"Mean: {round(np.mean(non_pneumonia_pixel), 2)}, STD: {round(np.std(non_pneumonia_pixel), 2)}")
non_pneumonia_dist = plt.hist(non_pneumonia_pixel, bins=256);
plt.savefig(f'Images/pneumonia_dist.png')
plt.savefig(f'Images/non_pneumonia_dist.png')
# Investigate a Single Image
# Get the First Pneumonia Image that was in the DataFrame
sample_img = os.path.join(TRAIN_DIR,'PNEUMONIA',train_pneumonia[0])
raw_image = plt.imread(sample_img)
plt.imshow(raw_image, cmap='gray')
plt.colorbar();
plt.title('Raw Pneumonia Positive Chest X-Ray');
print(f"The Dimension of the Image are {raw_image.shape[0]} pixels width and {raw_image.shape[1]} pixels height, one single color channel.")
print(f"The Maximum pixel value is {raw_image.max():.4f} and the minimum is {raw_image.min():.4f}")
print(f"The Mean Value of the Pixels is {raw_image.mean():.4f} and the Standard Deviation is {raw_image.std():.4f}")
# Plot a Histogram of the distribution
sns.set()
sns.distplot(raw_image.ravel(), label=f"Pixel Mean {np.mean(raw_image):.4f} & Standard Deviation {np.std(raw_image):.4f}", kde=False)
plt.legend(loc='upper center')
plt.title('Distribution of Pixels Intensities in the Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('# Pixels in Image');
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 30
VALIDATION_PCT = 0.20
def get_label(file_path):
    return 1 if file_path.split(os.path.sep)[-2] == "PNEUMONIA" else 0
filenames = glob(os.path.join(TRAIN_DIR, "*", "*"))
filenames.extend(glob(os.path.join(VALID_DIR, "*", "*")))
all_data = []
for path in filenames:
    label = get_label(path)
    all_data.append((path, label))

all_data = pd.DataFrame(all_data, columns=['Path','Pneumonia'], index=None)    
all_data = all_data.sample(frac=1.).reset_index(drop=True)
# 80-20 split into train and validation set
train_set, val_set = train_test_split(all_data, test_size=0.20, stratify=all_data['Pneumonia'])
# Number of images in train set
train_post_count = train_set['Pneumonia'].value_counts()
print(train_post_count)
# Number of images in validation set
val_post_count = val_set['Pneumonia'].value_counts()
print(val_post_count)
print(f'Total Pneumonia Cases: {all_data[all_data.Pneumonia==1].shape[0]}')
print(f'{(1-VALIDATION_PCT)*100}% Pneumonia Cases: {int(all_data[all_data.Pneumonia==1].shape[0]*(1-VALIDATION_PCT))}')
print(f'{VALIDATION_PCT*100}% Pneumonia Cases: {int(all_data[all_data.Pneumonia==1].shape[0]*VALIDATION_PCT)}')
print()
print(f'Pneumonia Cases in Training set: {train_set[train_set.Pneumonia==1].shape[0]}')
print(f'Pneumonia Cases in Validation set: {val_set[val_set.Pneumonia==1].shape[0]}')
print()
print(f'Train Set Size: {train_set.shape[0]}')
print(f'Pos %: {train_set[train_set.Pneumonia==1].shape[0] / train_set.shape[0] *100:.2f}')
print(f'Neg %: {train_set[train_set.Pneumonia==0].shape[0] / train_set.shape[0] *100:.2f}')
print()
print(f'Validation Set Size: {val_set.shape[0]}')
print(f'Pos %: {val_set[val_set.Pneumonia == 1].shape[0] / val_set.shape[0] *100:.2f}')
print(f'Neg % {val_set[val_set.Pneumonia == 0].shape[0] / val_set.shape[0] *100:.2f}')
test_files = glob(os.path.join(TEST_DIR,'*','*'))
test_set = []
for path in test_files:
    label = get_label(path)
    test_set.append((path, label))
    
test_set = pd.DataFrame(test_set, columns=['Path', 'Pneumonia'], index=None)
test_set = test_set.sample(frac=1.).reset_index(drop=True)
def get_train_generator(df, x_col, y_col, shuffle=True, batch_size=BATCH_SIZE, seed=1, target_w = 128, target_h = 128):
    """
    Return generator for training set, normalizing using batch
    statistics.

    Args:
      train_df (dataframe): dataframe specifying training data.
      x_col (str): name of column in df that holds filenames.
      y_col (list): name of column in df as target
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        train_generator (DataFrameIterator): iterator over training set
    """        
    print("[INFO] Getting Train Generator...") 
    # Normalize images
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True,
        shear_range=0.1,
        zoom_range=0.15,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.05,
        horizontal_flip=True, 
        vertical_flip = False, 
        fill_mode = 'reflect')
    
    # Flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=None,
            x_col=x_col,
            y_col=y_col,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w,target_h))
    return generator
def get_test_and_valid_generator(valid_df, test_df, train_df, x_col, y_col, sample_size=100, batch_size=BATCH_SIZE, seed=1, target_w = 128, target_h = 128):
    """
    Return generator for validation set and test test set using 
    normalization statistics from training set.

    Args:
      valid_df (dataframe): dataframe specifying validation data.
      test_df (dataframe): dataframe specifying test data.
      train_df (dataframe): dataframe specifying training data.
      x_col (str): name of column in df that holds filenames.
      y_col (list): name of column in df as target.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        test_generator (DataFrameIterator) and valid_generator: iterators over test set and validation set respectively
    """
    print("[INFO] Getting Valid and Test Generators...")
    # Get generator to sample dataset
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=None, 
        x_col=x_col, 
        y_col=y_col, 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h))
    
    # Get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]

    # Use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(featurewise_center=True,
        featurewise_std_normalization= True)
    
    # Fit generator to sample from training data
    image_generator.fit(data_sample)

    # Get Valid Generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=None,
            x_col=x_col,
            y_col=y_col,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    # Get Test Generator
    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=None,
            x_col=x_col,
            y_col=y_col,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    return valid_generator, test_generator
train_generator = get_train_generator(train_set, "Path", "Pneumonia")
valid_generator, test_generator = get_test_and_valid_generator(val_set, test_set, train_set, "Path", "Pneumonia")
x, y = train_generator.__getitem__(0)
plt.imshow(x[0]);
t_x, t_y = next(train_generator)
fig, m_axs = plt.subplots(2, 4, figsize = (8, 4))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'gray')
    if c_y == 1: 
        c_ax.set_title('Pneumonia')
    else:
        c_ax.set_title('Normal')
    c_ax.axis('off')
def compute_class_freqs(labels):
    """
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """
    # Total number of patients (rows)
    N = labels.shape[0]
    
    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = 1 - positive_frequencies
    
    return positive_frequencies, negative_frequencies
# Computing class frequencies for our training set
freq_pos, freq_neg = compute_class_freqs(train_generator.labels)
pos_weights = freq_neg
neg_weights = freq_pos
class_weights = {0: neg_weights, 1:pos_weights}
def setup_densenet_pretrained_model():
    # Create the base pre-trained model
    base_model = DenseNet121(weights='nih/densenet.hdf5', include_top=False)

    x = base_model.output

    # Add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)

    # And a logistic layer
    predictions = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model
x_ray_class = "Pneumonia_Detection_Model_Dense_FINAL_ATTEMPT"
_time = time.localtime()
prefix = time.strftime("%m_%d_%Y_%H_%M")

weight_path = f"Model_Checkpoints/{x_ray_class}_{prefix}_model_best.hdf5"

cb_checkpoint = ModelCheckpoint(weight_path,
                            monitor = 'val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min',
                            save_weights_only=True)

cb_early = EarlyStopping(monitor='val_loss',
                     mode='min',
                     patience=10)

cb_csv_logger = CSVLogger(f'Logs/{x_ray_class}_{prefix}_log.csv', append=True, separator=';')
call_backs_list = [cb_checkpoint, cb_early, cb_csv_logger]
run_custom_training = False
if run_custom_training:
    t_start = time.time()
    print(f"[INFO] Model Training Started : {time.ctime(t_start)}")    
    history = model.fit(train_generator,
                        steps_per_epoch=len(train_generator),
                        epochs=10,
                        validation_data=valid_generator,
                        validation_steps=len(valid_generator),
                        class_weight=class_weights,
                        callbacks = call_backs_list)
    model.load_weights(weight_path)
    print(f"[INFO] Model Training Finished : {time.ctime(time.time())}")
    model.save("Models/Pneumonia_Detection_Model_Final_Dense_FINAL_ATTEMPT.h5")
    print(f"[INFO] Model Saved")
    t_finish = time.time()
    print(f'Training took {(t_finish - t_start) // 60} minutes')
    plt.plot(history.history['loss'])
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("Training Loss Curve")
    plt.show()
# Functions for getting model predictions and plotting results
def get_model_predictions(model_filepath, data):
    """
    Takes in a filepath to the saved model, loads in the model, and gets predictions for all data sent in.
    :param model_filepath: filepath to a saved model, must have .h5 extension
    :param data: Data to make predictions on
    :return: True labels, predictions
    """
    model = tf.keras.models.load_model(model_filepath)
    pred_y = model.predict(data)
    return [data for answer in pred_y for data in answer]
## Look at a sample of predicted v. true values along with model probabilities:
fig, m_axs = plt.subplots(2, 4, figsize = (20, 10))
i = 0
for (c_x, c_y, c_ax) in zip(image_batch[0:8], true_y[0:8], m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone')
    prob = round(float(pred_y[i]),3)
    if c_y != pred_y_binary[i]:
        c_ax.set_title(f'True {c_y}, Predicted {pred_y_binary[i]}, Prob: {prob}', color = 'red')
    else:
        c_ax.set_title(f'True {c_y}, Predicted {pred_y_binary[i]}, Prob: {prob}', color = 'black')
    c_ax.axis('off')
    i=i+1
predicted_vals = model.predict(test_generator, steps = len(test_generator))
true_y = test_set['Pneumonia'].values
threshold = 0.135
pred_y_binary = [1.0 if pred > threshold else 0.0 for pred in predicted_vals]
pred_y_binary
cm = confusion_matrix(true_y, pred_y_binary)
plt.figure()
plot_confusion_matrix(cm, figsize=(4,4), hide_ticks=True, cmap=plt.cm.Blues)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.savefig(f'Images/CM_Full_Test_Data.png')
plt.show();
y_pred = model.predict(test_generator, steps=len(test_generator),verbose=1)
true_y = test_generator.labels
threshold = 0.5
pred_y_binary = [1.0 if pred > threshold else 0.0 for pred in y_pred]
pred_y_binary
cm = confusion_matrix(true_y, pred_y_binary)
plt.figure()
plot_confusion_matrix(cm, figsize=(4,4), hide_ticks=True, cmap=plt.cm.Blues)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.savefig(f'Images/CM_Full_Test_Data_Thres_50.png')
plt.show();
model_evaluation = model.evaluate(test_generator)
print(f"Loss of Model on Test Set: {model_evaluation[0]*100:.2f}%")
print(f"Accuracy of Model on Test Set: {model_evaluation[1]*100:.2f}%")
print(f"AUC of Model on Test Set: {model_evaluation[2]*100:.2f}%")
print(f"Precision of Model on Test Set: {model_evaluation[3]*100:.2f}%")
print(f"Recall of Model on Test Set: {model_evaluation[4]*100:.2f}%")
history_logs = pd.read_csv('Logs/Pneumonia_Detection_Model_Dense_FINAL_ATTEMPT_10_01_2020_11_49_log.csv',sep=';')
history_logs
# Plot Model History
def plot_history(model_history, model_name):
    N = len(model_history['loss'])
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0,N), model_history['loss'], label='train loss')
    plt.plot(np.arange(0,N), model_history['val_loss'], label='valuation loss')
    plt.plot(np.arange(0,N), model_history['accuracy'], label='train accuracy')
    plt.plot(np.arange(0,N), model_history['val_accuracy'], label='valuation accuracy')
    plt.title("Model Training Results")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss \ Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(f'Images/{model_name}_history.png')
    plt.plot()
plot_history(history_logs, "DenseNet")
def plot_roc_curve(model_name, true_y, pred_y):
    """
    Plot the ROC curve along with the curves AUC for a given model. Note make sure true_y and pred_y are from the same model as model_name
    :param model_name: Name of model used for saving plot
    :param true_y: true labels for dataset
    :param pred_y: predicted labels for dataset
    """
    fig, ax = plt.subplots(1,1, figsize=(7,7))
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(true_y, pred_y)
    ax.plot(fpr, tpr, label=f'{model_name} AUC: {sklearn.metrics.auc(fpr,tpr)}')
    ax.legend()
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    plt.savefig(f'Images/{model_name}_roc.png')
    return

def plot_precision_recall_curve(model_name, true_y, pred_y):
    """
    Plot the precision recall curve for a given model. Note make sure true_y and pred_y are from the same model as model_name
    :param model_name: Name of model used for saving plot
    :param true_y: true labels for dataset
    :param pred_y: predicted labels for dataset
    """
    fig, ax = plt.subplots(1,1, figsize=(7,7))
    precision, recall, thresholds = precision_recall_curve(true_y, pred_y)
    ax.plot(recall, precision, label=f'{model_name} AP Score: {sklearn.metrics.average_precision_score(true_y,pred_y)}')
    plt.legend()
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    plt.savefig(f'Images/{model_name}_precision_recall.png')
    return

def calculate_f1_scores(precision, recall):
    return [(2*p*r)/(p+r) for p,r in zip(precision,recall)]

def plot_f1_score(model_name, true_y, pred_y):
    """
    Plot F1 Scores for a given model. Note make sure true_y and pred_y are from the same model as model_name
    F1 = 2*(precision*recall) / (precision + recall)
    :param model_name: Name of model used for saving plot
    :param true_y: true labels for dataset
    :param pred_y: predicted labels for the dataset
    """
    fig, ax = plt.subplots(1,1,figsize=(7,7))
    precision, recall, thresholds = precision_recall_curve(true_y,pred_y)
    ax.plot(thresholds, precision[:-1], label=f'{model_name} Precision')
    ax.plot(thresholds, recall[:-1], label=f'{model_name} Recall')
    f1_scores = calculate_f1_scores(precision, recall)[:-1]
    ax.plot(thresholds, f1_scores, label=f'{model_name} F1 Score')
    plt.legend()
    ax.set_xlabel("Threshold Values")
    plt.savefig(f'Images/{model_name}_f1_score.png')
    return max(f1_scores)
predicted_vals = model.predict(test_generator, steps = len(test_generator))
true_y = test_set['Pneumonia'].values
plot_roc_curve("DenseNet", true_y, predicted_vals)
plot_precision_recall_curve("DenseNet", true_y, predicted_vals)
plot_f1_score('DenseNet', true_y, predicted_vals)
print("Original - PNEUMONIA")
compute_gradcam(model, 'PNEUMONIA\\person20_virus_51.jpeg', train_set, labels, labels_to_show)
print("Original - NORMAL")
compute_gradcam(model, 'NORMAL\\NORMAL2-IM-0051-0001.jpeg', train_set, labels, labels_to_show)
print("Original - PNEUMONIA")
compute_gradcam(model, 'PNEUMONIA\\person44_virus_93.jpeg', train_set, labels, labels_to_show)
print("Original - NORMAL")
compute_gradcam(model, 'NORMAL\\NORMAL2-IM-0374-0001.jpeg', train_set, labels, labels_to_show)
print("Original - PNEUMONIA")
compute_gradcam(model, 'PNEUMONIA\\person1679_virus_2896.jpeg', train_set, labels, labels_to_show)