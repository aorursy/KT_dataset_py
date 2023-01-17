# Necessary Dependencies
import numpy as np 
import pandas as pd 
from glob import glob
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from itertools import chain
from datetime import datetime
import statistics

print('Started')
# !git clone https://github.com/brucechou1983/CheXNet-Keras.git
# Establish Directories 

if not os.path.exists('logs'):
    os.makedirs('logs')
    
if not os.path.exists('callbacks'):
    os.makedirs('callbacks')
    
CALLBACKS_DIR = '/kaggle/output/callbacks'
# Disease Names 
disease_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion','Pneumonia', 'Pleural_Thickening',
'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']
# Load Stanford Images Distribution Files

labels_train_val = pd.read_csv('/kaggle/input/txt-file/train_val_list.txt')
labels_train_val.columns = ['Image_Index']

labels_test = pd.read_csv('/kaggle/input/txt-file/test_list.txt')
labels_test.columns = ['Image_Index']


print(labels_test['Image_Index'])
# print(len(labels_train_val),len(labels_test))
labels_df = pd.read_csv('/kaggle/input/data/Data_Entry_2017.csv')
labels_df
# NIH Dataset Labels CSV File 

labels_df = pd.read_csv('/kaggle/input/data/Data_Entry_2017.csv')

labels_df.columns = ['Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
                  'Patient_Age', 'Patient_Gender', 'View_Position',
                  'Original_Image_Width', 'Original_Image_Height',
                  'Original_Image_Pixel_Spacing_X',
                  'Original_Image_Pixel_Spacing_Y', 'dfd']
print(labels_df)
labels_df['Finding_Labels'] = labels_df['Finding_Labels'].map(lambda x: x.replace('No Finding', str(None)))

labels_df['Finding_Labels'] = labels_df['Finding_Labels'].replace('None', None)
# Binarizes Each Disease Class in their own column
from tqdm import tqdm

for diseases in tqdm(disease_labels): #TQDM is a progress bar setting
    labels_df[diseases] = labels_df['Finding_Labels'].map(lambda result: 1 if diseases in result else 0)
#     print(labels_df['Finding_Labels'].map(lambda result: 1 if diseases in result else 0))
train_df1 = labels_df.drop(['Finding_Labels','Follow_Up_#','Patient_Age','Patient_Gender','View_Position','Original_Image_Width','Original_Image_Height','Original_Image_Pixel_Spacing_X','Original_Image_Pixel_Spacing_Y','dfd'],axis=1)
train_df1
train_val_merge = pd.merge(left=labels_train_val, right=labels_df, left_on='Image_Index', right_on='Image_Index')

test_merge = pd.merge(left=labels_test, right=labels_df, left_on='Image_Index', right_on='Image_Index')



print(train_val_merge,test_merge)
# np.sum(train_val_merge['No Finding']==1)
# Splitting Finding Labels
train_val_merge['Finding_Labels'] = train_val_merge['Finding_Labels'].apply(lambda s: [l for l in str(s).split('|')])

test_merge['Finding_Labels'] = test_merge['Finding_Labels'].apply(lambda s: [l for l in str(s).split('|')])

print(test_merge)
# Mapping Images
num_glob = glob('/kaggle/input/data/*/images/*.png')
img_path = {os.path.basename(x): x for x in num_glob}

train_val_merge['Paths'] = train_val_merge['Image_Index'].map(img_path.get)
test_merge['Paths'] = test_merge['Image_Index'].map(img_path.get)
os.mkdir("/kaggle/working/data/")
# Mapping Images
num_glob = glob('/kaggle/input/data/*/images/*.png')
img_path = {os.path.basename(x): x for x in num_glob}
from tqdm import tqdm
import os,cv2
#
for value in tqdm(img_path.values()):
    I = cv2.cvtColor(cv2.imread(value),cv2.COLOR_BGR2RGB)
    I = cv2.resize(I,(320,320))
    filename = os.path.splitext(os.path.basename(value))[0]
    cv2.imwrite("/kaggle/working/data/"+filename+".jpg",I)
#     print(value)
train_val_merge
train_val_merge
print(len(num_glob))
# Delete No Finding Class

#train_val_merge = train_val_merge.drop(train_val_merge[train_val_merge.Del == 1].index)
#test_merge = test_merge.drop(test_merge[test_merge.Del == 1].index)
# No Overlap in patients between the Train and Validation Data Sets
patients = np.unique(train_val_merge['Patient_ID'])
test_patients = np.unique(test_merge['Patient_ID'])

print('Number of Patients Between Train-Val Overall: ', len(patients))
print('Number of Patients Between Train-Val Overall: ', len(test_patients))
# Train-Validation Split 
train_df, val_df = train_test_split(patients,
                                   test_size = 0.0669,
                                   random_state = 2019,
                                    shuffle= True)  


print('No. of Unique Patients in Train dataset : ',len(train_df))
train_df = train_val_merge[train_val_merge['Patient_ID'].isin(train_df)]
print('Training Dataframe   : ', train_df.shape[0],' images',)

print('\nNo. of Unique Patients in Validtion dataset : ',len(val_df))
val_df = train_val_merge[train_val_merge['Patient_ID'].isin(val_df)]
print('Validation Dataframe   : ', val_df.shape[0],' images')

print('\nNo. of Unique Patients in Testing dataset : ',len(test_patients))
test_df = test_merge[test_merge['Patient_ID'].isin(test_patients)]
print('Testing Dataframe   : ', test_df.shape[0],' images')
print(train_df)
train_df1 = train_df.drop(['Finding_Labels','Follow_Up_#','Patient_Age','Patient_Gender','View_Position','Original_Image_Width','Original_Image_Height','Original_Image_Pixel_Spacing_X','Original_Image_Pixel_Spacing_Y','dfd','Paths'],axis=1)
val_df1= val_df.drop(['Finding_Labels','Follow_Up_#','Patient_Age','Patient_Gender','View_Position','Original_Image_Width','Original_Image_Height','Original_Image_Pixel_Spacing_X','Original_Image_Pixel_Spacing_Y','dfd','Paths'],axis=1)
test_df1= test_df.drop(['Finding_Labels','Follow_Up_#','Patient_Age','Patient_Gender','View_Position','Original_Image_Width','Original_Image_Height','Original_Image_Pixel_Spacing_X','Original_Image_Pixel_Spacing_Y','dfd','Paths'],axis=1)

val_df1
train_df1
test_df1
labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']
def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=8, seed=1, target_w = 320, target_h = 320):
    """
    Return generator for training set, normalizing using batch
    statistics.

    Args:
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        train_generator (DataFrameIterator): iterator over training set
    """        
    print("getting train generator...") 
    # normalize images
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True)
    
    # flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w,target_h))
    
    return generator
def get_test_and_valid_generator(valid_df, test_df, train_df, image_dir, x_col, y_cols, sample_size=100, batch_size=8, seed=1, target_w = 320, target_h = 320):
    """
    Return generator for validation set and test test set using 
    normalization statistics from training set.

    Args:
      valid_df (dataframe): dataframe specifying validation data.
      test_df (dataframe): dataframe specifying test data.
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        test_generator (DataFrameIterator) and valid_generator: iterators over test set and validation set respectively
    """
    print("getting train and valid generators...")
    # get generator to sample dataset
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=IMAGE_DIR, 
        x_col="Image", 
        y_col=labels, 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h))
    
    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    # fit generator to sample from training data
    image_generator.fit(data_sample)

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))

    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    return valid_generator, test_generator
IMAGE_DIR = "nih/images-small/"
train_generator = get_train_generator(train_df, IMAGE_DIR, "Image", labels)
valid_generator, test_generator= get_test_and_valid_generator(valid_df, test_df, train_df, IMAGE_DIR, "Image", labels)
X_train = train_df['Paths']
y_train = train_df['Finding_Labels']

X_val = val_df['Paths']
y_val = val_df['Finding_Labels']

X_test = test_df['Paths']
y_test = test_df['Finding_Labels']
X_val
# y_val
print(y_train)
XX_train = []
for i in X_train[:]:
    XX_train.append(i)
print(len(XX_train))

X_train = XX_train
X_train
# Binarizing Labels 
from sklearn.preprocessing import MultiLabelBinarizer

print("Labels - ")
mlb = MultiLabelBinarizer()
mlb.fit(y_train)

N_LABELS = len(mlb.classes_)
print(N_LABELS,mlb.classes_)
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i, label))
print(mlb.classes_[0])


IMG_IND = 224
BATCH_SIZE = 32
# Print a link to Tensorflow Data 
import tensorflow as tf

# Checkout Tensorflow.Data for more elaboration
def parse_function(filename, label):

    # Read an image from a file
    image_string = tf.io.read_file(filename)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_png(image_string, channels=3) # channels=3
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [IMG_IND, IMG_IND])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    
    return image_normalized, label
AUTOTUNE = tf.data.experimental.AUTOTUNE 
SHUFFLE_BUFFER_SIZE = 1280 
def create_dataset(filenames, labels):

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    

    # Batch the data for multiple steps
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset
# os.makedirs("/kaggle/working/"+"images_001"+"/"+"images")
# os.makedirs("/kaggle/working/CheXNet-Keras/data/"+"images/")
y_train_bin = mlb.transform(y_train)
print(y_train,)
import cv2
y_train_bin = mlb.transform(y_train)
list(y_train_bin[0])
# print(np.asarray([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]))
z = np.array(train_df['Finding_Labels'])
z[0]
# transform the targets of the training and test sets
import cv2
y_train_bin = mlb.transform(y_train)

y_val_bin = mlb.transform(y_val)
y_test_bin = mlb.transform(y_test)
# transform the targets of the training and test sets
import cv2
y_train_bin = mlb.transform(y_train)
ko = 0
for k in range(len(y_train_bin)):
    
#     y_train
    path_name = X_train[k].split("/")[-3]
    foldr_name = X_train[k].split("/")[-2]
    file_name = X_train[k].split("/")[-1]
    file_name = file_name.split(".")[0]
    

    l = list(y_train_bin[k])
    if l == [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]:
        print(y_train_bin[k],X_train[k],z[k])
#     cv2.imwrite("/kaggle/working/save_img/x.jpg"
# print(y_train_bin)
y_val_bin = mlb.transform(y_val)
y_test_bin = mlb.transform(y_test)
# import glob

# print(len(os.listdir("/kaggle/working/CheXNet-Keras/data/images/")))
train_ds = create_dataset(X_train, y_train_bin.astype(np.float32))
val_ds = create_dataset(X_val, y_val_bin.astype(np.float32))
test_ds = create_dataset(X_test, y_test_bin.astype(np.float32))
# Load in Stanford Distribution charts of images 
# Print # of diseases in each Train-Val-Test
### Turn this into a function ###

print('# of Diseases in each Class - Training')
for i in disease_labels:
    print(i,int(train_df[i].sum()))

print('\n')

print('# of Diseases in each Class - Validation')
for i in disease_labels:
    print(i,int(val_df[i].sum()))

print('\n')

print('# of Diseases in each Class - Testing')
for i in disease_labels:
    print(i,int(test_df[i].sum()))

print('\n')
# Visualize a couple of images 
import matplotlib.style as style
from PIL import Image


nobs = 6 # Maximum number of images to display
ncols = 3 # Number of columns in display
nrows = nobs//ncols # Number of rows in display
'''
style.use("default")
plt.figure(figsize=(8,4*nrows))
for i in range(nrows*ncols):
    ax = plt.subplot(nrows, ncols, i+1)
    plt.imshow(Image.open(X_train[i]))
    plt.title(y_train[i], size=10)
    plt.axis('off')
    '''
# Show Original vs Downsampled Images
# Show Random Horizontal Flip

for i in range(4):
    #print(X_train[i], y_train_bin[i])
    #print(X_val[i], y_val_bin[i])
    print(X_test[i], y_test_bin[i])
    
# DenseNet Dependencies
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense,Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC, BinaryAccuracy, FalsePositives
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from keras import backend as K
import tensorflow as tf
print("TensorFlow version: ", tf.__version__)
# Hyperparameters

IMG_SHAPE = (224,224,3)

EPOCHS = 100

STEPS_PER_EPOCH = 300

OPTIMIZER = Adam(learning_rate=0.001, ####### Modified
                 beta_1=0.9,
                 beta_2=0.999)

LOSS = BinaryCrossentropy() # Not un-weighted 

METRICS = ['BinaryAccuracy']
# Test if GPU present
print("Num GPUs Used: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Section of Code written by brucechou1983 - https://github.com/brucechou1983/CheXNet-Keras
# I have no experience with class weighting, brucechou1983 provided a very thorough explanation of the topic with example code
CLASS_NAMES = disease_labels
def get_class_weights(total_counts, class_positive_counts, multiply):
    """
    Calculate class_weight used in training
    Arguments:
    total_counts - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    multiply - int, positve weighting multiply
    use_class_balancing - boolean 
    Returns:
    class_weight - dict of dict, ex: {"Effusion": { 0: 0.01, 1: 0.99 }, ... }
    """
    def get_single_class_weight(pos_counts, total_counts):
        denominator = (total_counts - pos_counts) * multiply + pos_counts
        return {
            0: pos_counts / denominator,
            1: (denominator - pos_counts) / denominator,
        }

    class_names = list(class_positive_counts.keys())
    label_counts = np.array(list(class_positive_counts.values()))
    class_weights = []
    for i, class_name in enumerate(class_names):
        class_weights.append(get_single_class_weight(label_counts[i], total_counts))

    return class_weights

def get_sample_counts(output_dir, dataset, class_names):
    """
    Get total and class-wise positive sample count of a dataset
    Arguments:
    output_dir - str, folder of dataset.csv
    dataset - str, train|dev|test
    class_names - list of str, target classes
    Returns:
    total_count - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    """
    df = pd.read_csv(os.path.join(output_dir, f"{dataset}.csv"))
    total_count = df.shape[0]
    labels = df[class_names].as_matrix()
    positive_counts = np.sum(labels, axis=0)
    class_positive_counts = dict(zip(class_names, positive_counts))
    return total_count, class_positive_counts

newfds = 'newfds'
train_counts, train_pos_counts = get_sample_counts(newfds, "/kaggle/input/newfds/train", CLASS_NAMES)
class_weights = get_class_weights(train_counts, train_pos_counts, multiply=1)

# Saves weights every 5 Epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=CALLBACKS_DIR, 
    verbose=1, 
    save_weights_only=True,
    period=25)

tensorboard_callback = tf.keras.callbacks.TensorBoard("logs")

reduced_lr = tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=.1,
                        patience=5,
                        verbose=1,
                        mode='min',
                        cooldown=0,
                        min_lr=1e-8 
                        )
with tf.device('/GPU:0'):

    base_model = tf.keras.applications.DenseNet121(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
                                                   #pooling="avg")

    base_model.trainable = True

    x = base_model.output

    x = GlobalAveragePooling2D()(x)
    predictions = Dense(14, activation='sigmoid',name='Final')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    #model.summary()
    
    #latest = tf.train.latest_checkpoint(pre_weights)
    #print('Loading Weights from file: ', latest)
    #weights_DIR = '/kaggle/input/bruce-weights/brucechou1983_CheXNet_Keras_0.3.0_weights.h5'
    #model.load_weights(weights_DIR)

    model.compile(loss = LOSS,
                  optimizer=OPTIMIZER,
                  metrics=METRICS
                                 )


history = model.fit(
                    train_ds,
                    steps_per_epoch = STEPS_PER_EPOCH,
                    validation_data= create_dataset(X_val, y_val_bin.astype(np.float32)),
                    validation_steps = STEPS_PER_EPOCH / 10, 
                    epochs=EPOCHS,
                    #use_multiprocessing=True,
                    #class_weight = class_weights,
                    callbacks=[reduced_lr, tensorboard_callback, cp_callback]
                    )
    
model.save_weights(CALLBACKS_DIR)
# Graph of Binary Accuracy 

acc = history.history['BinaryAccuracy']
val_acc = history.history['val_BinaryAccuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(40, 10))
plt.subplot(1, 2, 1)
plt.grid()
plt.plot(epochs_range, acc, label='Training Binary Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Binary Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Binary Accuracy', color='Green')
#fig.savefig('TrainingValidationAccuracy.png')
# Graph of Loss
plt.figure(figsize=(40, 10))
plt.subplot(1, 2, 2)
plt.grid()
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss', color='red')
plt.show()
# Load Model
#model.load_weights(weights_DIR)
# Loading Latest Weights 
#latest = tf.train.latest_checkpoint(checkpoint_dir)
#print('Loading Weights from file: ', latest)
STEPS = len(y_test_bin)
print(STEPS)
# Predict
pred = model.predict(test_ds,steps=STEPS,verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)
y_test = y_test_bin
pred = pred
# Print AUC scores
labels = []
scores = []
for i in range(14):
    
    try:
        print('{0:<20}\t{1:<1.6f}'.format(disease_labels[i], roc_auc_score(y_test[:,i], pred[:,i])))
        labels.append(disease_labels[i])
        scores.append(roc_auc_score(y_test[:,i], pred[:,i]))
        
        
    except:
        print('Not Working')
    
print('done') 
# True Postive Vs ...
from sklearn.metrics import roc_curve, auc, roc_auc_score
fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(disease_labels):
    fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), pred[:,idx])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('Completed_Training_CheXNet.png')
# Bar Graph
import matplotlib
my_mean_aur = round(statistics.mean(scores), 3) * 100 
print(f'My AUROC mean score: {my_mean_aur}% ')

Stanford_scores = [0.8094, 0.7901, 0.7345, 0.8887, 0.8878, 0.9371, 0.8047, 0.8638, 0.7680, 0.8062, 0.9248, 0.7802, 0.8676, 0.9164]
stanford_mean_aur = round(statistics.mean(Stanford_scores), 3) * 100 
print(f'Stanford Auroc mean score: {stanford_mean_aur}% ')

labels = ['Stanford Mean AUROC', 'My Mean AUROC']
mean_scores = [stanford_mean_aur, my_mean_aur]

matplotlib.pyplot.bar(labels, mean_scores)


plt.show()
# Stanford MEAN AUROC Scores
stanford_labels = disease_labels
print(stanford_labels)
stanford_scores = Stanford_scores
print(stanford_scores)


# Disease Class Scores

labels = stanford_labels

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(40,20))
rects1 = ax.bar(x - width/2, stanford_scores, width, label='Stanford AUROC', color='green')
rects2 = ax.bar(x + width/2, scores, width, label='MY CHEXNET',color='orange')

ax.set_ylabel('Scores', color='grey', fontsize= 30)
ax.set_title('Comparing AUROC Scores')
ax.set_xticks(x)
ax.set_xlabel('Disease Classes', color='grey', fontsize= 30)
ax.set_xticklabels(labels, color='grey', fontsize= 17)
ax.legend()


plt.show()
fig.savefig('ComparisonAUROC.png')
# Notes


