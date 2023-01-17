# check system that is running
import platform
import sys

# Show all warnings in IPython
import warnings
warnings.filterwarnings("always")
# Ignore specific numpy warning, as per: <https://github.com/numpy/numpy/issues/11788#issuecomment-422846396>
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings(
    "ignore", message="can't resolve package from __spec__ or __package__")

# navigate folders
from glob import glob
import os
from pathlib import Path

# saving output (with a timestamp)
import pickle
import shutil

# other utils
import time
import datetime
import re
import random

# to handle datasets
import numpy as np
import pandas as pd

# for plotting
from matplotlib import __version__ as mpl_version
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# to open the images
import cv2

# to display all the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns', None)

# data preprocessing
from sklearn import __version__ as sk_version
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# evaluate model and separate train and test
from sklearn.metrics import confusion_matrix

# model fitting
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

# Confirm expected versions (i.e. the versions running in the Kaggle Kernel)
assert platform.python_version() == '3.6.6'
print(f"Python version:\t\t{sys.version}")
assert pd.__version__ == '0.25.3'
print(f"pandas version:\t\t{pd.__version__}")
assert np.__version__ == '1.18.2'
print(f"numpy version:\t\t{np.__version__}")
assert mpl_version == '3.2.1'
print(f"matplotlib version:\t{mpl_version}")
assert sns.__version__ == '0.10.0'
print(f"seaborn version:\t{sns.__version__}")
assert cv2.__version__ == '4.2.0'
print(f"cv2 version:\t\t{cv2.__version__}")
assert sk_version == '0.22.2.post1'
print(f"sklearn version:\t{sk_version}")
assert tf.__version__ == '2.1.0'
print(f"tensorflow version:\t{tf.__version__}")
# Ignore warnings that can show up, specific to Keras
warnings.filterwarnings(
    "ignore", message="unclosed file <_io.TextIOWrapper name='/root/.keras/keras.json'")

# for the convolutional network
from keras import __version__ as keras_version
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras import losses
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing import image
from keras.utils import np_utils

# Confirm expected version
assert keras_version == '2.3.1'
print(f"keras version:\t{keras_version}")
# Configuration variables
NOTEBOOK_FOLDER = Path('/')  # Change this to the location of your notebook
DATA_FOLDER = NOTEBOOK_FOLDER / 'kaggle' / 'input' / 'v2-plant-seedlings-dataset'
# each weed class is in a dedicated folder
print('\t'.join(os.listdir(DATA_FOLDER)))
# let's walk over the directory structure, so we understand
# how the images are stored
max_print_subfolders = 4
max_print_files_per_folder = 3
subfolder_counter = 0
for class_folder_path in DATA_FOLDER.iterdir():
    subfolder_counter += 1
    if subfolder_counter > max_print_subfolders:
        print(str(DATA_FOLDER / '...') + "more subfolders in this folder...")
        break
    file_counter = 0
    for image_path in class_folder_path.glob("*.png"):
        file_counter += 1
        if file_counter > max_print_files_per_folder:
            print(str(class_folder_path / '...') + "more files in this folder...\n")
            break
        print(image_path)
# let's create a dataframe:
# the dataframe stores the image file name in one column
# and the class of the weed (the target) in the next column
images_df = pd.DataFrame.from_records([
    (image_file_path.name, image_file_path.parent.name) for 
    image_file_path in DATA_FOLDER.glob("*/*.png")  # Only look one subfolder down
], columns=['image', 'target']).sort_values(['target', 'image'])

def get_image_file_path(images_row, DATA_FOLDER=DATA_FOLDER):
    """Get the file path from a row of images_df"""
    return(DATA_FOLDER / images_row.target / images_row.image)

images_df.head(10)
# how many images do we have per class?
images_df['target'].value_counts()
# let's isolate a path, for demo
# we want to load the image in this path later
images_df.loc[0, :].agg(get_image_file_path)
# let's visualise a few images
# if the images you see in your notebook are not the same, don't worry

def plot_single_image(df, image_number):
    im = cv2.imread(str(df.loc[image_number, :].agg(get_image_file_path)))
    plt.title(df.loc[image_number, :].agg(lambda x: f"{x.target}: {x.image}"))
    plt.imshow(im)
    
plot_single_image(images_df, 0)
plot_single_image(images_df, 3000)
plot_single_image(images_df, 1000)
# let's go ahead and plot a bunch of our images together,
# so we get e better feeling of how our images look like

def plot_for_class(df, label):
    # function plots 9 images
    nb_rows = 3
    nb_cols = 3
    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(10, 10))
    n = 0
    for i in range(0, nb_rows):
        for j in range(0, nb_cols):
            tmp = df[df['target'] == label]
            tmp.reset_index(drop=True, inplace=True)
            im = cv2.imread(str(tmp.loc[n,:].agg(get_image_file_path)))
            axs[i, j].set_title(tmp.loc[n, :].agg(lambda x: f"{x.target}: {x.image}"))
            axs[i, j].imshow(im)
            n += 1 
plot_for_class(images_df, 'Cleavers')
plot_for_class(images_df, 'Maize')
plot_for_class(images_df, 'Common Chickweed')
# train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    images_df.target + '/' + images_df.image, images_df.target,
    test_size=0.20, random_state=101
)
print(X_train.shape)
print(X_test.shape)
# the indices of the training data are shuffled
# this will cause problems later
X_train.head()
# reset index, because later we iterate over row number
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

# reset index in target as well
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

print(X_train.head())
y_train.value_counts(normalize=True) - y_test.value_counts(normalize=True)
# percentage of images within each class for
# train should be (roughly) the same in the test set
thresh = 1.2e-2
assert (np.abs(
    y_train.value_counts(normalize=True) - y_test.value_counts(normalize=True)
) < thresh).all()
print(f'Proportions are within the threshold of: {thresh:.1%}\n')
y_train.value_counts(normalize=True).to_frame("Proportion of sample") \
.style.format('{:.2%}')
# let's prepare the target
# it is a multiclass classification, so we need to make 
# one hot encoding of the target

encoder = LabelEncoder()
encoder.fit(y_train)

train_y = np_utils.to_categorical(encoder.transform(y_train))
test_y = np_utils.to_categorical(encoder.transform(y_test))

print(train_y.shape)
print('')
print(train_y[:10])
# The images in our folders, are all different sizes
# For neural networks however, we need images in the same size
# The images will all be resized to this size:

IMAGE_SIZE = 150
def im_resize(image_location, image_size=IMAGE_SIZE, DATA_FOLDER=DATA_FOLDER):
    return(cv2.resize(
        cv2.imread(str(DATA_FOLDER / image_location)),
        (IMAGE_SIZE, IMAGE_SIZE)
    ))
tmp = im_resize(X_train[7])
tmp.shape
# the shape of the datasets needs to be (n1, n2, n3, n4)
# where n1 is the number of observations
# n2 and n3 are image width and length
# and n4 indicates that it is a color image, so 3 planes per image

def create_dataset(image_locations, **kwargs):
    """**kwargs: Additional arguments to im_resize()"""
    return(np.array([
        im_resize(image_location, **kwargs) for image_location in image_locations
    ]))
%%time
# Took me approx: 45 secs
x_train = create_dataset(X_train, image_size=IMAGE_SIZE)
print(f'Train Dataset Images shape: {x_train.shape}   size: {x_train.size:,}\n')
%%time
# Took me approx: 15 secs
x_test = create_dataset(X_test)
print(f'Train Dataset Images shape: {x_train.shape}   size: {x_train.size:,}')
# number of different classes
y_train.unique().shape[0]
# Specify the cnn
# Source: https://www.kaggle.com/fmarazzi/baseline-keras-cnn-roc-fast-5min-0-8253-lb

# CNN structure parameters
kernel_size = (3,3)
pool_size= (2,2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.3
dropout_dense = 0.3

model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu', 
                 input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(12, activation = "softmax"))

model.summary()
model.compile(
    Adam(lr=0.0001),
    loss=losses.categorical_crossentropy,
    metrics=[metrics.categorical_accuracy]
)
# Training parameters
batch_size = 10
epochs = 8  # Increaseing this would likely get a more accurate model, but increase fitting time
filepath = "model.h5"
# A checkpoint can monitor any one of the model's metrics
# which, in this case, are:
print(model.metrics_names)
# We can't add `val_` to monitor the metric calculated in the validation set
# because we have not specified a validation set.
# See: <https://stackoverflow.com/a/43782410>
# and <https://github.com/tensorflow/tensorflow/issues/33163#issuecomment-539978875>
# Define callbacks to run after specific epochs
checkpoint = ModelCheckpoint(
    filepath, monitor='categorical_accuracy', verbose=1, 
    save_best_only=True, mode='max'
)
reduce_lr = ReduceLROnPlateau(
    monitor='categorical_accuracy', factor=0.5, patience=1, 
    verbose=1, mode='max', min_lr=0.00001
)
callbacks_list = [checkpoint, reduce_lr]
%%time
# Fit model
# See the notebook cell output for how this took to run
# On default Kaggle settings (including after reducing number of threads to 1), took me approx: 45 mins

run_this_command = True  # Set to False to avoid inadvertently running this command

history_filename_base = "fitting_history"
if run_this_command:
    # Recommended commands for making Keras output reproducible
    # Also see Keras docs FAQ: <https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development>
    # Adapted to use TensorFlow v2
    seed_value = 7
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)
    # 5. Configure a new global `tensorflow` session
    session_conf = tfv1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tfv1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tfv1.keras.backend.set_session(sess)
    
    # Fit models
    history = model.fit(
        x=x_train, y=train_y,
        batch_size=batch_size, 
        validation_split=10,
        epochs=epochs,
        verbose=2,
        callbacks=callbacks_list
    )
    
    # Save history (with timestamp in the filename)
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
    new_filename = f"{history_filename_base}_{st}.pkl"
    with open(new_filename, "wb") as output_file:
        pickle.dump(history, output_file)
else:
    print("Command has *not* been run\n")
    previous_files = [path for path in Path(os.getcwd()).glob(f"{history_filename_base}_*.pkl")]
    if len(previous_files) == 0:
        print("No previous files available. History not loaded.\n")
    else:
        with open(sorted(previous_files)[-1], 'rb') as input_file:
            history = pickle.load(input_file)
        print("Most recent history file reloaded.\n")
# View fitting history
acc = history.history['categorical_accuracy']
loss = history.history['loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.title('Training accuracy')
plt.legend()
plt.show()
# Note that the History object also contains the fitted model object
# (Interactively, I checked that the predictions on the test set
# exactly matched between `history.model` and `model`)
type(history.model)
%%time
# calculate predictions on test set
# Took me approx: 20 secs
predictions = model.predict_classes(x_test, verbose=1)
# inspect predictions
predictions[:50]
# Get labels for the confusion matrix
# create a dict to map back the numbers onto the classes
encoding_dict = dict(zip(range(len(encoder.classes_)), encoder.classes_))
abbreviation_dict = {}
for code, label in encoding_dict.items():
    label_words = re.split(r"[\s-]", label)
    if len(label_words) == 1:
        abbreviation_dict[code] = label_words[0][:2]
    else:
        abbreviation_dict[code] = ''.join([label_word[0].upper() for label_word in label_words])
abbreviation = pd.DataFrame.from_dict(
    abbreviation_dict, columns=['abbrev'], orient='index'
).sort_index()
abbreviation.T
# get confusion matrix
cnf_matrix = confusion_matrix(encoder.transform(y_test), predictions)
# Plot
def plot_cnf_mx(cnf_matrix, labels=abbreviation.abbrev):
    fig, ax = plt.subplots(1)
    ax = sns.heatmap(cnf_matrix, ax=ax, cmap=plt.cm.Greens, annot=True, fmt='.0f')
    ax.set_xticklabels(abbreviation.abbrev)
    ax.set_yticklabels(abbreviation.abbrev)
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
plot_cnf_mx(cnf_matrix)
plt.title('Confusion Matrix')
#fig.savefig('Confusion matrix.png', dpi=300)
plt.show()
accuracy_score(encoder.transform(y_test), predictions, normalize=True, sample_weight=None)
print(classification_report(encoder.transform(y_test), predictions))
# Fitting history
tol = 1e-7

acc_previous_arr = np.array([0.24396299, 0.46874294, 0.5948996, 0.6765967, 0.7393365, 0.7885353, 0.81403744, 0.8483412])
acc_diff_arr = np.array(acc) - acc_previous_arr
if (np.abs(acc_diff_arr) < tol).all():
    print(f"Correct: Accuracy after each epoch matches to tolerance: {tol}")
else:
    print(f"INCORRECT: Accuracy after each epoch does *not* match to tolerance: {tol}")
print(acc_diff_arr)
print('')

loss_previous_arr = np.array([
    2.8593816470411126, 1.564362328898904, 1.195128699789411, 0.9741775560220298, 
    0.812239264459098, 0.6374371725804406, 0.5471700843447986, 0.44851682055426273
])
loss_diff_arr = np.array(loss) - loss_previous_arr
if (np.abs(loss_diff_arr) < tol).all():
    print(f"Correct: Loss after each epoch matches to tolerance: {tol}")
else:
    print(f"INCORRECT: Loss after each epoch does *not* match to tolerance: {tol}")
print(loss_diff_arr)
# Confusion matrix
cnf_matrix_previous = np.array(
    [[ 14,   0,   0,   0,  10,   0,  32,   0,   2,   0,   1,   2],
       [  0,  82,   4,   0,   0,   0,   0,   1,   0,   1,   0,   0],
       [  0,  10,  55,   0,   2,   2,   0,   0,   0,   0,   0,   0],
       [  0,   0,   1, 131,   0,   0,   0,   1,   5,   0,   1,   0],
       [  0,   0,   6,   1,  34,   4,   1,   1,   6,   0,   0,   1],
       [  0,   2,   6,   5,   1,  96,   0,   1,   3,   0,   2,   1],
       [  7,   0,   0,   2,   9,  14, 112,   1,   4,   0,   0,   1],
       [  0,   1,   0,   3,   1,   0,   0,  49,   2,   0,   3,   2],
       [  0,   0,   3,   4,   0,   0,   0,   4,  93,   5,   0,   5],
       [  0,   2,   1,   8,   0,   2,   0,   0,   6,  26,   2,   0],
       [  0,   4,   0,   2,   0,   2,   0,   3,   1,   1, 105,   0],
       [  0,   3,   4,   0,   0,   3,   0,   1,   3,   0,   2,  74]]
)
cnf_matrix_diff = cnf_matrix - cnf_matrix_previous
if (cnf_matrix_diff == 0).all():
    print(f"Correct: Confusion matrix on test set matches exactly")
else:
    print(f"INCORRECT: Confusion matrix on test set does *not* match exactly")
plot_cnf_mx(cnf_matrix_diff)
plt.title('Differences in Confusion Matrix: current minus previous run')
plt.show()
# Get the image names, actual and predicted categories on the test data
test_df = pd.concat([
    pd.DataFrame(
        X_test.str.split('/').to_list(),
        columns=['target', 'image']
    )[['image', 'target']],
    abbreviation.loc[predictions,:].reset_index(
        drop=True).rename(columns={'abbrev': 'pred_abbrev'}),
], axis=1, sort=False).merge(
    images_df.reset_index().rename(columns={'index': 'original_index'}),
    how='left', on=['image', 'target']
).set_index('original_index')
test_df.head()
# Look for one with these characteristics
def pick_a_plant(
    act_plant_type, pred_plant_abbrev, rand_num=None, original_idx=None,
    save_it=False, DATA_FOLDER=DATA_FOLDER, output_folder = Path('.') / 'sample_images_for_testing'
):
    choosen_bucket = test_df.query(
        "(target == @act_plant_type) & (pred_abbrev == @pred_plant_abbrev)"
    )
    num_in_bucket = choosen_bucket.shape[0]
    if original_idx is None:
        if rand_num is None:
            return(choosen_bucket)
        if rand_num >= num_in_bucket:
            print(f"There are only {num_in_bucket} examples in this bucket. Pick again")
            return(choosen_bucket)
        choosen_row = choosen_bucket.iloc[[rand_num],:]
    else:
        choosen_row = choosen_bucket.loc[[original_idx],:]
    plot_single_image(test_df, choosen_row.index.values[0])
    if save_it:
        to_file_path = output_folder / choosen_row.target.values[0] / choosen_row.image.values[0]
        to_file_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(
           str(DATA_FOLDER / choosen_row.target.values[0] / choosen_row.image.values[0]),
           str(to_file_path)
        )
        print(f"Image saved here: {to_file_path}")
    return(choosen_row)
pick_a_plant('Black-grass', 'LSB', 6, save_it=True)
pick_a_plant('Black-grass', 'BG', 13, save_it=True)
pick_a_plant('Common Chickweed', 'CC', 99, save_it=True)