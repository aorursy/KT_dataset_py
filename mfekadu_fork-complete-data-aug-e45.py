import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
import matplotlib.pyplot as plt
from random import seed # for setting seed
import tensorflow
from IPython import sys_info

import gc # garbage collection
MY_SEED = 42 # 480 could work too
seed(MY_SEED)
np.random.seed(MY_SEED)
tensorflow.set_random_seed(MY_SEED)

print(sys_info())
# get module information
!pip freeze > frozen-requirements.txt
# append system information to file
with open("frozen-requirements.txt", "a") as file:
    file.write(sys_info())
from tensorflow.python.client import device_lib
# print out the CPUs and GPUs
print(device_lib.list_local_devices())
# https://stackoverflow.com/questions/25705773/image-cropping-tool-python
# because painting images are hella big
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
# globals

DATA_DIR = '../input/painters-train-part-1/'

TRAIN_1_DIR = '../input/painters-train-part-1/train_1/train_1/'
TRAIN_2_DIR = '../input/painters-train-part-1/train_2/train_2/'
TRAIN_3_DIR = '../input/painters-train-part-1/train_3/train_3/'

TRAIN_4_DIR = '../input/painters-train-part-2/train_4/train_4/'
TRAIN_5_DIR = '../input/painters-train-part-2/train_5/train_5/'
TRAIN_6_DIR = '../input/painters-train-part-2/train_6/train_6/'

TRAIN_7_DIR = '../input/painters-train-part-3/train_7/train_7/'
TRAIN_8_DIR = '../input/painters-train-part-3/train_8/train_8/'
TRAIN_9_DIR = '../input/painters-train-part-3/train_9/train_9/'

TRAIN_DIRS = [TRAIN_1_DIR, TRAIN_2_DIR, TRAIN_3_DIR,
             TRAIN_4_DIR, TRAIN_5_DIR, TRAIN_6_DIR,
             TRAIN_7_DIR, TRAIN_8_DIR, TRAIN_9_DIR]

TEST_DIR = '../input/painter-test/test/test/'
df = pd.read_csv(DATA_DIR + 'all_data_info.csv')
print("df.shape", df.shape)
# quick fix for corrupted files
list_of_corrupted = ['3917.jpg','18649.jpg','20153.jpg','41945.jpg',
'79499.jpg','91033.jpg','92899.jpg','95347.jpg',
'100532.jpg','101947.jpg']
# display the corrupted rows of dataset for context
corrupt_df = df[df["new_filename"].isin(list_of_corrupted) == True]
print(corrupt_df.head(len(list_of_corrupted)))

# completely get rid of them
df = df[df["new_filename"].isin(list_of_corrupted) == False]

# try to see if they are still there
print(df[df["new_filename"].isin(list_of_corrupted) == True])

print("df.shape", df.shape)

train_df = df[df["in_train"] == True]
test_df = df[df['in_train'] == False]
train_df = train_df[['artist', 'new_filename']]
test_df = test_df[['artist', 'new_filename']]

print("test_df.shape", test_df.shape)
print("train_df.shape", train_df.shape)

artists = {} # holds artist hash & the count
for a in train_df['artist']:
    if (a not in artists):
        artists[a] = 1
    else:
        artists[a] += 1

training_set_artists = []
for a,count in artists.items():
    if(int(count) >= 300):
        training_set_artists.append(a)

print("number of artsits",len(training_set_artists))

print("\nlist of artists...\n", training_set_artists)

t_df = train_df[train_df["artist"].isin(training_set_artists)]

t_df.head(5)
t1_df = t_df[t_df['new_filename'].str.startswith('1')]

t2_df = t_df[t_df['new_filename'].str.startswith('2')]

t3_df = t_df[t_df['new_filename'].str.startswith('3')]

t4_df = t_df[t_df['new_filename'].str.startswith('4')]

t5_df = t_df[t_df['new_filename'].str.startswith('5')]

t6_df = t_df[t_df['new_filename'].str.startswith('6')]

t7_df = t_df[t_df['new_filename'].str.startswith('7')]

t8_df = t_df[t_df['new_filename'].str.startswith('8')]

t9_df = t_df[t_df['new_filename'].str.startswith('9')]

all_train_dfs = [t1_df, t2_df, t3_df,
                t4_df, t5_df, t6_df,
                t7_df, t8_df, t9_df]

t9_df.head(5)
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

len(training_set_artists)
num_classes = len(training_set_artists) # one class per artist
weights_notop_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
model = Sequential()
model.add(ResNet50(
  include_top=False,
  weights=weights_notop_path,
  pooling='avg'
))
model.add(Dense(
  num_classes,
  activation='softmax'
))

model.layers[0].trainable = False
model.compile(
  optimizer='adam', # lots of people reccommend Adam optimizer
  loss='categorical_crossentropy', # aka "log loss" -- the cost function to minimize 
  # so 'optimizer' algorithm will minimize 'loss' function
  metrics=['accuracy'] # ask it to report % of correct predictions
)
# model globals
IMAGE_SIZE = 224
BATCH_SIZE = 96
TEST_BATCH_SIZE = 17 # because test has 23817 images and factors of 23817 are 3*17*467
                     # it is important that this number evenly divides the total num images 
VAL_SPLIT = 0.25
# for loading the sample fit images
from keras.preprocessing.image import load_img, img_to_array
# set up a list of filenames for fitting a sample for the data augmentation
# because it is required if featurewise_center or featurewise_std_normalization
# https://keras.io/preprocessing/image/#fit

num_sample_classes = 2 # 2 files per artist for sample_generator.fit()

# get images
np_arrays_for_sample = []
sample_filenames = []
for a in training_set_artists:
    np_arr = t1_df[t1_df['artist'] == a][:num_sample_classes]
    np_arrays_for_sample.append(np_arr)
    sample_filenames += np_arr['new_filename'].get_values().tolist()

images = [] # takes up a lot of ram
for f in sample_filenames:
    img_path = TRAIN_DIRS[0] + f
    # load an array represention of the sample image
    # must be at the expected target size for data_gen.fit(sample) to work
    images.append(img_to_array(load_img(img_path, target_size=(IMAGE_SIZE,IMAGE_SIZE))))

sample = np.array(images)

del images
del sample_filenames
del np_arrays_for_sample
gc.collect()

# i = 1
# for a in np_arrays_for_sample:
#     artist_name = np_arrays_for_sample[0].artist.get_values()[0]
#     if (artist_name in training_set_artists and len(np_arrays_for_sample[0])):
#         print("good",i)
#         i+=1
#         continue
#     else:
#         print("fail")
#         break
def setup_generators(
    val_split, train_dataframe, train_dir,
    img_size, batch_size, my_seed, list_of_classes,
    test_dataframe, test_dir, test_batch_size, 
    sample # numpy array of sample images to fit onto generator
):
    print("-"*20)
    if not preprocess_input:
          raise Exception("please do import call 'from tensorflow.python.keras.applications.resnet50 import preprocess_input'")

    # setup resnet50 preprocessing 
    data_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=val_split,
        featurewise_center=True, # Set input mean to 0 over the dataset, feature-wise.
        featurewise_std_normalization=True, # Divide inputs by std of the dataset, feature-wise.
        rotation_range=180, # because picasso painting is still picasso if rotated
        zoom_range=0.2, # because picasso painting is still picasso if you look closely
        width_shift_range=0.2, # because still picasso if width = width +/- 0.2
        height_shift_range=0.2, # because still picasso if height = height +/- 0.2
        shear_range=0.3, # bc still picasso if sheer by 0.3 deg in counter-clockwise direction
        horizontal_flip=True, # bc still picasso if flipped horizontally
        vertical_flip=True, # bc still picasso if flipped vertically
        fill_mode='reflect' # bc still picasso if mirrored
    )
    
    data_gen.fit(sample, seed=my_seed)

    print(len(train_dataframe), "images in", train_dir, "and validation_split =", val_split)
    print("\ntraining set ImageDataGenerator")
    train_gen = data_gen.flow_from_dataframe(
        dataframe=train_dataframe.reset_index(), # call reset_index() so keras can start with index 0
        directory=train_dir,
        x_col='new_filename',
        y_col='artist',
        has_ext=True,
        target_size=(img_size, img_size),
        subset="training",
        batch_size=batch_size,
        seed=my_seed,
        shuffle=True,
        class_mode='categorical',
        classes=list_of_classes
    )    

    val_data_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=val_split,
        featurewise_center=True, #
        featurewise_std_normalization=True #
    )
    
    val_data_gen.fit(sample, seed=my_seed)   

    print("\nvalidation set ImageDataGenerator")
    valid_gen = val_data_gen.flow_from_dataframe(
        dataframe=train_dataframe.reset_index(), # call reset_index() so keras can start with index 0
        directory=train_dir,
        x_col='new_filename',
        y_col='artist',
        has_ext=True,
        subset="validation",
        batch_size=batch_size,
        seed=my_seed,
        shuffle=True,
        target_size=(img_size,img_size),
        class_mode='categorical',
        classes=list_of_classes
    )

    test_data_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        featurewise_center=True, # 
        featurewise_std_normalization=True #
    )

    test_data_gen.fit(sample, seed=my_seed)

    print("\ntest set ImageDataGenerator")
    test_gen = test_data_gen.flow_from_dataframe(
        dataframe=test_dataframe.reset_index(), # call reset_index() so keras can start with index 0
        directory=test_dir,
        x_col='new_filename',
        y_col=None,
        has_ext=True,
        batch_size=test_batch_size,
        seed=my_seed,
        shuffle=False, # dont shuffle test directory
        class_mode=None,
        target_size=(img_size,img_size)
    )

    return (train_gen, valid_gen, test_gen)

print("defined setup_generators()")
# delete some unused dataframes to free some RAM for training
del df
del t_df
del t1_df
del t2_df
del t3_df
del t4_df
del t5_df
del t6_df
del t7_df
del t8_df
del t9_df
gc.collect()
train_gens = [None]*len(TRAIN_DIRS)
valid_gens = [None]*len(TRAIN_DIRS)
test_gen  = None # only 1 test_gen
i = 0
for i in range(0, len(TRAIN_DIRS)):
    train_gens[i], valid_gens[i], test_gen = setup_generators(
        train_dataframe=all_train_dfs[i], train_dir=TRAIN_DIRS[i],
        val_split=VAL_SPLIT, img_size=IMAGE_SIZE, batch_size=BATCH_SIZE, my_seed=MY_SEED, 
        list_of_classes=training_set_artists, test_dataframe=test_df, 
        test_dir=TEST_DIR, test_batch_size=TEST_BATCH_SIZE, 
        sample=sample
    )
    i += 1
del sample
gc.collect()
# the tutorial had 10 epochs... 
MAX_EPOCHS = 5 * len(train_gens) # should be a multiple of 9 because need evenly train each train_dir
DIR_EPOCHS = 1 # fit each train_dir at least this many times before overfitting
histories = []

e=0
while ( e < MAX_EPOCHS):
    for i in range(0, len(train_gens)):
        # train_gen.n = number of images for training
        STEP_SIZE_TRAIN = train_gens[i].n//train_gens[i].batch_size
        # train_gen.n = number of images for validation
        STEP_SIZE_VALID = valid_gens[i].n//valid_gens[i].batch_size
        print("STEP_SIZE_TRAIN",STEP_SIZE_TRAIN)
        print("STEP_SIZE_VALID",STEP_SIZE_VALID)
        histories.append(
            model.fit_generator(generator=train_gens[i],
                                steps_per_epoch=STEP_SIZE_TRAIN,
                                validation_data=valid_gens[i],
                                validation_steps=STEP_SIZE_VALID,
                                epochs=DIR_EPOCHS)
        )
        e+=1
accuracies = []
val_accuracies = []
losses = []
val_losses = []
for hist in histories:
    if hist:
        accuracies += hist.history['acc']
        val_accuracies += hist.history['val_acc']
        losses += hist.history['loss']
        val_losses += hist.history['val_loss']
# Plot training & validation accuracy values
plt.plot(accuracies)
plt.plot(val_accuracies)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(losses)
plt.plot(val_losses)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
import time
timestr = time.strftime("%Y%m%d-%H%M%S") # e.g: 20181109-180140
model.save('painters_adam_e45_aug_'+timestr+'.h5')
PRED_STEPS = len(test_gen) #100 # default would have been len(test_gen)
# Need to reset the test_gen before calling predict_generator
# This is important because forgetting to reset the test_generator results in outputs with a weird order.
test_gen.reset()
pred=model.predict_generator(test_gen, verbose=1, steps=PRED_STEPS)
print(len(pred),"\n",pred)
predicted_class_indices=np.argmax(pred,axis=1)
print(len(predicted_class_indices),"\n",predicted_class_indices)
print("it has values ranging from ",min(predicted_class_indices),"...to...",max(predicted_class_indices))
labels = (train_gens[0].class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
print("*"*20+"\nclass_indices\n"+"*"*20+"\n",train_gens[0].class_indices,"\n")
print("*"*20+"\nlabels\n"+"*"*20+"\n",labels,"\n")
print("*"*20+"\npredictions has", len(predictions),"values that look like","'"+str(predictions[0])+"' which is the first prediction and corresponds to this index of the classes:",train_gens[0].class_indices[predictions[0]])
# Save the results to a CSV file.
filenames=test_gen.filenames[:len(predictions)] # because "ValueError: arrays must all be same length"

real_artists = []
for f in filenames:
    real = test_df[test_df['new_filename'] == f].artist.get_values()[0]
    real_artists.append(real)

results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions,
                      "Real Values":real_artists})
results.to_csv("results.csv",index=False)
results.head()
len(training_set_artists)
print(training_set_artists)
count = 0
match = 0
unexpected_count = 0
unexpected_match = 0
match_both_expected_unexpected = 0

for p, r in zip(results['Predictions'], results['Real Values']):
    if r in training_set_artists:
        count += 1
        if p == r:
            match += 1
    else:
        unexpected_count += 1
        if p == r:
            unexpected_match += 1

print("test accuracy on new images for TRAINED artsits")
acc = match/count
print(match,"/",count,"=","{:.4f}".format(acc))

print("test accuracy on new images for UNEXPECTED artsits")
u_acc = unexpected_match/unexpected_count
print(unexpected_match,"/",unexpected_count,"=","{:.4f}".format(u_acc))

print("test accuracy on new images")
total_match = match+unexpected_match
total_count = count+unexpected_count
total_acc = (total_match)/(total_count)
print(total_match,"/",total_count,"=","{:.4f}".format(total_acc))