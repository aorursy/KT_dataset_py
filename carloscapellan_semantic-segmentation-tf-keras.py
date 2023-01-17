import os

import multiprocessing

import glob

import datetime

import random

import time

import numpy as np

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator

import PIL

from PIL import Image

from PIL import ImageChops

from PIL import ImageOps

import pandas as pd

from IPython.display import FileLink
TRAIN_HOME_DIR = "/kaggle/working/stage1_train"

TEST_HOME_DIR = "/kaggle/working/stage1_test"

MODEL_CHECKPOINT_NAME = "/kaggle/working/nucleii_segmentation.h5"

TB_LOGDIR = "/kaggle/working/tensorboardlogs"

TRAIN_SPLIT = 0.8

IMG_SIZE = (512, 512)

NUM_CLASSES = 2

BATCH_SIZE = 4

EPOCHS = 50

AUGMENTATION_ON = False
# Unzip the training data, download and unzip ngrok to run TensorBoard later

!mkdir -p /kaggle/working/stage1_train && unzip -n -q /kaggle/input/data-science-bowl-2018/stage1_train.zip -d /kaggle/working/stage1_train

# !mkdir -p /kaggle/working/stage1_test  && unzip -n -q /kaggle/input/data-science-bowl-2018/stage1_test.zip -d /kaggle/working/stage1_test

!wget -Nq https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip; unzip -n -q ngrok-stable-linux-amd64.zip -d /kaggle/working
## Start Up TensorBoard and ngrok

# Start TensorBoard, `ngrok` opens a tunnel to our Kaggle session to connect to TensorBoard

# A clickable URL appears in the output below



pool = multiprocessing.Pool(processes = 10)

results_of_processes = [pool.apply_async(os.system, args=(cmd, ), callback = None )

                        for cmd in [

                        f"tensorboard --logdir {TB_LOGDIR} --host 0.0.0.0 --port 6006 &",

                        "./ngrok http 6006 &"

                        ]]

time.sleep(2)

!curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
def generate_combined_mask(home_dir):

    path_list = [d.path for d in os.scandir(home_dir) if d.is_dir()]

    for img_path in tqdm(iterable=path_list, desc="Processing mask files"):

        searchpath = os.path.join(img_path, "masks", "*.png")

        masklist = glob.glob(searchpath)

        firstmask = Image.open(masklist[0], 'r')

        img_w, img_h = firstmask.size

        background_image = Image.new('L', (img_w, img_h), 0)

        for m in masklist:

            background_image = ImageChops.lighter(background_image, Image.open(m))

        new_mask_dir = os.path.join(img_path, "masks2")

        os.makedirs(new_mask_dir, exist_ok=True)

        new_mask_path = os.path.join(new_mask_dir, "newmask.png")

        background_image.save(new_mask_path)

    image_list = [f.path for i in path_list for f in os.scandir(os.path.join(i, "images")) if f.is_file()]

    masks_list = [f.path for i in path_list for f in os.scandir(os.path.join(i, "masks2")) if f.is_file()]

    return image_list, masks_list



class Nucleii(keras.utils.Sequence):

    # Helper to turn images into Sequence object for TF model

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths, img_dtype="float32", tgt_dtype="uint8"):

        self.batch_size = batch_size

        self.img_size = img_size

        self.input_img_paths = input_img_paths

        self.target_img_paths = target_img_paths

        self.img_dtype = img_dtype

        self.tgt_dtype = tgt_dtype



    def __len__(self):

        return len(self.target_img_paths) // self.batch_size



    def __getitem__(self, idx):

        # Returns tuple (input, target) correspond to batch #idx.

        i = idx * self.batch_size

        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]

        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype=self.img_dtype)

        for j, path in enumerate(batch_input_img_paths):

            img = load_img(path, target_size=self.img_size)

            x[j] = img

        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype=self.tgt_dtype)

        for j, path in enumerate(batch_target_img_paths):

            img = load_img(path, target_size=self.img_size, color_mode="grayscale")

            tgt_array = np.array(img) / 255

            y[j] = np.expand_dims(tgt_array, 2)

        return x, y



def image_and_mask_generator(image_list, masks_list, generator_args, image_size, batch_size):

    generator_list = []

    seed = 1

    colormode = {0:"rgb", 1:"grayscale"}

    for i, j in enumerate([image_list, masks_list]):

        dtype = ("uint8" if i==1 else None) # for mask dtype

        generator_args["dtype"] = dtype

        datagen = ImageDataGenerator(**generator_args)

        generator = datagen.flow_from_dataframe(

        dataframe=pd.DataFrame(j),

        directory=None,

        x_col=0,

        target_size=image_size,

        color_mode=colormode[i],

        class_mode=None,

        batch_size=batch_size,

        seed=seed)

        generator_list.append(generator)

    return zip(*generator_list)

    

def get_model(img_size, num_classes):

    inputs = keras.Input(shape=img_size + (3,))



    ### [First half of the network: downsampling inputs] ###

    # Entry block

    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)



    previous_block_activation = x  # Set aside residual



    # Blocks 1, 2, 3 are identical apart from the feature depth.

    for filters in [64, 128, 256]:

        x = layers.Activation("relu")(x)

        x = layers.SeparableConv2D(filters, 3, padding="same")(x)

        x = layers.BatchNormalization()(x)



        x = layers.Activation("relu")(x)

        x = layers.SeparableConv2D(filters, 3, padding="same")(x)

        x = layers.BatchNormalization()(x)



        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)



        # Project residual

        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(

            previous_block_activation

        )

        x = layers.add([x, residual])  # Add back residual

        previous_block_activation = x  # Set aside next residual



    ### [Second half of the network: upsampling inputs] ###



    previous_block_activation = x  # Set aside residual



    for filters in [256, 128, 64, 32]:

        x = layers.Activation("relu")(x)

        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)

        x = layers.BatchNormalization()(x)



        x = layers.Activation("relu")(x)

        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)

        x = layers.BatchNormalization()(x)



        x = layers.UpSampling2D(2)(x)



        # Project residual

        residual = layers.UpSampling2D(2)(previous_block_activation)

        residual = layers.Conv2D(filters, 1, padding="same")(residual)

        x = layers.add([x, residual])  # Add back residual

        previous_block_activation = x  # Set aside next residual



    # Add a per-pixel classification layer

    outputs = layers.Conv2D(num_classes, 3, activation="sigmoid", padding="same")(x)



    # Define the model

    model = keras.Model(inputs, outputs)

    return model



# This is a bug fix for the Keras MeanIoU metric 

# From https://stackoverflow.com/questions/61824470/dimensions-mismatch-error-when-using-tf-metrics-meaniou-with-sparsecategorical

class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):

  def __init__(self,

               y_true=None,

               y_pred=None,

               num_classes=None,

               name=None,

               dtype=None):

    super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)



  def update_state(self, y_true, y_pred, sample_weight=None):

    y_pred = tf.math.argmax(y_pred, axis=-1)

    return super().update_state(y_true, y_pred, sample_weight)
image_list, masks_list = generate_combined_mask(TRAIN_HOME_DIR)
##--> DATA SETUP: Split our image and mask paths into training/validation sets

val_samples = int(len(image_list) * (1 - TRAIN_SPLIT))

random.Random(1337).shuffle(image_list)

random.Random(1337).shuffle(masks_list)

train_image_list = image_list[:-val_samples]

train_masks_list = masks_list[:-val_samples]

val_image_list = image_list[-val_samples:]

val_masks_list = masks_list[-val_samples:]



# NO AUGMENTATION: Instantiate data Sequence objects for each split

train_seq = Nucleii(BATCH_SIZE, IMG_SIZE, train_image_list, train_masks_list)

val_seq = Nucleii(BATCH_SIZE, IMG_SIZE, val_image_list, val_masks_list)



# ADD AUGMENTATION: Create generator objects that can create infinite augmented images from base dataset

train_data_gen_args  =  dict(rescale=1./255,

                        shear_range=0.5,

                        rotation_range=50,

                        zoom_range=0.2,

                        width_shift_range=0.2,

                        height_shift_range=0.2,

                        fill_mode='reflect'

                        )

                          

val_data_gen_args = dict(rescale=1./255,

                        )



train_gen_aug = image_and_mask_generator(train_image_list, train_masks_list, train_data_gen_args, IMG_SIZE, BATCH_SIZE)

val_gen_aug = image_and_mask_generator(val_image_list, val_masks_list, val_data_gen_args, IMG_SIZE, BATCH_SIZE)
# Free up RAM in case the model definition cells were run multiple times

keras.backend.clear_session()



# Set up logging directory, callback functions, and metrics (for TensorBoard)

log_dir = os.path.join(TB_LOGDIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

callbacks = [

    keras.callbacks.ModelCheckpoint(MODEL_CHECKPOINT_NAME, save_best_only=True),

    keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=5, write_graph=True, write_images=True, embeddings_freq=5),

    tf.keras.callbacks.EarlyStopping(monitor="val_loss", verbose=1, patience=7)

]

metrics = [

    UpdatedMeanIoU(num_classes=NUM_CLASSES), # bug fix for tf.keras.metrics.MeanIoU, see above

]



# Set up model layers with get_model(), choose optimizer and loss function in compile step

model = get_model(IMG_SIZE, NUM_CLASSES)

#model.summary()

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=metrics)



# Train the model, doing validation at the end of each epoch

if(AUGMENTATION_ON):

    # WITH AUGMENTATION

    print(f"Beginning training for {EPOCHS} epochs, batch size: {BATCH_SIZE}, augmentation: {AUGMENTATION_ON}")

    model.fit(train_gen_aug, steps_per_epoch=1500//BATCH_SIZE, epochs=EPOCHS, validation_data=val_seq, callbacks=callbacks)

else:

    # WITHOUT AUGMENTATION

    print(f"Beginning training for {EPOCHS} epochs, batch size: {BATCH_SIZE}, augmentation: {AUGMENTATION_ON}")

    model.fit(train_seq, epochs=EPOCHS, validation_data=val_seq, callbacks=callbacks)
val_preds = model.predict(val_seq)
# Randomly select four images from validation data

# Display input image, input mask, and predicted mask

def plot_images(image_list, mask_list, predictions, sample_size):

    i = np.random.randint(0, len(image_list)-1, size=sample_size)

    f, axarr = plt.subplots(sample_size//2, 6, figsize=(24,int(sample_size*2)))

    axarr = axarr.flatten()

    _ = [a.set_axis_off() for a in axarr.ravel()]

    for x in range(sample_size):

        axarr[3*x].imshow(load_img(image_list[i[x]]))

        axarr[3*x].set_title("Original Image")

        axarr[3*x+1].imshow(load_img(mask_list[i[x]]), cmap="gray")

        axarr[3*x+1].set_title("Ground Truth Mask")

        pred_mask = np.argmax(predictions[i[x]], axis=-1)

        pred_mask = np.expand_dims(pred_mask, axis=-1)

        pred_img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(pred_mask))

        axarr[3*x+2].imshow(pred_img, cmap="gray")

        axarr[3*x+2].set_title("Predicted Mask")



VAL_IMG_SAMPLE = 6

plot_images(val_image_list, val_masks_list, val_preds, VAL_IMG_SAMPLE)
# !rm -f /kaggle/working/tensorboardlogs.tar; cd /kaggle/working; tar czf tensorboardlogs.tar.gz tensorboardlogs

# FileLink(r'tensorboardlogs.tar.gz')
# TESTING CODE

def plot_images2(image_list, mask_list, predictions, sample_size):

    i = np.random.randint(0, len(image_list)-1, size=sample_size)

    f, axarr = plt.subplots(sample_size//2, 6, figsize=(24,int(sample_size*2)))

    axarr = axarr.flatten()

    _ = [a.set_axis_off() for a in axarr.ravel()]

    for x in range(sample_size):

        axarr[3*x].imshow((image_list[i[x]]))

        axarr[3*x].set_title("Original Image")

        axarr[3*x+1].imshow((mask_list[i[x]][:,:,0]), cmap="gray")

        axarr[3*x+1].set_title("Ground Truth Mask")

        pred_mask = np.argmax(predictions[i[x]], axis=-1)

        pred_mask = np.expand_dims(pred_mask, axis=-1)

        pred_img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(pred_mask))

        axarr[3*x+2].imshow(pred_img, cmap="gray")

        axarr[3*x+2].set_title("Predicted Mask")



# aug_batch = next(train_gen_aug)

# t_img = aug_batch[0]

# t_msk = aug_batch[1]

# plot_images2(t_img, t_msk, model.predict(aug_batch), 6)

# plot_images2(t_img, t_msk, model.predict(aug_batch), 6)