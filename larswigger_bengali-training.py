!pip install prefetch_generator

from prefetch_generator import background
import numpy as np

import pandas as pd

import tensorflow as tf

import gc

import os

# define constants

ORIGINAL_HEIGHT = 137

ORIGINAL_WIDTH = 236

PROCESSED_HEIGHT = 128

PROCESSED_WIDTH = 128

#for generator batching

ROWS_PER_FILE = 20084+1

NUM_FILES = 200840 // ROWS_PER_FILE

VALID_SIZE = 200840 % ROWS_PER_FILE

"""Set manually depending on chosen preprocessing"""

TRAIN_BATCH_SIZE = 195

VALID_BATCH_SIZE = 365

"""End of choose manually"""

assert ROWS_PER_FILE % TRAIN_BATCH_SIZE == 0, "TRAIN_BATCH_SIZE is not a divisor of ROWS_PER_FILE"

assert VALID_SIZE % VALID_BATCH_SIZE == 0, "VALID_BATCH_SIZE is not a divisor of VALID_SIZE"

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
## Grid Mask

# code takesn from https://www.kaggle.com/haqishen/gridmask



import albumentations

from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform

from albumentations.augmentations import functional as F



class GridMask(DualTransform):

    """GridMask augmentation for image classification and object detection.



    Args:

        num_grid (int): number of grid in a row or column.

        fill_value (int, float, lisf of int, list of float): value for dropped pixels.

        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int

            an angle is picked from (-rotate, rotate). Default: (-90, 90)

        mode (int):

            0 - cropout a quarter of the square of each grid (left top)

            1 - reserve a quarter of the square of each grid (left top)

            2 - cropout 2 quarter of the square of each grid (left top & right bottom)



    Targets:

        image, mask



    Image types:

        uint8, float32



    Reference:

    |  https://arxiv.org/abs/2001.04086

    |  https://github.com/akuxcw/GridMask

    """



    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):

        super(GridMask, self).__init__(always_apply, p)

        if isinstance(num_grid, int):

            num_grid = (num_grid, num_grid)

        if isinstance(rotate, int):

            rotate = (-rotate, rotate)

        self.num_grid = num_grid

        self.fill_value = fill_value

        self.rotate = rotate

        self.mode = mode

        self.masks = None

        self.rand_h_max = []

        self.rand_w_max = []



    def init_masks(self, height, width):

        if self.masks is None:

            self.masks = []

            n_masks = self.num_grid[1] - self.num_grid[0] + 1

            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):

                grid_h = height / n_g

                grid_w = width / n_g

                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)

                for i in range(n_g + 1):

                    for j in range(n_g + 1):

                        this_mask[

                             int(i * grid_h) : int(i * grid_h + grid_h / 2),

                             int(j * grid_w) : int(j * grid_w + grid_w / 2)

                        ] = self.fill_value

                        if self.mode == 2:

                            this_mask[

                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),

                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)

                            ] = self.fill_value

                

                if self.mode == 1:

                    this_mask = 1 - this_mask



                self.masks.append(this_mask)

                self.rand_h_max.append(grid_h)

                self.rand_w_max.append(grid_w)



    def apply(self, image, mask, rand_h, rand_w, angle, **params):

        h, w = image.shape[:2]

        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask

        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask

        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)

        return image



    def get_params_dependent_on_targets(self, params):

        img = params['image']

        height, width = img.shape[:2]

        self.init_masks(height, width)



        mid = np.random.randint(len(self.masks))

        mask = self.masks[mid]

        rand_h = np.random.randint(self.rand_h_max[mid])

        rand_w = np.random.randint(self.rand_w_max[mid])

        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0



        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}



    @property

    def targets_as_params(self):

        return ['image']



    def get_transform_init_args_names(self):

        return ('num_grid', 'fill_value', 'rotate', 'mode')

    

gridMaskTransform = albumentations.Compose([

    GridMask(num_grid=3, rotate=15, p=1),

])
@background(max_prefetch=1)

def train_data_generator_function():

    while True:

        for file_index in range(NUM_FILES):

            X = np.load(f"/kaggle/input/bengali-preprocessing/processed_20085_128_{file_index}.npy").reshape(-1, PROCESSED_HEIGHT, PROCESSED_WIDTH, 1)

            root = np.load(f"/kaggle/input/bengali-preprocessing/root_20085_label_{file_index}.npy")

            vowel = np.load(f"/kaggle/input/bengali-preprocessing/vowel_20085_label_{file_index}.npy")

            consonant = np.load(f"/kaggle/input/bengali-preprocessing/consonant_20085_label_{file_index}.npy")

            for batch_index in range(ROWS_PER_FILE // TRAIN_BATCH_SIZE):             

                #yprocess images in batch range

                for i in range(batch_index*TRAIN_BATCH_SIZE,(batch_index+1)*TRAIN_BATCH_SIZE):

                        X[i] = gridMaskTransform(image=X[i])["image"]

                yield (X[batch_index*TRAIN_BATCH_SIZE:(batch_index+1)*TRAIN_BATCH_SIZE], 

                       [root[batch_index*TRAIN_BATCH_SIZE:(batch_index+1)*TRAIN_BATCH_SIZE],

                        vowel[batch_index*TRAIN_BATCH_SIZE:(batch_index+1)*TRAIN_BATCH_SIZE],

                        consonant[batch_index*TRAIN_BATCH_SIZE:(batch_index+1)*TRAIN_BATCH_SIZE]])              

            del X, root, vowel, consonant

            gc.collect()



train_data_generator = train_data_generator_function()        
#generator does not provide advantage for this one

X = np.load("/kaggle/input/bengali-preprocessing/processed_20085_128_valid.npy").reshape(-1, PROCESSED_HEIGHT, PROCESSED_WIDTH, 1)

root = np.load("/kaggle/input/bengali-preprocessing/root_20085_label_valid.npy")

vowel = np.load("/kaggle/input/bengali-preprocessing/vowel_20085_label_valid.npy")

consonant = np.load("/kaggle/input/bengali-preprocessing/consonant_20085_label_valid.npy")

valid_data = (X, [root, vowel, consonant])
inputs = tf.keras.layers.Input(shape=(128,128,1))

model = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)#rescaling before passing into main model

model = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(model)



resnet50 = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(128,128,3), pooling="max")(model)

resnet50.trainable = False

model = tf.keras.layers.Flatten()(resnet50)

model = tf.keras.layers.BatchNormalization()(model)

model = tf.keras.layers.Dropout(0.3)(model)

model = tf.keras.layers.Dense(256, activation="relu")(model)

model = tf.keras.layers.BatchNormalization()(model)

model = tf.keras.layers.Dropout(0.3)(model)



head_root = tf.keras.layers.Dense(168, activation="softmax", name="Root")(model)

head_vowel = tf.keras.layers.Dense(11, activation="softmax", name="Vowel")(model)

head_consonant = tf.keras.layers.Dense(7, activation="softmax", name="Consonant")(model)



model = tf.keras.Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
tf.keras.utils.plot_model(model, to_file='model.png')
learning_rate_reduction_root = tf.keras.callbacks.ReduceLROnPlateau(monitor='Root_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.00001)

learning_rate_reduction_vowel = tf.keras.callbacks.ReduceLROnPlateau(monitor='Vowel_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.00001)

learning_rate_reduction_consonant = tf.keras.callbacks.ReduceLROnPlateau(monitor='Consonant_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.00001)

backup_callback = tf.keras.callbacks.ModelCheckpoint(filepath="backup_{epoch}.h5",

                                                     save_weights_only=False,

                                                     period=5,

                                                     verbosity=1)



last_model_callback = tf.keras.callbacks.ModelCheckpoint(filepath="Last_Model.h5",

                                                        save_weights_only=False,

                                                        verbosity=1)
EPOCHS = 20 #for reuse later on

history = model.fit(train_data_generator,

                    validation_data=valid_data,

                    epochs=EPOCHS,

                    steps_per_epoch=NUM_FILES*(ROWS_PER_FILE // TRAIN_BATCH_SIZE),

                    #validation_steps=(VALID_SIZE // VALID_BATCH_SIZE),

                    callbacks=[learning_rate_reduction_root , learning_rate_reduction_vowel, learning_rate_reduction_consonant, backup_callback, last_model_callback])
hist_df = pd.DataFrame(history.history)

hist_df["Epoch"] = np.arange(len(hist_df))+1

hist_df.set_index("Epoch", inplace=True)

with open('history.csv', mode='w') as f:

    hist_df.to_csv(f)
hist_df
#plot loss

selection = [col for col in hist_df.columns if "loss" in col]

hist_df[selection].plot()
#plot accuracy

selection = [col for col in hist_df.columns if "accuracy" in col]

hist_df[selection].plot()
print("Freed after analysis: " + str(gc.collect()))