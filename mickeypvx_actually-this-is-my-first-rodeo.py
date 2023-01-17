# Main imports, training data load, and label mapping

import pandas as pd

import numpy as np

import os

import pickle

import matplotlib.pyplot as plt

import tensorflow as tf

import keras.backend.tensorflow_backend as tfb

import seaborn as sbn



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MultiLabelBinarizer, minmax_scale



from keras import layers, regularizers, optimizers, callbacks

from keras import backend as K

from keras.utils import normalize

from keras.utils.generic_utils import get_custom_objects

from keras.preprocessing import image

from keras_preprocessing.image import Iterator, ImageDataGenerator

from keras.applications import ResNet50

from keras.models import Model, load_model

from keras.losses import binary_crossentropy



from functools import partial



base_path = '../input/human-protein-atlas-image-classification/'

model_path = '../input/actually-this-is-my-first-rodeo/'



# Label map provided

label_map = {0: 'Nucleoplasm',

             1: 'Nuclear membrane',

             2: 'Nucleoli',

             3: 'Nucleoli fibrillar center',

             4: 'Nuclear speckles',

             5: 'Nuclear bodies',

             6: 'Endoplasmic reticulum',

             7: 'Golgi apparatus',

             8: 'Peroxisomes',

             9: 'Endosomes',

             10: 'Lysosomes',

             11: 'Intermediate filaments',

             12: 'Actin filaments',

             13: 'Focal adhesion sites',

             14: 'Microtubules',

             15: 'Microtubule ends',

             16: 'Cytokinetic bridge',

             17: 'Mitotic spindle',

             18: 'Microtubule organizing center',

             19: 'Centrosome',

             20: 'Lipid droplets',

             21: 'Plasma membrane',

             22: 'Cell junctions',

             23: 'Mitochondria',

             24: 'Aggresome',

             25: 'Cytosol',

             26: 'Cytoplasmic bodies',

             27: 'Rods & rings'}



all_img_labels = pd.read_csv("../input/human-protein-atlas-image-classification/train.csv")

all_img_labels['Target'] = all_img_labels['Target'].apply(lambda x: set(map(int, x.split())))



print('Labels converted to sets of integers:\n\n', all_img_labels.head())
mlb = MultiLabelBinarizer()

all_img_bin = pd.DataFrame(data=mlb.fit_transform(all_img_labels['Target']),

                           index=all_img_labels['Id'],

                           columns=label_map.values())

print(all_img_bin.head())
# Balancing class weights

class_counts = all_img_bin.sum(axis=0)

class_weights = {class_idx: (class_counts.sum() / (len(class_counts) * class_counts[class_idx]))

                 for class_idx, class_name in enumerate(class_counts.index)}



print('Top 10 Label Weights\n')

print(pd.Series(class_weights).sort_values(ascending=False)[:10])
# Messing with getting all channels into one array

sample_id = all_img_bin.index.values[np.random.randint(len(all_img_bin))] # select random image ID from training set



def img_load(img_id, img_path, img_size):

    img_array = np.zeros(shape=img_size)



    for i, channel in enumerate(['_red.png', '_green.png', '_blue.png', '_yellow.png']):

        sample_img_path = os.path.join(img_path, '{}{}'.format(img_id, channel))

        sample_img_channel = image.load_img(sample_img_path, color_mode='grayscale',

                                           target_size=img_size[:2])

        img_array[:,:,i] = sample_img_channel



    img_array = image.img_to_array(img_array)

    img_array = normalize(img_array, axis=0)

    

    return img_array



sample_img = img_load(img_id=sample_id, img_path=os.path.join(base_path, 'train'),

                     img_size=(256, 256, 4))

colors = ['Reds', 'Greens', 'Blues', 'Oranges']

fig, axes = plt.subplots(1, 4, figsize=(20, 20))



print('Sample image channels separated:')



for idx in range(len(colors)):

    axes[idx].imshow(sample_img[:,:,idx], colors[idx])
# Subclassing the keras ImageDataGenerator to fit our needs



class ImageDataGenMod(ImageDataGenerator):

    

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        

    def flow_from_gen(self, x_set, data_path, y_set=None,

                     batch_size=32, target_size=(512, 512), color_mode='rgba',

                     data_format='channels_last', **kwargs):

        return SimpleImageGen(x_set, self, data_path, y_set, target_size, color_mode,

                              data_format, batch_size, **kwargs)





class SimpleImageGen(Iterator):

    

    def __init__(self, x_set, image_data_generator, data_path, y_set=None,

                 target_size=(512, 512), color_mode='rgba',

                 data_format='channels_last', batch_size=32,

                 shuffle=True, seed=None,

                 save_to_dir=None, save_prefix='', save_format='png',

                 subset=None, interpolation='nearest'):

        super().common_init(image_data_generator,

                            target_size,

                            color_mode,

                            data_format,

                            save_to_dir,

                            save_prefix,

                            save_format,

                            subset,

                            interpolation)

        self.x, self.y = x_set, y_set

        self.batch_size = batch_size

        self.target_size = target_size

        self.color_mode = color_mode

        self.samples = len(self.x)

        self.colors = ['_red.png', '_green.png', '_blue.png', '_yellow.png']

        self.data_path = data_path

        

        super().__init__(self.samples,

                        batch_size,

                        shuffle,

                        seed)

    

    def __len__(self):

        return int(np.ceil(len(self.x) / float(self.batch_size)))

    

    def _get_batches_of_transformed_samples(self, index_array):

        #print(index_array)

        batch_x = np.zeros((len(index_array),) + self.image_shape)

        if self.y is not None:

            batch_y = self.y[index_array]

            #print(batch_y)

        

        for n, idx in enumerate(index_array):

            img_id = self.x[idx]

            for i in range(len(self.color_mode)):

                img_path = os.path.join(self.data_path, img_id + self.colors[i])

                img_channel = image.load_img(img_path, color_mode='grayscale',

                                            target_size=self.target_size)

                batch_x[n, :, :, i] = img_channel

            

            batch_x[n] = image.img_to_array(batch_x[n])

            

            params = self.image_data_generator.get_random_transform(batch_x[n].shape)

            batch_x[n] = self.image_data_generator.apply_transform(batch_x[n], params)

            batch_x[n] = self.image_data_generator.standardize(batch_x[n])

        

        if self.y is not None:

            return batch_x, np.array(batch_y)

        else:

            return batch_x

        

    def next(self):

        with self.lock:

            index_array = next(self.index_generator)

        

        return self._get_batches_of_transformed_samples(index_array)





# Flow test...

data_gen_args = dict(preprocessing_function=partial(normalize, axis=0),

                     rotation_range=45,

                     width_shift_range=0.2,

                     height_shift_range=0.2,

                     shear_range=0.2,

                     zoom_range=0.2,

                     channel_shift_range=0.2,

                     brightness_range=(0.3, 1.0),

                     horizontal_flip=True,

                     vertical_flip=True,

                     fill_mode='constant',

                     cval=0)



test_gen = ImageDataGenMod(**data_gen_args)

test_data = test_gen.flow_from_gen(x_set=all_img_bin.index.values,

                                   data_path=os.path.join(base_path, 'train'),

                                   y_set=all_img_bin.values,

                                   batch_size=5)



test_batch = next(test_data)

print('Modded ImageDataGenerator test:\n')

print('Data batch shape: ' + str(test_batch[0].shape),

      'Labels batch shape: ' + str(test_batch[1].shape),

      sep='\n')



fig, axes = plt.subplots(1, 5, figsize=(20, 20))



for idx in range(len(test_batch[0])):

    axes[idx].imshow(test_batch[0][idx])
# Macro F1 loss/metrics functions

def f1_metric(y_true, y_pred):

    y_pred = K.round(y_pred)



    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)

    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)

    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    

    p = tp / (tp + fp + K.epsilon())

    r = tp / (tp + fn + K.epsilon())

    

    f1 = 2 * p * r / (p + r + K.epsilon())

    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)



def f1_loss(y_true, y_pred):

    

    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)

    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)

    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    

    p = tp / (tp + fp + K.epsilon())

    r = tp / (tp + fn + K.epsilon())

    

    f1 = 2 * p * r / (p + r + K.epsilon())

    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return 1 - K.mean(f1)



# Found this custom weighted binary crossentropy loss function here:

# https://stackoverflow.com/questions/42158866/neural-network-for-multi-label-classification-with-large-number-of-classes-outpu/47313183#47313183

POS_WEIGHT = 10 # Seems arbitrary, needs to be tuned



def weighted_binary_crossentropy(target, output):

    # transform back to logits

    _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)

    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)

    output = tf.log(output / (1 - output))

    

    # compute weighted loss

    loss = tf.nn.weighted_cross_entropy_with_logits(targets=target,

                                                    logits=output,

                                                    pos_weight=POS_WEIGHT)

    return tf.reduce_mean(loss, axis=-1)



def combo_loss(y_true, y_pred):

    f1_loss_comp = f1_loss(y_true, y_pred)

    bce_loss_comp = binary_crossentropy(y_true, y_pred)

    return f1_loss_comp + bce_loss_comp
# If I've already saved a model, load it, otherwise build a new one

test_shape = (256, 256, 4)

regs = regularizers.l1_l2(l1=0.001, l2=0.001)

opt = optimizers.Adam(lr=1e-6)



build_new = False



if not build_new:

    get_custom_objects().update({'weighted_binary_crossentropy': weighted_binary_crossentropy,

                                 'combo_loss': combo_loss,

                                 'f1_loss': f1_loss,

                                 'f1_metric': f1_metric})

    model = load_model(os.path.join(model_path, 'hpa_model.h5'))

    print('Model loaded...')

else:

    model_in = layers.Input(test_shape)



    # Basic model build...    

    ### 2x2 Features

    model_in2 = layers.SeparableConv2D(32, (2, 2), padding='same', kernel_regularizer=regs)(model_in)

    model_in2 = layers.BatchNormalization()(model_in2)

    model_in2 = layers.Activation('relu')(model_in2)

    model_in2 = layers.SeparableConv2D(32, (2, 2), padding='same', kernel_regularizer=regs)(model_in2)

    model_in2 = layers.BatchNormalization()(model_in2)

    model_in2 = layers.Activation('relu')(model_in2)

    model_in2 = layers.MaxPool2D((2, 2), name='maxpool_2')(model_in2)

    

    model_in2 = layers.SeparableConv2D(64, (2, 2), padding='same', kernel_regularizer=regs)(model_in2)

    model_in2 = layers.BatchNormalization()(model_in2)

    model_in2 = layers.Activation('relu')(model_in2)

    model_in2 = layers.MaxPool2D((2, 2))(model_in2)

    

    model_in2 = layers.SeparableConv2D(128, (2, 2), padding='same', kernel_regularizer=regs)(model_in2)

    model_in2 = layers.BatchNormalization()(model_in2)

    model_in2 = layers.Activation('relu')(model_in2)

    model_in2 = layers.MaxPool2D((2, 2))(model_in2)

    

    model_in2 = layers.SeparableConv2D(256, (2, 2), padding='same', kernel_regularizer=regs)(model_in2)

    model_in2 = layers.BatchNormalization()(model_in2)

    model_in2 = layers.Activation('relu')(model_in2)

    model_in2 = layers.MaxPool2D((2, 2))(model_in2)

    

    ### 4x4 Features

    model_in4 = layers.SeparableConv2D(32, (4, 4), padding='same', kernel_regularizer=regs)(model_in)

    model_in4 = layers.BatchNormalization()(model_in4)

    model_in4 = layers.Activation('relu')(model_in4)

    model_in4 = layers.SeparableConv2D(32, (4, 4), padding='same', kernel_regularizer=regs)(model_in4)

    model_in4 = layers.BatchNormalization()(model_in4)

    model_in4 = layers.Activation('relu')(model_in4)

    model_in4 = layers.MaxPool2D((2, 2), name='maxpool_4')(model_in4)

    

    model_in4 = layers.SeparableConv2D(64, (4, 4), padding='same', kernel_regularizer=regs)(model_in4)

    model_in4 = layers.BatchNormalization()(model_in4)

    model_in4 = layers.Activation('relu')(model_in4)

    model_in4 = layers.MaxPool2D((2, 2))(model_in4)

    

    model_in4 = layers.SeparableConv2D(128, (4, 4), padding='same', kernel_regularizer=regs)(model_in4)

    model_in4 = layers.BatchNormalization()(model_in4)

    model_in4 = layers.Activation('relu')(model_in4)

    model_in4 = layers.MaxPool2D((2, 2))(model_in4)

    

    model_in4 = layers.SeparableConv2D(256, (4, 4), padding='same', kernel_regularizer=regs)(model_in4)

    model_in4 = layers.BatchNormalization()(model_in4)

    model_in4 = layers.Activation('relu')(model_in4)

    model_in4 = layers.MaxPool2D((2, 2))(model_in4)

    

    ### 8x8 Features

    model_in8 = layers.SeparableConv2D(32, (8, 8), padding='same', kernel_regularizer=regs)(model_in)

    model_in8 = layers.BatchNormalization()(model_in8)

    model_in8 = layers.Activation('relu')(model_in8)

    model_in8 = layers.SeparableConv2D(32, (8, 8), padding='same', kernel_regularizer=regs)(model_in8)

    model_in8 = layers.BatchNormalization()(model_in8)

    model_in8 = layers.Activation('relu')(model_in8)

    model_in8 = layers.MaxPool2D((2, 2), name='maxpool_8')(model_in8)

    

    model_in8 = layers.SeparableConv2D(64, (8, 8), padding='same', kernel_regularizer=regs)(model_in8)

    model_in8 = layers.BatchNormalization()(model_in8)

    model_in8 = layers.Activation('relu')(model_in8)

    model_in8 = layers.MaxPool2D((2, 2))(model_in8)

    

    model_in8 = layers.SeparableConv2D(128, (8, 8), padding='same', kernel_regularizer=regs)(model_in8)

    model_in8 = layers.BatchNormalization()(model_in8)

    model_in8 = layers.Activation('relu')(model_in8)

    model_in8 = layers.MaxPool2D((2, 2))(model_in8)

    

    model_in8 = layers.SeparableConv2D(256, (8, 8), padding='same', kernel_regularizer=regs)(model_in8)

    model_in8 = layers.BatchNormalization()(model_in8)

    model_in8 = layers.Activation('relu')(model_in8)

    model_in8 = layers.MaxPool2D((2, 2))(model_in8)

    

    ### 16x16 Features

    model_in16 = layers.SeparableConv2D(32, (16, 16), padding='same', kernel_regularizer=regs)(model_in)

    model_in16 = layers.BatchNormalization()(model_in16)

    model_in16 = layers.Activation('relu')(model_in16)

    model_in16 = layers.SeparableConv2D(32, (16, 16), padding='same', kernel_regularizer=regs)(model_in16)

    model_in16 = layers.BatchNormalization()(model_in16)

    model_in16 = layers.Activation('relu')(model_in16)

    model_in16 = layers.MaxPool2D((2, 2), name='maxpool_16')(model_in16)

    

    model_in16 = layers.SeparableConv2D(64, (16, 16), padding='same', kernel_regularizer=regs)(model_in16)

    model_in16 = layers.BatchNormalization()(model_in16)

    model_in16 = layers.Activation('relu')(model_in16)

    model_in16 = layers.MaxPool2D((2, 2))(model_in16)

    

    model_in16 = layers.SeparableConv2D(128, (16, 16), padding='same', kernel_regularizer=regs)(model_in16)

    model_in16 = layers.BatchNormalization()(model_in16)

    model_in16 = layers.Activation('relu')(model_in16)

    model_in16 = layers.MaxPool2D((2, 2))(model_in16)

    

    model_in16 = layers.SeparableConv2D(256, (16, 16), padding='same', kernel_regularizer=regs)(model_in16)

    model_in16 = layers.BatchNormalization()(model_in16)

    model_in16 = layers.Activation('relu')(model_in16)

    model_in16 = layers.MaxPool2D((2, 2))(model_in16)



    # Concatenate

    model_body = layers.Concatenate()([model_in2,

                                      model_in4,

                                      model_in8,

                                      model_in16])

    

    model_body = layers.SeparableConv2D(1024, (3, 3), padding='same', kernel_regularizer=regs)(model_body)

    model_body = layers.BatchNormalization()(model_body)

    model_body = layers.Activation('relu')(model_body)

    model_body = layers.MaxPool2D((2, 2))(model_body)

    

    model_body = layers.Flatten()(model_body)



    model_out = layers.Dropout(0.5)(model_body)

    model_out = layers.Dense(256, kernel_regularizer=regs)(model_out)

    model_out = layers.BatchNormalization()(model_out)

    model_out = layers.Activation('relu')(model_out)

    model_out = layers.Dense(28, activation='sigmoid')(model_out)



    model = Model(model_in, model_out)



    model.compile(loss=combo_loss,

                 optimizer=opt,

                 metrics=[f1_metric, 'acc'])

    model.save('hpa_model.h5')

    print('Model built...')



model.summary()
# Split data into training and validation sets

train_val_split = 0.15

batch_size = 16



x_train, x_test, y_train, y_test = train_test_split(all_img_bin.index.values,

                                                   all_img_bin.values,

                                                   test_size=train_val_split,

                                                   random_state=0)



print('Training set shape: ', x_train.shape, y_train.shape)

print('Validation set shape: ', x_test.shape, y_test.shape)



train_datagen = ImageDataGenMod(**data_gen_args)

test_datagen = ImageDataGenMod(preprocessing_function=partial(normalize, axis=0))



train_gen = train_datagen.flow_from_gen(x_set=x_train,

                                        data_path=os.path.join(base_path, 'train'),

                                        y_set=y_train,

                                        target_size=test_shape[:2],

                                        batch_size=batch_size,

                                       color_mode='rgba')

val_gen = test_datagen.flow_from_gen(x_set=x_test,

                                     data_path=os.path.join(base_path, 'train'),

                                     y_set=y_test,

                                     target_size=test_shape[:2],

                                     batch_size=batch_size,

                                    color_mode='rgba')
# Train the model on augmented images, plot the loss and F1 over epochs

cbacks = [callbacks.ReduceLROnPlateau(monitor='val_loss',

                                      mode='min',

                                      factor=0.5,

                                      patience=1,

                                      verbose=1,

                                      cooldown=2,

                                      min_lr=1e-10),

         callbacks.ModelCheckpoint(filepath='hpa_model.h5',

                                   monitor='val_f1_metric',

                                   mode='max',

                                   save_best_only=True,

                                   verbose=1)]



if not build_new:

    with open(os.path.join(model_path, 'all_history.pickle'), 'rb') as pkl:

        all_history = pickle.load(pkl)

    print('History loaded')

else:

    all_history = {'metric': [],

                  'loss': [],

                  'val_metric': [],

                  'val_loss': []}

    print('History created')



history = model.fit_generator(generator=train_gen,

                              steps_per_epoch=len(train_gen),

                              epochs=4,

                              validation_data=val_gen,

                              validation_steps=len(val_gen),

                              class_weight=class_weights,

                              callbacks=cbacks)



all_history['metric'].extend(history.history['f1_metric'])

all_history['loss'].extend(history.history['loss'])

all_history['val_metric'].extend(history.history['val_f1_metric'])

all_history['val_loss'].extend(history.history['val_loss'])



with open('all_history.pickle', 'wb') as pkl:

    pickle.dump(all_history, pkl)



# Plot the model fitting

start_index = 40



metric = all_history['metric'][-start_index:]

val_metric = all_history['val_metric'][-start_index:]

loss = all_history['loss'][-start_index:]

val_loss = all_history['val_loss'][-start_index:]

epochs = range(1, len(metric) + 1)



fig, axes = plt.subplots(1, 2)

axes[0].plot(epochs, metric, 'bo', label='Training F1')

axes[0].plot(epochs, val_metric, 'b', label='Validation F1')

axes[0].legend()

axes[1].plot(epochs, loss, 'bo', label='Training Loss')

axes[1].plot(epochs, val_loss, 'b', label='Validation Loss')

axes[1].legend()
# Make predictions with the model and output to a submission file

test_ids = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))['Id']



test_preds = test_datagen.flow_from_gen(x_set=list(test_ids),

                                        data_path=os.path.join(base_path, 'test'),

                                        target_size=test_shape[:2],

                                        batch_size=batch_size,

                                        color_mode='rgba')

preds = model.predict_generator(test_preds,

                               steps=len(test_preds))

target = [' '.join(map(str, x)) for x in mlb.inverse_transform(np.round(preds))]



output = pd.DataFrame(data=list(zip(test_ids, target)),

                      columns=['Id','Predicted'])

print(output.head())



output.to_csv('submission.csv', index=False)
# What's the model 'seeing?'

# Pick random test image ID

view_id = test_ids[np.random.randint(len(test_ids))]

view_img = img_load(img_id=view_id, img_path=os.path.join(base_path, 'test'),

                   img_size=test_shape)



layer_depth = 4

images_per_row = 8



view_img_tensor = np.expand_dims(view_img, axis=0)



plt.imshow(view_img_tensor[0])

plt.title('Test Image (RGBA)')

plt.show()



# Get the outputs of the initial pooling layers

layer_outputs = [layer.output for layer in model.layers if 'maxpool' in layer.name][:layer_depth]

activation_model = Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(view_img_tensor)



# Looking at some layers and channels of the model

layer_names = [layer.name for layer in model.layers if 'maxpool' in layer.name][:layer_depth]



for layer_name, layer_activation in zip(layer_names, activations):

    n_features = layer_activation.shape[-1]

    

    size = layer_activation.shape[1]

    

    n_cols = n_features // images_per_row

    display_grid = np.zeros((size * n_cols, size * images_per_row))

    

    for col in range(n_cols):

        for row in range(images_per_row):

            channel_img = layer_activation[0, :, :, col * images_per_row + row]

            

            display_grid[col * size : (col + 1) * size,

                        row * size : (row + 1) * size] = channel_img

    

    scale = 2.0 / size

    plt.figure(figsize=(scale * display_grid.shape[1],

                       scale * display_grid.shape[0]))

    plt.title(layer_name)

    plt.grid(False)

    plt.imshow(display_grid, aspect='auto', cmap='viridis')
# Are the predictions reasonable to the training data?

fig, axes = plt.subplots(1, 2, figsize=(20, 7.5), sharey=True)



sbn.barplot(data=np.round(preds), ax=axes[0]).set_title('Label Distribution (Predicted)')

sbn.barplot(data=all_img_bin.values, ax=axes[1]).set_title('Label Distribution (Training)')



plt.show()



num_pred, counts_pred = np.unique(np.sum(np.round(preds), axis=1), return_counts=True)

num_train, counts_train = np.unique(np.sum(all_img_bin.values, axis=1), return_counts=True)



fig, axes = plt.subplots(1, 2, figsize=(20, 7.5), sharey=True)



sbn.barplot(x=num_pred, y=counts_pred, ax=axes[0]).set_title('Labels/Image (Predicted)')

sbn.barplot(x=num_train, y=counts_train, ax=axes[1]).set_title('Labels/Image (Training)')



plt.show()