# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
base_dir = '/kaggle/input/100-bird-species/'

import os

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')
train_lstnames = os.listdir(train_dir)
valid_lstnames = os.listdir(validation_dir)
test_lstnames = os.listdir(test_dir)
train_species_dir = []
for idx, spec_name in enumerate(train_lstnames):
    spec_dir = os.path.join(train_dir, spec_name)
    train_species_dir.append(spec_dir)

valid_species_dir = []
for idx, spec_name in enumerate(valid_lstnames):
    spec_dir = os.path.join(validation_dir, spec_name)
    valid_species_dir.append(spec_dir)

test_species_dir = [] 
for idx, spec_name in enumerate(test_lstnames):
    spec_dir = os.path.join(test_dir, spec_name)
    test_species_dir.append(spec_dir)
train_species_dir
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
number_per_species = {}

for idx, train_spec_dir in enumerate(train_species_dir):
    num_spec = len(os.listdir(train_spec_dir))
    spec_name = train_lstnames[idx]
    number_per_species[spec_name] = num_spec
spec_number_df = pd.DataFrame(data=[number_per_species[i] for i in number_per_species.keys()], index=number_per_species.keys(), columns=['Quantity'])
spec_number_df
spec_number_df['Quantity'].sum()
spec_number_df.plot(kind = 'bar', x = None, y = 'Quantity', figsize = (50, 10), title = 'Number of bird species', fontsize = 10)
spec_number_df.describe()
def generate_random_bird(train_lstnames, number_random_species, number_random_per_kind, train_species_dir, df):
    species_idx = np.random.choice(np.arange(len(train_lstnames)), number_random_species, replace = False)

    next_bird = {}
    for idx in species_idx:
        label = train_lstnames[idx]
        lst_name = os.listdir(train_species_dir[idx])
        random_idxs = np.random.choice(int(df.loc[df.index.values[idx]].values), number_random_per_kind, replace = False)
        choose_bird = []
        for bird_idx in random_idxs:
            bird_name = os.path.join(train_species_dir[idx], lst_name[bird_idx])
            choose_bird.append(bird_name)
        next_bird[label] = choose_bird
        
    return next_bird
def plot_image(number_random_species, number_random_per_kind, image_size, next_bird, class_names = None, model  = None, get_prediction = False):
    fig = plt.gcf()

    nrows = number_random_species
    ncols = number_random_per_kind

    fig.set_size_inches(image_size*nrows, image_size*ncols)
    
    if not get_prediction:
        model = None
        class_names = None
    else:
        inverse_class_names = {}
        for k, v in class_names.items():
            inverse_class_names[int(v)] = k
            
#         print(inverse_class_names)
        
    count = 0
    for label, img_paths in next_bird.items():
        for img_path in img_paths:
            sb = plt.subplot(nrows, ncols, count + 1)
            img = mpimg.imread(img_path)
            sb.set_title(f'{label}', color = 'r')
            
            if get_prediction:
                from tensorflow.keras.preprocessing.image import img_to_array
                import cv2
                new_img = cv2.resize(img, (224, 244))
                new_img = img_to_array(img)
                new_img = np.expand_dims(img, axis = 0)
#                 print('Shape', new_img.shape)
                std_img = new_img / 255.0
                pred = model.predict(std_img)
                print('Prediction', np.argmax(pred))
                pred_name = inverse_class_names[int(np.argmax(pred))]
                sb.set_title(f'Predicted: {pred_name} \nGround True: {label}', color = 'r', fontsize = 10)
                
            sb.axis('Off')
            
            plt.imshow(img)
            count += 1
    plt.show()
import numpy as np
import matplotlib.image as mpimg
number_random_species = 5
number_random_per_kind = 4

next_bird = generate_random_bird(train_lstnames, number_random_species, number_random_per_kind, train_species_dir, spec_number_df)
plot_image(number_random_species, number_random_per_kind, 4, next_bird)
input_shape = (224, 224, 3)
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
model = InceptionV3(input_shape=input_shape, weights='imagenet', include_top=False)
model.summary()
for layer in model.layers:
    layer.trainable = False
    
# last_layer = model.get_layer('mixed8')

# last_output = last_layer.output
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, BatchNormalization, Activation, Dense

num_classes = len(train_lstnames)


x = GlobalAveragePooling2D()(model.output)

x = Dense(units = 512, kernel_initializer='he_normal')(x)
x = BatchNormalization(axis = -1)(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
# x = Dense(units = 1024, activation = 'elu', kernel_initializer='he_normal')(x)
# x = Dropout(0.4)(x)
x = Dense(units = num_classes, activation = 'softmax')(x)

final_model = Model(inputs = [model.input], outputs = [x])

final_model.summary()
from tensorflow.keras.optimizers import Adam

final_model.compile(loss = 'sparse_categorical_crossentropy', optimizer=Adam(lr = 0.2), metrics = ['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=10,
    zoom_range=0.2,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale = 1.0/255)

test_datagen = ImageDataGenerator(rescale = 1.0/255)

train_generator = train_datagen.flow_from_directory(
            train_dir,
            batch_size=512,
            class_mode='sparse',
            shuffle=True,
            target_size = (224, 224)
)

valid_generator = valid_datagen.flow_from_directory(
            validation_dir,
            batch_size=128,
            class_mode='sparse',
            shuffle=True,
            target_size = (224, 224)
)

test_generator = test_datagen.flow_from_directory(
            test_dir,
            batch_size=128,
            class_mode='sparse',
            shuffle=True,
            target_size = (224, 224)
)
import os
root_logdir = os.path.join('/kaggle/working', "my_logs")

def get_run_logdir():
  import time
  run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
  return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
from tensorflow.keras.callbacks import TensorBoard
tensorboard_cb = TensorBoard(run_logdir)
history = final_model.fit(train_generator, epochs = 5, validation_data=valid_generator, steps_per_epoch=27503/512, 
                                    validation_steps=1000/128, verbose = 1, callbacks=[tensorboard_cb])
trainableIdx = 0
for layer in final_model.layers:
    print(f'{layer.name} is trainable {layer.trainable}')
    if layer.trainable:
        trainableIdx += 1
for layer in model.layers[-28 : ]:
    layer.trainable = True
for layer in final_model.layers:
    print(f'{layer.name} is trainable {layer.trainable}')
from tensorflow.keras.optimizers import Adam

final_model.compile(loss = 'sparse_categorical_crossentropy', optimizer=Adam(lr = 0.001), metrics = ['accuracy'])
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0*0.1**(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(lr0=0.01, s = 10)
lr_scheduler = LearningRateScheduler(exponential_decay_fn)
model_checkpoint = ModelCheckpoint('my_checkpoint.h5', save_best_only=True)
early_stop = EarlyStopping(patience = 10, restore_best_weights=True)

class StopTraining(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if logs.get('val_accuracy') >= 0.95:
            print('Reach the desirable accuracy')
            self.model.stop_training = True
            
stopTrain = StopTraining()
%load_ext tensorboard
%tensorboard --logdir=./my_logs --port 6006
history = final_model.fit(train_generator, epochs = 20, validation_data=valid_generator, steps_per_epoch=27503/512, 
                                    validation_steps=1000/128, verbose = 1, callbacks=[lr_scheduler, model_checkpoint, early_stop, stopTrain, tensorboard_cb])
final_model.save('/kaggle/working/BirdNet.h5')
final_model.evaluate(test_generator)
class_names = train_generator.class_indices
class_names
number_per_species_test = {}

for idx, test_spec_dir in enumerate(test_species_dir):
    num_spec = len(os.listdir(test_spec_dir))
    spec_name = test_lstnames[idx]
    number_per_species_test[spec_name] = num_spec
    
spec_number_df_test = pd.DataFrame(data=[number_per_species_test[i] for i in number_per_species_test.keys()], index=number_per_species_test.keys(), columns=['Quantity'])

number_random_species = 5
number_random_per_kind = 3
image_size = 4

next_bird = generate_random_bird(test_lstnames, number_random_species, number_random_per_kind, test_species_dir, spec_number_df_test)

plot_image(number_random_species, number_random_per_kind, image_size, next_bird, class_names = class_names, model  = final_model, get_prediction = True)
import matplotlib.pyplot as plt
%matplotlib inline

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epoch = history.epoch

plt.plot(epoch, acc, label = 'Training accuracy', color = 'r')
plt.plot(epoch, val_acc, label = 'Validation accuracy', color = 'b')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()


plt.plot(epoch, loss, label = 'Training loss', color = 'r')
plt.plot(epoch, val_loss, label = 'Validation loss', color = 'b')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()
number_per_species_test = {}

for idx, test_spec_dir in enumerate(test_species_dir):
    num_spec = len(os.listdir(test_spec_dir))
    spec_name = test_lstnames[idx]
    number_per_species_test[spec_name] = num_spec
spec_number_df_test = pd.DataFrame(data=[number_per_species_test[i] for i in number_per_species_test.keys()], index=number_per_species_test.keys(), columns=['Quantity'])
number_random_species = 1
number_random_per_kind = 1

next_bird = generate_random_bird(test_lstnames, number_random_species, number_random_per_kind, test_species_dir, spec_number_df_test)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

img = image.load_img(list(next_bird.values())[0][0], target_size = (224, 224))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis = 0)

img_tensor /= 255.0

layer_outputs = [layer.output for layer in final_model.layers[: -6]]
activation_model = Model(inputs = final_model.input, outputs = layer_outputs)
activations = activation_model.predict(img_tensor)

layer_names = []
for layer in final_model.layers[-15: -6]:
    layer_names.append(layer.name)
    
image_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    
    size = layer_activation.shape[1]
    
    n_cols = n_features // image_per_row
    display_grid = np.zeros((size * n_cols, image_per_row *size))
    
    for col in range(n_cols):
        for row in range(image_per_row):
            channel_image = layer_activation[0, :, :, col*image_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col *size : (col +1)*size, row*size : (row+1)*size] = channel_image
    scale = 1./size
    plt.figure(figsize = (scale*display_grid.shape[1], scale*display_grid.shape[0]))
    
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect = 'auto', cmap = 'viridis')
!pip install tensorflowjs
!tensorflowjs_converter --input_format=keras  {'/kaggle/working/BirdNet.h5'} ./kaggle/working/
