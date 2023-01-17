import os

import numpy as np

import pandas as pd
print(os.listdir('../input/10-monkey-species/training/training'))

print(os.listdir('../input/10-monkey-species/validation/validation'))
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, Input, GlobalAveragePooling2D

from tensorflow.keras.optimizers import RMSprop, Adam, SGD

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from tensorflow.keras.applications.xception import Xception

from tensorflow.keras.models import Model, model_from_json, load_model

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.utils import class_weight
label_map = {}

with open('../input/10-monkey-species/monkey_labels.txt', 'r') as f:

    for line in f:

        try:

            label, _, name, _, _ = line.split(',')

            label = int(label.split('n')[1].strip())

            name = name.strip()

            label_map[label] = name

        except (ValueError, IndexError): 

            pass



label_map
train_gen = ImageDataGenerator(

    rescale=1./255,

    zoom_range=0.2,

    horizontal_flip=True,

    vertical_flip=True,

    shear_range=0.2,

    fill_mode='nearest',

    rotation_range=40)
train_generator = train_gen.flow_from_directory(

    '../input/10-monkey-species/training/training/',

    target_size=(250,250),

    color_mode='rgb',

    class_mode='categorical',

    batch_size=128,

    shuffle=True,

    seed=10)
test_gen = ImageDataGenerator(rescale=1./255)
def create_test_genearor(test_gen, to_shuffle=True):

    test_generator = test_gen.flow_from_directory(

        '../input/10-monkey-species/validation/validation/',

        target_size=(250,250),

        color_mode='rgb',

        class_mode='categorical',

        batch_size=16,

        shuffle=to_shuffle)

    return test_generator



test_generator = create_test_genearor(test_gen)
test_batch = next(test_generator)

print('dimentions of array returned by generator:- ', len(test_batch))

print('size of image array:- ', test_batch[0].shape)

print('labels array:-\n', test_batch[1])



true_labels = test_batch[1].argmax(axis=1)
def plot_images(test_batch, true_labels, predicted_labels=None):

    plt.figure(figsize=(15,18))

    for no, image in enumerate(test_batch[0]):

        plt.subplot(len(test_batch[0])/4, 4, no+1)

        if predicted_labels is None:

            plt.title(label_map[true_labels[no]])

        else:

            plt.title('true:{}\npred:{}'.format(label_map[true_labels[no]],

                                                     label_map[predicted_labels[no]]))

        plt.imshow(test_batch[0][no])
plot_images(test_batch, true_labels)
test_generator = create_test_genearor(test_gen)

print(test_generator.batch_index) 
plt.figure(figsize=(8,4))

sns.countplot(train_generator.classes)
classweights = class_weight.compute_class_weight(

    'balanced', 

    np.unique(train_generator.classes), 

    train_generator.classes)



classweights
base = Xception(weights='imagenet', 

                include_top=False, 

                input_shape=(250,250,3))
print(base.layers[0])

print(base.layers[0].name)

print(base.layers[0].input_shape)

print(base.layers[0] == base._layers[0])
new_gap1 = GlobalAveragePooling2D(data_format='channels_last', 

                                  name='new_gap1_layer')(base.output)

new_dense1 = Dense(units=10, 

                   activation='softmax', 

                   name='new_dense1_layer')(new_gap1)
model = Model(inputs=base.input, outputs=new_dense1)
model.summary()
model.layers[-8:]
for layer in model.layers[:-8]:

    layer.trainable = False
model.summary()
model.compile(optimizer=RMSprop(),

              loss='categorical_crossentropy',

              metrics=['accuracy'])
class CustomCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if (logs['accuracy'] >= 0.97) and (logs['val_accuracy'] >= 0.97):

            self.model.stop_training = True

            print('\n\n Both accuracies >= 97% on epoch {}\n'.format(epoch))



stop_training_on_97_acc = CustomCallback()
# filename = 'monkey_classification_model_epoch_{epoch:02d}_trainacc_{accuracy:.2f}_testacc_{val_accuracy:.2f}.hdf5'

# mc_cb = ModelCheckpoint(filepath=filename, 

#                         monitor='accuracy', 

#                         verbose=1, 

#                         save_best_only=True, 

#                         save_weights_only=False, 

#                         mode='auto', 

#                         period=1)
tb_cb = TensorBoard(log_dir='tf_logs', 

                    histogram_freq=1, 

                    write_graph=True, 

                    update_freq='epoch')
history = model.fit_generator(train_generator, 

                              validation_data=test_generator, 

                              epochs=30,

                              callbacks=[tb_cb, stop_training_on_97_acc],

                              class_weight=classweights)
history.history
plt.figure(figsize=[12,4])

plt.subplot(1,2,1)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'])



plt.subplot(1,2,2)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'])

plt.show()
test_generator = create_test_genearor(test_gen, to_shuffle=False)
metrics = model.evaluate_generator(test_generator)

print('test data loss :- {}'.format(metrics[0]))

print('test data accuracy :- {}'.format(metrics[1]))
probabilities = model.predict_generator(test_generator)

predicted_labels = probabilities.argmax(axis=1)

true_labels = test_generator.classes
plt.figure(figsize=[8, 6])

sns.heatmap(confusion_matrix(true_labels, predicted_labels), annot=True, cmap='ocean_r',

            xticklabels=list(label_map.values()), yticklabels=list(label_map.values()))
test_generator = create_test_genearor(test_gen, to_shuffle=True)
batch = next(test_generator)

probabilities = model.predict_on_batch(batch)

probabilities.shape
predicted_labels = probabilities.numpy().argmax(axis=1)

true_labels = batch[1].argmax(axis=1)
plot_images(batch, true_labels, predicted_labels)
plt.imshow(load_img('../input/10-monkey-species/validation/validation/n4/n408.jpg', target_size=(250,250)))
def preprocess_image(path):

    img = load_img(path, target_size=(250,250))

    img = img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = img * 1./ 255

    return img
img = preprocess_image('../input/10-monkey-species/validation/validation/n4/n408.jpg')

prob = model.predict(img)

print(prob)

print(prob.argmax())

print(label_map[prob.argmax()])
img = preprocess_image('../input/10-monkey-species/validation/validation/n4/n408.jpg')
custOutput = [layer.output for layer in model.layers if 'conv2d' in layer.name]

custOutput
customOutputModel = Model(inputs=model.input, outputs=custOutput)
feature_map = customOutputModel.predict(img)



for i in range(len(feature_map)):

    print(feature_map[i].shape)
channel_no_to_display = [1,2,3,4,5,-5,-4,-3,-2,-1]

n_features = len(channel_no_to_display)
for fm in feature_map:

    size = fm.shape[1]

    display_grid = np.zeros((size, size * n_features))

    print(display_grid.shape)



    for i in range(n_features):

        # Postprocess the feature to make it visually palatable

        no = channel_no_to_display[i]

        x = fm[0, :, :, no]

        x -= x.mean()

        x /= x.std()

        x *= 64

        x += 128

        x = np.clip(x, 0, 255).astype('uint8')

        # We'll tile each filter into this big horizontal grid

        display_grid[:, i * size : (i + 1) * size] = x



    scale = 20. / n_features

    plt.figure(figsize=(scale * n_features, scale))

    #plt.title(layer_name)

    plt.grid(False)

    plt.imshow(display_grid, aspect='auto', cmap='viridis')
custOutput2 = [layer.output for layer in model.layers if 'block14_sepconv1_act' in layer.name or 'block14_sepconv2_act' in layer.name]

custOutput2
custModel2 = Model(inputs=model.input, outputs=custOutput2) 
feature_maps2 = custModel2.predict(img)



for i in range(len(feature_maps2)):

    print(feature_maps2[i].shape)
channel_no_to_display = [1,2,3,4,5,-5,-4,-3,-2,-1]

n_features = len(channel_no_to_display)



for fm in feature_maps2:

    size = fm.shape[1]

    display_grid = np.zeros((size, size * n_features))

    print(display_grid.shape)



    for i in range(n_features):

        # Postprocess the feature to make it visually palatable

        no = channel_no_to_display[i]

        x = fm[0, :, :, no]

        x -= x.mean()

        x /= x.std()

        #x *= 64

        #x += 128

        x = np.clip(x, 0, 255).astype('uint8')

        # We'll tile each filter into this big horizontal grid

        display_grid[:, i * size : (i + 1) * size] = x



    scale = 20. / n_features

    plt.figure(figsize=(scale * n_features, scale))

    #plt.title(layer_name)

    plt.grid(False)

    plt.imshow(display_grid, aspect='auto', cmap='viridis')
model_dir = '../working/model/'

if not os.path.exists(model_dir):

    os.makedirs(model_dir)
model.save(model_dir + 'monkey_species_classification_model.hdf5')