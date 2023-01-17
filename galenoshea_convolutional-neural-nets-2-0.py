import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.metrics import Accuracy, AUC, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, SeparableConv2D, MaxPooling2D, Dense, BatchNormalization, Flatten, Activation, Dropout, Add
path = '../input/chest-xray-pneumonia/chest_xray'
train_dir = f'{path}/train'
val_dir = f'{path}/val'
test_dir = f'{path}/test'
len_normal = len(os.listdir(f'{train_dir}/NORMAL'))
len_pneumonia = len(os.listdir(f'{train_dir}/PNEUMONIA'))
percent_normal = len_normal / (len_normal + len_pneumonia)
percent_pneumonia = len_pneumonia / (len_normal + len_pneumonia)

x = ["Normal", "Pneumonia"]
y = [len_normal, len_pneumonia]
plt.title("Class Distributions")
plt.ylabel("Number of x-rays")
plt.bar(x, y)

print(f'Normal x-rays: {len_normal} ({percent_normal:.2}%) | Pneumonia x-rays: {len_pneumonia} ({percent_pneumonia:.2f}%)')
width = height = 64 # used for resizing the image later
channels = 1 # x-rays have 1 colour channel since they're not rgb
input_size = (width, height)
input_shape = (width, height, channels)

colour_mode = 'grayscale' # x-rays are grayscale
class_mode ='binary' # because there are only 2 classes
seed = 42

epochs = 50 
learning_rate = 0.001
batch_size = 32
val_size = 0.05 # 95% for training & 5% for validation
class_names = os.listdir(train_dir)

weight_normal = 1 / percent_normal
weight_pneumonia = 1 / percent_pneumonia
class_weight = {0: weight_normal, 1: weight_pneumonia} #we create a inversely proportional ratio to negate the imbalanced classes

train_gen = ImageDataGenerator(
    rescale = 1./255, #Normalize the pixels to be in a range (0-1)
    rotation_range = 10,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    validation_split = val_size # This is for splitting training and validation up
)
train_flow = train_gen.flow_from_directory(
    directory = train_dir, 
    target_size = input_size, # We pass the designated size we want to resize our images too so it's lighter on training
    class_mode = class_mode,
    batch_size = batch_size,
    color_mode = colour_mode,
    shuffle = True,
    seed = seed,
    subset = 'training' #Special key to let Keras know this subset gets (1 - val_size) samples
)

val_flow = train_gen.flow_from_directory(
    directory = train_dir, # We're using the same directory as train_flow since we are splitting the directory
    target_size = input_size,
    class_mode = class_mode,
    batch_size = batch_size,
    color_mode = colour_mode,
    shuffle = True,
    seed = seed,
    subset = 'validation'
)
test_gen = ImageDataGenerator(rescale = 1./255) # Testing should not augment data since we'd like to see how it performs on new data

test_flow = test_gen.flow_from_directory(
    directory = test_dir,
    target_size = input_size,
    class_mode = class_mode,
    batch_size = batch_size,
    color_mode = colour_mode,
    seed = seed
)
def show_batch(image_batch, label_batch):
    plt.figure(figsize=input_size)  
    for n in range(16):
        ax = plt.subplot(4,4,n+1)
        img = image_batch[n]
        plt.imshow(np.squeeze(img), cmap='gray')
        class_name = class_names[int(label_batch[n])]
        plt.title(class_name, fontsize=50)
        plt.axis('off')
image_batch, label_batch = next(train_flow)
show_batch(image_batch, label_batch)
image_batch, label_batch = next(test_flow)
show_batch(image_batch, label_batch)
xray_input = Input(shape = input_shape)

x = SeparableConv2D(filters = 32, kernel_size = 3)(xray_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
# x = MaxPooling(2)(x) add pooling dude

x = SeparableConv2D(filters = 32, kernel_size = 3)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

#skip conenction 
b1_skip = x
b1_skip = SeparableConv2D(filters = 64, kernel_size = 5)(b1_skip) #convolve to remerge later
b1_skip = BatchNormalization()(b1_skip)

x = SeparableConv2D(filters = 64, kernel_size = 3)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = SeparableConv2D(filters = 64, kernel_size = 3)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Add()([x, b1_skip])

b2_skip = x
b2_skip = SeparableConv2D(filters = 128, kernel_size = 5)(b2_skip)
b2_skip = BatchNormalization()(b2_skip)

x = SeparableConv2D(filters = 128, kernel_size = 3)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = SeparableConv2D(filters = 128, kernel_size = 3)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Add()([x, b2_skip])

b3_skip = x
b3_skip = SeparableConv2D(filters = 256, kernel_size = 5)(b3_skip)
b3_skip = BatchNormalization()(b3_skip)

x = SeparableConv2D(filters = 256, kernel_size = 3)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = SeparableConv2D(filters = 256, kernel_size = 3)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Add()([x, b3_skip])

x = MaxPooling2D(pool_size=2)(x)

x = Flatten()(x)

x = Dense(units = 1024)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Dropout(0.1)(x)

x = Dense(units = 1024)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

output = Dense(units = 1, activation = 'sigmoid')(x)

model = Model(xray_input, output, name = 'xrayz')

model.compile(
    loss = BinaryCrossentropy(),
    optimizer = Adam(learning_rate=learning_rate),
    metrics = [
        'accuracy',
        AUC(name='auc'),
        TruePositives(name='tp'), 
        FalsePositives(name='fp'), 
        TrueNegatives(name='tn'), 
        FalseNegatives(name='fn')
    ]
)

model.summary()
plot_model(model, f'{model.name}.png')
early_stop = EarlyStopping(patience=3)

callbacks = [early_stop]
history = model.fit(
    train_flow,
    epochs = epochs,
    validation_data = val_flow,
    callbacks = callbacks,
    class_weight = class_weight
)
print(history.history)
tp = int(history.history['val_tp'][-1])
fp = int(history.history['val_fp'][-1])
tn = int(history.history['val_tn'][-1])
fn = int(history.history['val_fn'][-1])

acc = history.history['val_accuracy'][-1]
auc = history.history['val_auc'][-1]
loss = history.history['val_loss'][-1]

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
recall =  tp / (tp + fn)
#accuracy = acc (tp + tn) / (tp + fp + tn + fn)

print(f'Sensitivity: {sensitivity}')
print(f'Specificity: {specificity}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Accuracy: {acc}')
print(f'AUC: {auc}')

data = np.array([[tp, fp],
                   [fn, tn]])

labels = [[f'True Positive\n {tp}', f'False Positive\n {fp}'], 
          [f'False Negative\n {fn}', f'True Negative\n {tn}']]

heatmap = sns.heatmap(data, annot=labels, fmt='', cmap='Blues')
heatmap.set_title('Confusion Matrix')

heatmap.set_xticklabels(['1', '0'])
heatmap.set_yticklabels(['1', '0'])

plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="Validation loss")
ax1.legend(loc='best', shadow=True)

ax2.plot(history.history['tp'], color='b', label="Training True Positive")
ax2.plot(history.history['val_tp'], color='r',label="Validation True Positive")
ax2.legend(loc='best', shadow=True)

#plt.tight_layout()
plt.show()
score = model.evaluate(test_flow, batch_size=batch_size)

print('Test Loss:', score[0]) 
print('Test Accuracy:', score[1])
