import numpy as np 
import pandas as pd 
from glob import glob
%matplotlib inline
import matplotlib.pyplot as plt
import os
import sklearn
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Input
from keras.layers import Conv2D, Activation, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input, ResNet50
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/data/Data_Entry_2017.csv')
image_path = {os.path.basename(x): x for x in 
                   glob(os.path.join('..', 'input/data/', 'images*', '*', '*.png'))}
data['path'] = data['Image Index'].map(image_path.get)
print(data.head(5))
# Total number of entries 
print(len(data))
# Total number of image_path
print(len(image_path))
# Top 10 labels in the dataset 
disease_counts = data.groupby('Finding Labels')['Image Index'].count().sort_values(ascending=False).iloc[:10]
print(disease_counts)

# plotting top 10 labels' count
plt.figure(figsize=(12,8))
plt.bar(np.arange(len(disease_counts))+0.5, disease_counts, tick_label=disease_counts.index)
plt.xticks(rotation=90)
# Create a sort of one hot encoding for each instances labels
# Remember multiclass multilabel classification
data['Finding Labels'] = data['Finding Labels'].map(lambda x: x.replace('No Finding','Nothing'))

from itertools import chain
labels = np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
print(labels)
for lbl in labels: 
    data[lbl] = data['Finding Labels'].map(lambda find: 1 if lbl in find else 0)
data['encoding'] = [[1 if l in lbl.split('|') else 0 for l in labels] for lbl in data['Finding Labels']]
print(data[['encoding','Finding Labels']])
class_count = {}
for lbl in labels:
    class_count[lbl] = data[lbl].sum()

classweight = {}
for lbl in labels :
    classweight[lbl] = 1/class_count[lbl]

classweight['Nothing'] /= 2   #Extra penalising the none class 
def func(row):
    weight = 0
    for lbl in labels: 
        if(row[lbl]==1):
            weight += classweight[lbl]
    return weight
new_weights = data.apply(func, axis=1)
sampled_data = data.sample(40000, weights = new_weights)
    
sampled_data.to_csv('sampled_data.csv')
sampled_data = pd.read_csv('/kaggle/input/nihmodelset/sampled_data.csv')
# Top 20 labels in the dataset 
disease_counts = sampled_data.groupby('Finding Labels')['Image Index'].count().sort_values(ascending=False).iloc[:20]

# plotting top 10 labels' count
plt.figure(figsize=(12,8))
plt.bar(np.arange(len(disease_counts))+0.5, disease_counts, tick_label=disease_counts.index)
plt.xticks(rotation=90)
# Getting train and test data
from sklearn.model_selection import train_test_split
train_data , test_data = train_test_split(sampled_data, test_size=0.2)
train_data , valid_data = train_test_split(train_data, test_size=0.25)
print(len(train_data))
print(len(test_data))
print(len(valid_data))
from keras.preprocessing.image import ImageDataGenerator 
IMG_SIZE = (299, 299)
# core imagedatagenerator used to create train and test imageDatagenerators
core_idg = ImageDataGenerator(samplewise_center=True, 
                              samplewise_std_normalization=True, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range= 0.05, 
                              width_shift_range=0.1, 
                              rotation_range=5, 
                              shear_range = 0.1,
                              fill_mode = 'reflect',
                              zoom_range=0.15)
# Fix for image datagenerator 

valid_data['newLabel'] = valid_data.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
train_data['newLabel'] = train_data.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
test_data['newLabel'] = test_data.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
train_gen = core_idg.flow_from_dataframe(
    dataframe=train_data,
    directory=None,
    x_col = 'path',
    y_col = 'newLabel',
    class_mode = 'categorical',
    target_size = IMG_SIZE,
    color_mode = 'rgb',
    batch_size = 32)

valid_gen = core_idg.flow_from_dataframe(
    dataframe=valid_data,
    directory=None,
    x_col = 'path',
    y_col = 'newLabel',
    class_mode = 'categorical',
    target_size = IMG_SIZE,
    color_mode = 'rgb',
    batch_size = 256) # we can use much larger batches for evaluation

test_X, test_Y = next(core_idg.flow_from_dataframe(
    dataframe=valid_data,
    directory=None,
    x_col = 'path',
    y_col = 'newLabel',
    class_mode = 'categorical',
    target_size = IMG_SIZE,
    color_mode = 'rgb',
    batch_size = 1024))
t_x, t_y = next(train_gen)
fig, m_axs = plt.subplots(4, 4, figsize = (16, 16))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = -1.5, vmax = 1.5)
    c_ax.set_title(', '.join([n_class for n_class, n_score in zip(labels, c_y) 
                             if n_score>0.5]))
    c_ax.axis('off')
# transfer learning on ResNet50
base_model = keras.applications.xception.Xception(include_top=False, weights='imagenet')
# completing the model
n_classes = len(labels)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation='sigmoid')(avg)
model = keras.Model(inputs=base_model.inputs, outputs = output)
model.summary()
keras.utils.plot_model(model)
met = ['categorical_accuracy', keras.metrics.Precision(), keras.metrics.AUC(), 'binary_accuracy']
# focal loss 
from keras import backend as K
def focal_loss(alpha=0.5,gamma=2.0):
    def focal_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
        
        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
        modulating_factor = K.pow((1-p_t), gamma)

        # compute the final loss and return
        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)
    return focal_crossentropy
for layer in base_model.layers:
    layer.trainable = False
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01), metrics=met)
history = model.fit_generator(train_gen, steps_per_epoch=100, validation_data=(test_X, test_Y), epochs=1,max_queue_size=100, workers=-1, use_multiprocessing=True)
model.save('Xception.h5')
model = keras.models.load_model('/kaggle/input/nihmodelset/Xception (2).h5')
for layer in base_model.layers:
    layer.trainable = True
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=met)
history = model.fit_generator(train_gen, steps_per_epoch=100, validation_data=(test_X, test_Y), epochs=3,max_queue_size=100, workers=-1, use_multiprocessing=True)
# Plotting the ROC curve
def plot_roc():
    pred_Y =  model.predict(test_X, batch_size = 32)
    from sklearn.metrics import roc_curve, auc
    fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
    for (idx, c_label) in enumerate(labels):
        fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    fig.savefig('XceptionRoc.png')

plot_roc();
model.summary()
print(model.layers[-1].name)
print(model.layers[-2].name)
properties = {
    "vgg16": {
        "img_size": (224, 224),
        "last_conv_layer": "block5_conv3",
        "last_classifier_layers": [
            "block5_pool",
            "flatten",
            "fc1",
            "fc2",
            "predictions",
        ],
        "model_builder": keras.applications.vgg16.VGG16,
        "preprocess_input": keras.applications.vgg16.preprocess_input,
        "decode_predictions": keras.applications.vgg16.decode_predictions,
    },
    "xception": {
        "img_size": (299, 299),
        "last_conv_layer": "block14_sepconv2_act",
        "last_classifier_layers": [
            "global_average_pooling2d",
            "dense",
        ],
        "model_builder": keras.applications.xception.Xception,
        "preprocess_input": keras.applications.xception.preprocess_input,
        "decode_predictions": keras.applications.xception.decode_predictions,
        
    }
}
NETWORK = "xception"
IMG_PATH = sampled_data['path'][1]
IMG_SIZE = properties[NETWORK]["img_size"]
LAST_CONV_LAYER = properties[NETWORK]["last_conv_layer"]
CLASSIFIER_LAYER_NAMES = properties[NETWORK]["last_classifier_layers"]
TOP_N = 15
print(IMG_PATH)
print(sampled_data['Finding Labels'][1])
print(labels)
model_builder = properties[NETWORK]["model_builder"]
preprocess_input = properties[NETWORK]["preprocess_input"]
decode_predictions = properties[NETWORK]["decode_predictions"]
def get_img_array(img_path=IMG_PATH, size=IMG_SIZE):
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array
DICT_BY_NAME = {}
for i in range(len(labels)):
    DICT_BY_NAME[labels[i]] = i;
CLASS_DICT = {}
for i in range(len(labels)):
    CLASS_DICT[i] = labels[i]
print(DICT_BY_NAME)
print(CLASS_DICT)
def get_top_predicted_indices(predictions, top_n):
    return np.argsort(-predictions).squeeze()[:top_n]

def make_gradcam_heatmap(img_array, model,last_conv_layer_name,
                         classifier_layer_names,top_n,class_indices):
#     Create a model that maps the input image to the activations of the last convolution layer 
    img_array = preprocess_input(img_array)
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
#     Create another model that maps from last convolution layer to final class predictions
    
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(inputs=classifier_input, outputs=x)
    
    
    if(top_n > 0):
        last_conv_layer_output = last_conv_layer_model(img_array)
        preds = classifier_model(last_conv_layer_output)
        class_indices = get_top_predicted_indices(preds, top_n)
    else:
        top_n = len(class_indices)
        
    heatmaps = []
    for index in np.arange(top_n):
        with tf.GradientTape() as tape:
            last_conv_layer_output = last_conv_layer_model(img_array)
#             print(last_conv_layer_output)
            tape.watch(last_conv_layer_output)
            preds = classifier_model(last_conv_layer_output)
#             print(preds)
            class_channel = preds[:, class_indices[index]]
            
            
        grads = tape.gradient(
            class_channel,
            last_conv_layer_output
        )
#         print(np.sum(grads))
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))    
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]
        heatmap = np.mean(last_conv_layer_output, axis=-1)
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap)+float(1e-7))
        heatmaps.append({
            "class_id": class_indices[index],
            "heatmap": heatmap
        })
    return heatmaps
    
class_indices = np.arange(15)
heatmaps = make_gradcam_heatmap(
    get_img_array(), 
    model, 
    LAST_CONV_LAYER, 
    CLASSIFIER_LAYER_NAMES, 
    0, 
    class_indices
)
import cv2
def superimpose_heatmap(image_path, heatmap):
    img = keras.preprocessing.image.load_img(image_path)
    img = keras.preprocessing.image.img_to_array(img)
    
    # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = keras.preprocessing.image.array_to_img(heatmap)
    heatmap = heatmap.resize((img.shape[1], img.shape[0]))
    
    heatmap = keras.preprocessing.image.img_to_array(heatmap)
    superimposed_img = cv2.addWeighted(heatmap, 0.4, img, 0.6, 0)
    superimposed_img = np.uint8(superimposed_img)
    
    return superimposed_img
def display_superimposed_heatmaps(heatmaps, image_path, image_id):
    n = len(heatmaps)
    n_rows = (n // 3) + 1 if n % 3 > 0 else n // 3
    plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.labelsize'] = False
    plt.rcParams['ytick.labelsize'] = False
    plt.rcParams['xtick.top'] = False
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.right'] = False
    plt.rcParams['figure.figsize'] = [30, 15]
    for index in np.arange(n):
        heatmap = heatmaps[index]["heatmap"]
        class_id = heatmaps[index]["class_id"]
#         class_name = CLASS_DICT[str(class_id)].split(",")[0].capitalize()
        superimposed_image = superimpose_heatmap(image_path, heatmap)
        plt.subplot(n_rows, 3, index+1)
        plt.title(f"{class_id}", fontsize= 30)
        plt.imshow(superimposed_image)
        
    plt.show()
display_superimposed_heatmaps(heatmaps, IMG_PATH, 1)

