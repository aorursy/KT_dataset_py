import matplotlib.pyplot as plt

from os.path import isfile

import pandas as pd

import numpy as np

import warnings

import sklearn

from tqdm._tqdm_notebook import tqdm_notebook

from keras import backend as K

from keras.layers import Input, Dense, Dropout, Conv2D, GlobalMaxPooling2D, BatchNormalization, MaxPool2D, concatenate, merge, Lambda

from keras.models import Model, Sequential

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from keras.optimizers import Adam

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

from sklearn import preprocessing

from keras.applications.resnet50 import ResNet50, preprocess_input



# Image cropping

import os

import PIL

from PIL import Image

from PIL.ImageDraw import Draw

import numpy as np

import matplotlib.pyplot as plt

from keras.models import load_model

from keras.preprocessing import image



'''

# Image processing

import imageio

import skimage

import skimage.io

import skimage.transform

from skimage import color

'''



from sklearn.model_selection import train_test_split

from collections import defaultdict

import glob

from os.path import join



import seaborn as sns

plt.style.use('fivethirtyeight')

%matplotlib inline
img_width = 128

img_height = 128

img_channels = 1

img_to_load = 25000

new_whale_distance = 0.1



batch_size = 200

epochs = 300

patience = 10



crop_margin = 0.2

embedding_size = 256
MODEL_BASE = '../input/bbox-model-whale-recognition'

BOXES = '../input/generating-whale-bounding-boxes/bounding_boxes.csv'

model = load_model(os.path.join(MODEL_BASE, 'cropping.model'))

Image.MAX_IMAGE_PIXELS = 3360368385340928



boxes = pd.read_csv(BOXES).set_index('Image')

    

def crop_image(rb_img_arr, img_width = img_width, img_height = img_height, draw = False):

    bbox  = model.predict(np.expand_dims(rb_img_arr, axis=0))[0]

    

    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:

        bbox = [0,0,128,128]

        

    if (draw):

        # draw rectangle

        draw = Draw(rimg)

        draw.rectangle(bbox, outline='red')

    

    rb_img = image.array_to_img(rb_img_arr)

    try:

        img_crop = rb_img.crop(tuple(bbox))

    except MemoryError:

        img_crop = rb_img

    

    img_crop = img_crop.convert('L')

    img_crop = img_crop.resize((img_width, img_height), PIL.Image.ANTIALIAS) 

    img_crop_arr = image.img_to_array(img_crop)    

    

    return img_crop_arr / 255.



def load_single_image(img_path, img_width, img_height):

    main_img = image.load_img(img_path)

    r_img = main_img.resize((128, 128), PIL.Image.ANTIALIAS)    

    r_img = r_img.convert('L')

    

    return image.img_to_array(r_img) / 255.

    

def make_bbox_image(img_name, img_width, img_height, draw = False):

    """

    :param img: path to image

    """

    main_img = image.load_img(expand_path(img_name))

    x0, y0, x1, y1 = tuple(boxes.loc[img_name,['x0','y0','x1','y1']].tolist())

    width, height = main_img.size

    dx            = x1 - x0

    dy            = y1 - y0

    x0           -= dx*crop_margin

    x1           += dx*crop_margin + 1

    y0           -= dy*crop_margin

    y1           += dy*crop_margin + 1

    

    if (x0 < 0     ): x0 = 0

    if (x1 > width): x1 = width

    if (y0 < 0     ): y0 = 0

    if (y1 > height): y1 = height

        

    try:

        img_crop = main_img.crop((x0, y0, x1, y1))

    except MemoryError:

        img_crop = main_img

    

    img_crop = img_crop.convert('L')

    img_crop = img_crop.resize((img_width, img_height), PIL.Image.ANTIALIAS)

    img_crop_arr = image.img_to_array(img_crop)    

    return img_crop_arr / 255.
def expand_path(image_name):

    if isfile('../input/humpback-whale-identification/train/' + image_name): return '../input/humpback-whale-identification/train/' + image_name

    if isfile('../input/humpback-whale-identification/test/' + image_name): return '../input/humpback-whale-identification/test/' + image_name

    return image_name



def read_data(path):

    whales = pd.read_csv(path, index_col=False)

    return whales



def read_train_data():

    return read_data('../input/humpback-whale-identification/train.csv')



def read_test_files():

    return [image.split('/')[4] for image in glob.glob(join('../input/humpback-whale-identification/test', '*.jpg'))]



def filter_whales(whales):

    not_new_whale = (whales.Id != "new_whale")

    return whales[not_new_whale]



def read_img(file, img_width, img_height):

    """

    Read and resize img, adjust channels. 

    @param file: file name without full path

    """

    return load_single_image(expand_path(file), img_width, img_height)

    #return make_bbox_image(file, img_width, img_height)

    '''

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        img = skimage.io.imread(expand_path(file))

        img = skimage.transform.resize(img, (img_width, img_height), mode='reflect', )

        img = color.rgb2gray(img)

        img = np.reshape(img, (img_width, img_height, 1))

    return img    

    '''



def load_images(files, img_width, img_height):

    """

    Load images for features, drop other columns

    One hot encode for label, drop other columns

    @return: train images, validation images, test images, train labels, validation labels, test labels

    """

    # Bees already splitted to train, validation and test

    # Load and transform images to have equal width/height/channels. 

    # Use np.stack to get NumPy array for CNN input



    # Train data

    tqdm_notebook.pandas()

    

    return {image_name:read_img(image_name, img_width, img_height) for image_name in tqdm_notebook(files)}

    

def load_images_and_target(train_whales, img_width, img_height):

    """

    Load images for features, drop other columns

    One hot encode for label, drop other columns

    @return: train images, validation images, test images, train labels, validation labels, test labels

    """

    # Bees already splitted to train, validation and test

    # Load and transform images to have equal width/height/channels. 

    # Use np.stack to get NumPy array for CNN input



    return load_images(train_whales['Image'], img_width, img_height)

    

def show_triplets(triplets, batch = 5):

    _, ax = plt.subplots(nrows = batch, ncols = 3, figsize = (100, 100))

    for i, (anchor, positive, negative) in enumerate(zip(triplets['anchor_input'], triplets['positive_input'], triplets['negative_input'])):

        ax[i][0].set_title("anchor", fontsize = 60)

        ax[i][0].imshow(anchor.squeeze(), cmap='gray')

        ax[i][1].set_title("positive", fontsize = 60)

        ax[i][1].imshow(positive.squeeze(), cmap='gray')

        ax[i][2].set_title("negative", fontsize = 60)

        ax[i][2].imshow(negative.squeeze(), cmap='gray')

        if (i >= batch - 1): break

    plt.tight_layout()

    plt.show()

    

def show_predictions(predictions, batch = 5):

    _, ax = plt.subplots(nrows = batch, ncols = 2, figsize = (100, 100))

    for i, prediction in enumerate(predictions):

        ax[i][1].set_title("test", fontsize = 60)

        ax[i][1].imshow(prediction['test_image'].squeeze(), cmap='gray')

        ax[i][0].set_title("predicted", fontsize = 60)

        ax[i][0].imshow(prediction['best_predicted_image'].squeeze(), cmap='gray')

        if (i >= batch - 1): break

    plt.tight_layout()

    plt.show()
all_train_whales = read_train_data()

all_train_whales.head()
all_train_whales.shape
np.unique(all_train_whales['Id']).shape
train_whales = filter_whales(all_train_whales)[:img_to_load]
file_mapping_image = load_images_and_target(train_whales, 

                                  img_width, 

                                  img_height 

                                  )
#img = make_bbox_image(train_whales['Image'][1], img_width, img_height)

#img.shape

img = file_mapping_image[train_whales['Image'][2]]

plt.imshow(img.squeeze(), cmap = 'gray')
#plt.imshow(np.uint8(image.img_to_array(img)))

#plt.show()
labels_count = train_whales.Id.value_counts()



plt.figure(figsize=(18, 4))

plt.subplot(121)

plt.hist(labels_count.values)

plt.ylabel("frequency")

plt.xlabel("class size")



plt.title('class distribution')

labels_count.head()
def triplet_loss(y_true, y_pred, margin=3):

    anchor_embedding = y_pred[:,:embedding_size]

    positive_embedding = y_pred[:,embedding_size:embedding_size*2]

    negative_embedding = y_pred[:,embedding_size*2:]

    positive_distance = K.square(anchor_embedding - positive_embedding)

    negative_distance = K.square(anchor_embedding - negative_embedding) 

    positive_distance = K.mean(positive_distance, axis=-1, keepdims=True)

    negative_distance = K.mean(negative_distance, axis=-1, keepdims=True)

    loss = K.maximum(0.0, margin + positive_distance - negative_distance)

    return K.mean(loss)

    

def add_convolutional_layer(model, filter_size, conv_layer_number = 2, batch_normalization = False, dropout = False):

    for _ in range(conv_layer_number):

        model.add(Conv2D(filters = filter_size, kernel_size = (3,3), padding='Same', activation ='relu', input_shape = (img_height, img_width, img_channels)))

        model.add(BatchNormalization()) if (batch_normalization) else False

        model.add(Dropout(0.5)) if (dropout) else False

    

    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    

def add_convolutional_base(model, conv_layer_number = 2, batch_normalization = False):

    add_convolutional_layer(model, 32, conv_layer_number, batch_normalization, True)

    add_convolutional_layer(model, 64, conv_layer_number, batch_normalization)

    add_convolutional_layer(model, 128, conv_layer_number, batch_normalization, True)

    add_convolutional_layer(model, 256, conv_layer_number, batch_normalization)

    add_convolutional_layer(model, 512, conv_layer_number, batch_normalization, True)

    add_convolutional_layer(model, 1024, conv_layer_number, batch_normalization)

    model.add(GlobalMaxPooling2D())

    

def add_embeddings(model, embeddings_number = 128, kernel_regularizer = None):

    model.add(Dense(1024, activation='relu', kernel_regularizer = kernel_regularizer, name='dense'))

    model.add(Dropout(0.5))

    model.add(Dense(embeddings_number, activation='relu', kernel_regularizer = kernel_regularizer, name='embeddings'))

    

def create_embedding_network_with_custom_model():

    model = Sequential()

    add_convolutional_base(model, 2, True)

    add_embeddings(model, embedding_size, None)

    

    return model



def add_resnet50(model):

    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='max')    

    for layer in resnet_model.layers:

        layer.trainable = False

    

    model.add(resnet_model)

    

def create_embedding_network_with_resnet():

    model = Sequential()

    add_resnet50(model)

    add_embeddings(model, embedding_size, None)

    

    return model



def build_model():

    positive_input = Input(shape=(img_height, img_width, img_channels), name = 'positive_input')

    negative_input = Input(shape=(img_height, img_width, img_channels), name = 'negative_input')

    anchor_input = Input(shape=(img_height, img_width, img_channels), name = 'anchor_input')



    # Create Common network to share the weights along different examples (+/-/Anchor)

    embedding_network = create_embedding_network_with_custom_model()



    positive_embedding = embedding_network(positive_input)

    negative_embedding = embedding_network(negative_input)

    anchor_embedding = embedding_network(anchor_input)

    

    output_vector = concatenate([anchor_embedding, positive_embedding, negative_embedding], name="positive_labels")

        

    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=output_vector)

    #model.add_loss(triplet_loss(output_vector))



    model.compile(optimizer=Adam(lr=1e-4), loss=triplet_loss, metrics=[triplet_loss])

    return model, embedding_network



def train_model(model, gen_train, gen_test):

    earlystopper = EarlyStopping(monitor='loss', patience=patience, verbose=1,restore_best_weights=True)

    return model.fit_generator(gen_train, 

                              validation_data=gen_test,

                              callbacks=[earlystopper],

                              epochs=epochs, 

                              verbose=0,

                              #workers=4,

                              #use_multiprocessing=True,

                              steps_per_epoch=steps_per_epoch,

                              validation_steps=validation_steps)



def eval_plot(History, epoch):

    epochs = len(History.history['loss'])

    plt.figure(figsize=(20,10))

    plt.figure(figsize=(20,10))

    sns.lineplot(range(1, epochs + 1), History.history['loss'], label='Train loss')

    sns.lineplot(range(1, epochs + 1), History.history['val_loss'], label='Test loss')

    plt.legend(['train', 'validation'], loc='upper left')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.title("Loss Graph")

    plt.show()
class sample_gen(object):

    def __init__(self, image_name_mapping_whale_id, other_class = "new_whale"):

        self.whale_id_with_images = dict()

        

        for image_name in image_name_mapping_whale_id:

            whale_id = image_name_mapping_whale_id[image_name]

            if whale_id in self.whale_id_with_images:

                self.whale_id_with_images[whale_id].append(image_name)

            else:

                self.whale_id_with_images[whale_id] = [image_name]

                

    def get_random_whale_id(self, exclude_whale_id, more_than_one_image):

        whale_ids = list(self.whale_id_with_images.keys())

        filtered_whale_ids = list(filter(lambda whale_id: whale_id != exclude_whale_id, whale_ids))

        # Get 0 probability for ids with just one image

        images_to_remove_count = 1 if more_than_one_image else 0

        filtered_whale_ids_probabilities = [len(self.whale_id_with_images[whale_id]) - images_to_remove_count for whale_id in filtered_whale_ids]

        filtered_whale_ids_probabilities = filtered_whale_ids_probabilities / np.sum(filtered_whale_ids_probabilities)

        return np.random.choice(filtered_whale_ids, 1, p = filtered_whale_ids_probabilities)[0]

    

    def get_sample(self):

        anchor_whale_id = self.get_random_whale_id('new_whale', True)

        negative_whale_id = self.get_random_whale_id(anchor_whale_id, False)

        

        positive_images = np.random.choice(self.whale_id_with_images[anchor_whale_id], 2, replace=False)

        negative_image = np.random.choice(self.whale_id_with_images[negative_whale_id], 1)



        return ((positive_images[0], positive_images[1], negative_image[0]), 

                (anchor_whale_id, negative_whale_id))



image_gen = ImageDataGenerator(rescale=(1/255),

                            rotation_range=5,

                           width_shift_range=0.0025,

                           height_shift_range=0.0025,

                           shear_range=0.001,

                           zoom_range=[0.95, 1.05],

                           horizontal_flip=True,

                           vertical_flip=False,

                           fill_mode='nearest',

                           data_format='channels_last',

                           #preprocessing_function=crop_image,

                           brightness_range=[0.9, 1.1])



image_gen.fit(list(file_mapping_image.values()))



le = preprocessing.LabelEncoder()

le.fit(np.array(train_whales.Id.values))



def create_aug_gen(in_gen, batch_size=25):

    for in_x, label in in_gen:

        anchor_input = image_gen.flow(in_x['anchor_input'], shuffle=False, batch_size=batch_size)

        positive_input = image_gen.flow(in_x['positive_input'], shuffle=False, batch_size=batch_size)

        negative_input = image_gen.flow(in_x['negative_input'], shuffle=False, batch_size=batch_size)

        

        yield ({'anchor_input': next(anchor_input), 'positive_input': next(positive_input), 'negative_input': next(negative_input)}, label)

        

def gen(triplet_gen, batch_size=25):

    while True:

        list_positive_examples_1 = []

        list_negative_examples = []

        list_positive_examples_2 = []

        positive_labels = []

        negative_labels = []



        for i in range(batch_size):

            ((positive_example_1, positive_example_2, negative_example),(positive_label, negative_label)) = triplet_gen.get_sample()

            

            list_positive_examples_1.append(file_mapping_image[positive_example_1])

            list_negative_examples.append(file_mapping_image[negative_example])

            list_positive_examples_2.append(file_mapping_image[positive_example_2])

            positive_labels.append(positive_label)

            negative_labels.append(negative_label)



        A = np.array(list_positive_examples_1)

        B = np.array(list_positive_examples_2)

        C = np.array(list_negative_examples)

         

        label = None

        

        yield ({'anchor_input': A, 'positive_input': B, 'negative_input': C}, {'positive_labels': le.transform(np.array(positive_labels)), 'negative_labels': le.transform(np.array(negative_labels))})
import matplotlib.pyplot as plt

import matplotlib.patheffects as PathEffects

import seaborn as sns

sns.set_style('darkgrid')

sns.set_palette('muted')

sns.set_context('notebook', font_scale=1.5,

                rc={"lines.linewidth": 2.5})



from sklearn.manifold import TSNE



def scatter(x, labels, subtitle=None):

    scatter_le = preprocessing.LabelEncoder()

    labels = scatter_le.fit_transform(np.array(labels))

    clusters = max(labels) + 1

    # We choose a color palette with seaborn.

    palette = np.array(sns.color_palette("hls", clusters))



    # We create a scatter plot.

    f = plt.figure(figsize=(8, 8))

    ax = plt.subplot(aspect='equal')

    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,

                    c=palette[labels.astype(np.int)])

    plt.xlim(-25, 25)

    plt.ylim(-25, 25)

    ax.axis('off')

    ax.axis('tight')



    # We add the labels for each digit.

    txts = []

    for i in range(clusters):

        # Position of each label.

        xtext, ytext = np.median(x[labels == i, :], axis=0)

        txt = ax.text(xtext, ytext, str(i), fontsize=24)

        txt.set_path_effects([

            PathEffects.Stroke(linewidth=5, foreground="w"),

            PathEffects.Normal()])

        txts.append(txt)

        

    if subtitle != None:

        plt.suptitle(subtitle)

        

    plt.show()
train, test = train_test_split(train_whales, train_size=0.7, test_size = 0.3, random_state=1337)

validation_steps = int(len(test) / batch_size) + 1

steps_per_epoch = int(len(train) / batch_size) + 1

print("Validation steps: " + str(validation_steps))

print("Steps per epoch: " + str(steps_per_epoch))

file_id_mapping_train = {k: v for k, v in zip(train.Image.values, train.Id.values)}

file_id_mapping_test = {k: v for k, v in zip(test.Image.values, test.Id.values)}

gen_train = create_aug_gen(gen(sample_gen(file_id_mapping_train),batch_size), batch_size)

#gen_train = gen(sample_gen(file_id_mapping_train),batch_size)

gen_test = gen(sample_gen(file_id_mapping_test),batch_size)

one_batch = next(gen_train)

show_triplets(one_batch[0], min(batch_size, 5))
def get_images_from_triplets(triplet_images, triplet_labels):

    images = []

    labels = []

    images.extend(triplet_images['anchor_input'])

    labels.extend(triplet_labels['positive_labels'])

    images.extend(triplet_images['positive_input'])

    labels.extend(triplet_labels['positive_labels'])

    images.extend(triplet_images['negative_input'])

    labels.extend(triplet_labels['negative_labels'])



    return np.array(images), np.array(labels)
tsne = TSNE()

images, labels = next(gen_train)

images, labels = get_images_from_triplets(images, labels)

train_tsne_embeds = tsne.fit_transform(images.reshape(-1, img_width*img_height*img_channels))

scatter(train_tsne_embeds, labels, "Samples from Training Data")



images, labels = next(gen_test)

images, labels = get_images_from_triplets(images, labels)

train_tsne_embeds = tsne.fit_transform(images.reshape(-1, img_width*img_height*img_channels))

scatter(train_tsne_embeds, labels, "Samples from Test Data")
triplet_model, embedding_model = build_model()

history = train_model(triplet_model, gen_train, gen_test)
eval_plot(history, epochs)
tsne = TSNE()

images, labels = next(gen_train)

images, labels = get_images_from_triplets(images, labels)

train_tsne_embeds = tsne.fit_transform(embedding_model.predict(images))

scatter(train_tsne_embeds, labels, "Samples from Training Data")



images, labels = next(gen_test)

images, labels = get_images_from_triplets(images, labels)

train_tsne_embeds = tsne.fit_transform(embedding_model.predict(images))

scatter(train_tsne_embeds, labels, "Samples from Test Data")
from sklearn.neighbors import NearestNeighbors

from __future__ import division



def data_generator(file_mapping_whale_id, batch = 25):

    i = 0

    for file, whale_id in file_mapping_whale_id.items():

        if i == 0:

            whale_ids = []

            whale_images = []

        whale_image = file_mapping_image[file]

        whale_images.append(whale_image)

        whale_ids.append(whale_id)

        i += 1

        if i == batch:

            i = 0

            yield whale_ids, np.array(whale_images)

    

    if i != 0:

        yield whale_ids, np.array(whale_images)

        

    raise StopIteration()

            

def predict_embeddings(file_mapping_whale_id):

    embedding_mapping_whale_id  = dict()

    embedding_mapping_image = dict()

    for whale_ids, whale_imgs in tqdm_notebook(data_generator(file_mapping_whale_id, batch=32)):

        predict_embeddings = embedding_model.predict(whale_imgs)

        for i, predict_embedding in enumerate(predict_embeddings):

            embedding_mapping_whale_id[tuple(predict_embedding)] = whale_ids[i]

            embedding_mapping_image[tuple(predict_embedding)] = whale_imgs[i]

    

    return embedding_mapping_whale_id, embedding_mapping_image



def remove_duplicates(predicted_whales):

    unique_whales_id = []

    no_duplicated_whales = []

    for predicted_whale in predicted_whales:

        if (predicted_whale['whale_id'] not in unique_whales_id):

            no_duplicated_whales.append(predicted_whale)

            unique_whales_id.append(predicted_whale['whale_id'])

    

    return no_duplicated_whales

    

def predict_nearest_neighbors(train_whales, test_whales):

    file_mapping_whale_id = {k: v for k, v in zip(train_whales.Image.values, train_whales.Id.values)}                      

    embedding_mapping_whale_id, embedding_mapping_image = predict_embeddings(file_mapping_whale_id)

    train_embeddings = list(embedding_mapping_whale_id.keys())

    neigh = NearestNeighbors(n_neighbors=20)

    neigh.fit(train_embeddings)

    

    test_whales_list = list(test_whales.items())

    test_images = np.array([image for file, image in test_whales_list])

    test_embeddings = embedding_model.predict(test_images)

    distances_test, neighbors_test = neigh.kneighbors(test_embeddings)

    predictions = []

    for i, (distance_test, neighbor_test) in enumerate(zip(distances_test, neighbors_test)):

        nearest_embedding = train_embeddings[neighbor_test[0]]

        

        predicted_whales = [{ 

            'whale_id': embedding_mapping_whale_id[train_embeddings[neighbor]],

            'distance': distance

        } for distance, neighbor in zip(distance_test, neighbor_test)]

        if "new_whale" not in [prediction['whale_id'] for prediction in predicted_whales]:

            predicted_whales.append({

                'whale_id': "new_whale",

                'distance': new_whale_distance

            })

        predicted_whales.sort(key=lambda prediction: prediction['distance'])

        

        predictions.append({ 

            'predicted_whale_id': remove_duplicates(predicted_whales)[:5],

            'best_predicted_image': embedding_mapping_image[nearest_embedding], 

            'test_image': test_whales_list[i][1],

            'test_file': test_whales_list[i][0]

        })

        

    return predictions



def calculate_prediction_accuracy(train_whales, predictions):

    file_mapping_whale_id = {k: v for k, v in zip(train_whales.Image.values, train_whales.Id.values)}

    accuracy = 0

    for prediction in predictions:

        accuracy += 1 if file_mapping_whale_id[prediction['test_file']] in [predicted_whale_id['whale_id'] for predicted_whale_id in prediction['predicted_whale_id']] else 0

        

    return accuracy / len(predictions)
train, test = train_test_split(train_whales, train_size=0.7, test_size = 0.3, random_state=1337)

test_whales = dict([(whale['Image'], file_mapping_image[whale['Image']]) for i, whale in test.iterrows()])

predictions = predict_nearest_neighbors(train, test_whales)

show_predictions(predictions)
calculate_prediction_accuracy(train_whales, predictions)
test_whales = load_images(read_test_files(), img_height, img_width)
predictions = predict_nearest_neighbors(train_whales, test_whales)

show_predictions(predictions)
def create_results(predictions):

    results = {

        'Id' : [],

        'Image': []

    }

    for prediction in predictions:

        results['Id'].append(" ".join([prediction['whale_id'] for prediction in prediction['predicted_whale_id']]))

        results['Image'].append(prediction['test_file'])

        

    return results
results = create_results(predictions)
df = pd.DataFrame(data=results)

df.to_csv("sub_humpback.csv", index=False)

df.head()