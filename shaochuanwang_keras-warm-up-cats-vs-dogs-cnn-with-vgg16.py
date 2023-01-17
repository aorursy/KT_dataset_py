import glob
import os, sys
import random
from tqdm import tqdm

import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

from keras import backend as K
K.tensorflow_backend._get_available_gpus()
train_data_dir = '../input/dogs-vs-cats-redux-kernels-edition/train'
test_data_dir = '../input/dogs-vs-cats-redux-kernels-edition/test'

# Make sure you include https://www.kaggle.com/keras/vgg16/data as your data source
vgg_model_path = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

epochs = 20
batch_size = 20
img_width, img_height = 150, 150

training_n_bound = 5000  # set to None to use the entire training dataset; it took about 2 hours at my Macbook Pro.
def gen_image_label(directory):
    ''' A generator that yields (label, id, jpg_filename) tuple.'''
    for root, dirs, files in os.walk(directory):
        for f in files:
            _, ext = os.path.splitext(f)
            if ext != '.jpg':
                continue
            basename = os.path.basename(f)
            splits = basename.split('.')
            if len(splits) == 3:
                label, id_, ext = splits
            else:
                label = None
                id_, ext = splits
            fullname = os.path.join(root, f)
            yield label, int(id_), fullname
# Wrap training data into pandas' DataFrame.
lst = list(gen_image_label(train_data_dir))
random.shuffle(lst)
if training_n_bound is not None:
    lst = lst[:training_n_bound]
train_df = pd.DataFrame(lst, columns=['label', 'id', 'filename'])
train_df = train_df.sort_values(by=['label', 'id'])
train_df.head(3)
train_df['label_code'] = train_df.label.map({'cat':0, 'dog':1})
train_df.head(3)
# Wrap testing data into pandas' DataFrame.
lst = list(gen_image_label(test_data_dir))
test_df = pd.DataFrame(lst, columns=['label', 'id', 'filename'])
test_df = test_df.sort_values(by=['label', 'id'])
test_df['label_code'] = test_df.label.map({'cat':0, 'dog':1})

test_df.head(3)
sns.countplot(train_df.label)
plt.title('Number of training images per category')
def display_images(label, n=5):
    fig = plt.figure(figsize=(16, 8))
    for j, fn in enumerate(train_df.loc[train_df.label == label].head(n).filename):
        img = load_img(fn, target_size=(img_width, img_height))
        fig.add_subplot(1, n, j + 1)
        f = plt.imshow(img)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.title(label)
    plt.show()

display_images('dog', 5)
display_images('cat', 5)
def gen_label_image_batch(df, batch_size, n_max_batch=10):
    ''' A generator that yields image as np array, batch by batch.'''
    stacked = None
    img_arrays = []
    label_arrays = []
    n_batch = 0
    for index, row in df.iterrows():
        img_arrays.append(
            img_to_array(
                load_img(row['filename'], target_size=(img_width, img_height))))
        label_arrays.append(row['label_code'])
        if len(img_arrays) % batch_size == 0:
            yield np.array(label_arrays), np.stack(img_arrays)
            n_batch += 1
            img_arrays = []
            label_arrays = []
            if n_max_batch is not None and n_batch == n_max_batch:
                break
    if img_arrays and label_arrays:
        yield np.array(label_arrays), np.stack(img_arrays)
datagen = ImageDataGenerator(rescale=1./255)
def gen_embedding_batch(df, batch_size, n_max_batch=None):
    ''' A generator that yields the embeddings, batch by batch 
        The embedding comes from pretrained VGG16 model.
    '''
    batches = gen_label_image_batch(df, 
                                    batch_size=batch_size, 
                                    n_max_batch=n_max_batch)
    model = applications.VGG16(include_top=False, 
                               weights=vgg_model_path)
    for i, (label, imgs) in tqdm(enumerate(batches)):
        generator = datagen.flow(
            imgs,
            label,
            batch_size=batch_size,
            shuffle=False)
        embedding_batch = model.predict_generator(
            generator, workers=4, verbose=0)
        yield embedding_batch
def gen_or_load_embedding(df, saved_embedding, force_gen=False):
    if os.path.exists(saved_embedding) and not force_gen:
        print('Loading embedding from %s...' % (saved_embedding,))
        embedding = np.load(open(saved_embedding, 'rb'))
    else:
        embedding = np.stack(
            gen_embedding_batch(df, 
                                batch_size=batch_size), 
            axis=0)
        embedding = embedding.reshape(
            [embedding.shape[0] * embedding.shape[1]] + list(embedding.shape[2:]))
        np.save(open(saved_embedding, 'wb'), embedding)
    return embedding
train_embeddings = gen_or_load_embedding(train_df, 'train_embeddings.npy', force_gen=True)
test_embeddings = gen_or_load_embedding(test_df, 'test_embeddings.npy', force_gen=False)
# Check embeddings' dimensions
[train_embeddings.shape, test_embeddings.shape]
train_indices = np.nonzero((train_df.id[:train_embeddings.shape[0]] % 4 != 0).values)[0]
validate_indices = np.nonzero((train_df.id[:train_embeddings.shape[0]] % 4 == 0).values)[0]
train_labels = train_df.label_code.values[train_indices]
validation_labels = train_df.label_code.values[validate_indices]
embedding_fc_model = 'embedding_fc_model.h5'
model = Sequential()
model.add(Flatten(input_shape=train_embeddings.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(train_embeddings[train_indices,:],
          train_labels,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(train_embeddings[validate_indices,:],
                           validation_labels))
model.save_weights(embedding_fc_model)
from sklearn.metrics import f1_score, accuracy_score

pred_validation = model.predict(train_embeddings[validate_indices,:])

f1 = f1_score(validation_labels, pred_validation > 0.5)
acc = accuracy_score(validation_labels, pred_validation > 0.5)
(f1, acc)
pred_test = model.predict(test_embeddings)
pred_test.shape
# Adjust n here if you want to see more results for testing dataset
n = 10
for i, (index, row) in enumerate(test_df.iterrows()):
    if i >= n:
        break
    fig = plt.figure(figsize=(8, 32))
    img = load_img(row['filename'], target_size=(img_width, img_height))
    subfig = fig.add_subplot(n, 1, i + 1)
    pred = pred_test[i][0]
    pred_label = 'dog' if pred > 0.5 else 'cat'
    pred = pred if pred > 0.5 else 1-pred
    plt.title('Looks like a {0} with probability {1}'.format(pred_label, pred))
    f = plt.imshow(img)
    f.axes.get_xaxis().set_visible(False)
    f.axes.get_yaxis().set_visible(False)
results = pd.DataFrame({'id': pd.Series(test_df.id.values[:pred_test.shape[0]]),
                        'label': pd.Series(pred_test.T[0])})
results.to_csv('submission.csv', index=False)
results.head(10)
from sklearn.decomposition import PCA
pca = PCA(n_components=3)

row_count = train_embeddings[train_indices,:].shape[0]
embedding_d = int(train_embeddings[train_indices,:].size / row_count)
vectors_train = train_embeddings[train_indices,:].reshape(row_count, embedding_d)
X = pca.fit_transform(vectors_train)
df = pd.DataFrame(np.concatenate((X,
                                  train_labels[:train_embeddings[train_indices,:].shape[0]].reshape(train_embeddings[train_indices,:].shape[0],1)),
                                 axis=1),
                  columns=['X', 'Y', 'Z', 'label'])
g = sns.FacetGrid(df, hue="label", size=7)
g.map(plt.scatter, "X", "Y", alpha=.5)
g.add_legend();

g = sns.FacetGrid(df, hue="label", size=7)
g.map(plt.scatter, "Y", "Z", alpha=.5)
g.add_legend();

g = sns.FacetGrid(df, hue="label", size=7)
g.map(plt.scatter, "X", "Z", alpha=.5)
g.add_legend();
