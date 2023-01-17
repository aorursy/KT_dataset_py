# To get pi

import math



# To do linear algebra

import numpy as np



# To store data

import pandas as pd



# To create nice plots

import seaborn as sns



# To count things

from collections import Counter



# To create interactive plots

import plotly.graph_objs as go

from plotly.offline import iplot



# To handle tensors

import tensorflow as tf

import tensorflow.keras.backend as K



# To handle datasets

from kaggle_datasets import KaggleDatasets



# To create plots

import matplotlib.pyplot as plt

from matplotlib.colors import Normalize

from matplotlib.colors import LinearSegmentedColormap



# To create a model

from keras.models import Model

from keras.layers import Input

from keras.layers.merge import concatenate

from keras.layers.pooling import MaxPooling2D

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.layers.convolutional import Conv2D, Conv2DTranspose
try:

    # Detect the hardware

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    # Configure tensorflow to use TPUs

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    

    # Get a strategy for distributing the model to TPUs

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default strategy to use CPUs or a GPU

    strategy = tf.distribute.get_strategy()



# Display number of model replicas

print("REPLICAS: ", strategy.num_replicas_in_sync)









# Allow optimization

MIXED_PRECISION = True

XLA_ACCELERATE = True



if MIXED_PRECISION:

    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    if tpu: policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')

    else: policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')

    mixed_precision.set_policy(policy)

    print('Mixed precision enabled')



if XLA_ACCELERATE:

    tf.config.optimizer.set_jit(True)

    print('Accelerated Linear Algebra enabled')
# Set batch size

BATCH_SIZE = 16 * strategy.num_replicas_in_sync



# Set number of epochs

EPOCHS = 250



# Patience for the learning rate

LR_PATIENCE = 5



# Patience for early stopping

STOPPING_PATIENCE = 30



# Serialized array shape

SHAPE = (600, 800, 4)



# Number of classes

NUM_CLASSES = 13



#Set number of images in buffer

BUFFER_SIZE = 1024 



# Set paths to the data

PATH_DATA = 'semantic-segmentation-with-carla-and-tpus'



# Allow self optimization

AUTO = tf.data.experimental.AUTOTUNE



# Label of the classes

CLASSES = {0:'Unlabeled',

           1:'Building',

           2:'Fence',

           3:'Other',

           4:'People',

           5:'Posts',

           6:'Road Marking',

           7:'Street',

           8:'Sidewalk',

           9:'Vegatation',

           10:'Vehicle',

           11:'Wall',

           12:'Traffic Sign'}



# RGB colors of the classes

COLORS = [(80/255, 168/255, 250/255),

          (242/255, 130/255, 30/255),

          (50/255, 50/255, 50/255),

          (27/255, 44/255, 129/255),

          (163/255, 68/255, 222/255),

          (115/255, 0/255, 0/255),

          (255/255, 255/255, 255/255),

          (191/255, 191/255, 191/255),

          (150/255, 150/255, 150/255),

          (22/255, 146/255, 0/255),

          (245/255, 239/255, 46/255),

          (181/255, 103/255, 10/255),

          (235/255, 0/255, 0/255)]
def loadDataset(filenames):

    # Disable order to increase speed

    options = tf.data.Options()

    options.experimental_deterministic = False

    

    # Define a TFRecords dataset with all filenames

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    # Add options for the order to the dataset

    dataset = dataset.with_options(options)

    # Return a dataset

    dataset = dataset.map(readTFRecord, num_parallel_calls=AUTO)

    return dataset







def readTFRecord(example):

    # Parse the serialized example

    example = tf.io.parse_single_example(example, exampleStructure)

    return example['raw_image']







# Define structure of examples

exampleStructure = {'raw_image': tf.io.FixedLenFeature([np.prod(SHAPE)], tf.int64)}





# Set the path to all data files

gcs_path = KaggleDatasets().get_gcs_path(PATH_DATA)



# Get a list of all files from the training folder

train_files = tf.io.gfile.glob(gcs_path + '/data_train/data_train/*.tfrecords')

# Get a list of all files from the testing folder

test_files = tf.io.gfile.glob(gcs_path + '/data_test/data_test/*.tfrecords')





# Create datasets from file names

train_dataset = loadDataset(train_files)

test_dataset = loadDataset(test_files)
def countLabels(dataset):

    # Counter for the labels

    label_counter = Counter()

    

    # Number of images

    n_images = 0



    # Iterate over the whole dataset

    for image_tensor in dataset:

        

        # Count images

        n_images += 1



        # Get the label channel

        flat_mask = tf.reshape(image_tensor, SHAPE)[:,:,3].numpy()



        # Count the labels

        x = np.bincount(flat_mask.flatten())

        y = np.nonzero(x)[0]



        # Update the counter

        for key, val in zip(y,x[y]):

            label_counter[key] += val

            

    # Return counted labels and number of images

    return label_counter, n_images





# Count labels in train- and test-dataset

train_label_counts, n_train_images = countLabels(train_dataset)

test_label_counts, n_test_images = countLabels(test_dataset)





# Create dataframe for the counted train labels

df_train_label_counts = pd.DataFrame.from_dict(train_label_counts, orient='index').rename({0:'Train'}, axis=1)

df_train_label_counts.index = [CLASSES[i] for i in df_train_label_counts.index]



# Create dataframe for the counted test labels

df_test_label_counts = pd.DataFrame.from_dict(test_label_counts, orient='index').rename({0:'Test'}, axis=1)

df_test_label_counts.index = [CLASSES[i] for i in df_test_label_counts.index]





# Combine the label counts dataframes

df_label_counts = df_train_label_counts.join(df_test_label_counts)

df_label_counts = df_label_counts.stack().to_frame().reset_index().rename({'level_0':'Label', 'level_1':'Dataset', 0:'Count'}, axis=1)





##### Create Label Count Plot #####



df_tmp_train = df_label_counts[df_label_counts['Dataset']=='Train'].sort_values('Count', ascending=False)

df_tmp_test = df_label_counts[df_label_counts['Dataset']=='Test'].sort_values('Count', ascending=False)





# Set up the data

trace1 = go.Bar(y = df_tmp_train['Label'],

                x = df_tmp_train['Count'],

                base = 0,

                name = 'Train',

                textposition = 'auto',

                hovertemplate = 'Pixel:  %{x}<br>Label: %{y}',

                width = 0.4,

                marker = dict(color = '#cc3600'),

                orientation='h',

                opacity = 1.0)



trace2 = go.Bar(y = df_tmp_test['Label'],

                x = df_tmp_test['Count'],

                base = 0,

                name = 'Test',

                textposition = 'auto',

                hovertemplate = 'Pixel:  %{x}<br>Label: %{y}',

                width = 0.4,

                marker = dict(color = '#cc9c00'),

                orientation='h',

                opacity = 1.0)



# Set up the layout

layout = dict(title = 'How Many Pixels Are There Per Label?',

              xaxis = dict(title = 'Count On Logscale',

                           type='log'),

              yaxis = dict(title = 'Label'),

              font = dict(family = 'sans-serif',

                          size = 18,

                          color = '#2f2f2f'),

              plot_bgcolor = '#dddddd')



# Create the plot

fig = go.Figure(data=[trace1, trace2], layout=layout)

iplot(fig)
# Create a colormap for the labels

cm = LinearSegmentedColormap.from_list('semantic_map', 

                                       COLORS, 

                                       N=NUM_CLASSES)



# Normalize the labels

norm = Normalize(vmin=0, vmax=12)



# Setup subplots

rows, cols = 5, 4



# Create iterator for dataset tensors

tensor_iterator = iter(train_dataset)



# Create the subplots

fig, arr = plt.subplots(rows, cols, figsize=(14, 14))



# Iterate over the first images

for i in range(int(rows*cols/2)):

    

    # Get the next data tensor

    image_tensor = next(tensor_iterator)

    

    # Get the image and mask channels

    flat_image = tf.reshape(image_tensor, SHAPE)[:,:,:3].numpy()

    flat_mask = tf.reshape(image_tensor, SHAPE)[:,:,3].numpy()

    

    # Reshape the image and mask 

    image = flat_image.reshape((SHAPE[0], SHAPE[1], 3))

    mask = flat_mask.reshape((SHAPE[0], SHAPE[1]))

    

    

    # Populate the subplots

    arr[i//2][i*2%cols].imshow(image)

    arr[i//2][i*2%cols].set_title('Image: {}'.format(i))

    arr[i//2][i*2%cols].axis('off')

    arr[i//2][i*2%cols+1].imshow(mask, cmap=cm, norm=norm)

    arr[i//2][i*2%cols+1].set_title('Segmentation: {}'.format(i))

    arr[i//2][i*2%cols+1].axis('off')

plt.show()
def trainingDataset(dataset, augmentation=True):

    # Reshape image tensor

    dataset = dataset.map(reshapeImage, num_parallel_calls=AUTO)

    

    # Perform augmentation

    if augmentation:

        # Random horizontal flip

        dataset = dataset.map(tf.image.random_flip_left_right, num_parallel_calls=AUTO)

        # Random rotation, shear, zoom and shift

        dataset = dataset.map(transformImage, num_parallel_calls=AUTO)

    

    # Split image and mask

    dataset = dataset.map(splitImage, num_parallel_calls=AUTO)

    

    # Repeat the dataset 

    dataset = dataset.repeat()

    # Set a buffersize to randomly choose images from

    dataset = dataset.shuffle(BUFFER_SIZE)

    # Set batchsize

    dataset = dataset.batch(BATCH_SIZE)

    # Prepare the next batch while training

    dataset = dataset.prefetch(AUTO)

    return dataset







def reshapeImage(tensor):

    # Reshape serialized data to tensor

    return tf.reshape(tensor, SHAPE)







def splitImage(tensor):

    # Slice image from tensor

    image = tensor[:,:,:3]

    # Slice mask from tensor

    mask = tensor[:,:,3]

    

    # Cast and normalize the ints to floats in [0, 1] range

    image = tf.cast(image, tf.float32) / 255.0

    return image, mask







def transformImage(tensor):

    # Get dimensions of image

    dim_x, dim_y, dim_z = SHAPE

    

    # Get random values for the transformation

    rot = 15. * tf.random.normal([1],dtype='float32')

    shr = 15. * tf.random.normal([1],dtype='float32')

    h_zoom = 1. + tf.random.normal([1],dtype='float32')/5.

    w_zoom = 1.33 + tf.random.normal([1],dtype='float32')/5.

    h_shift = 50. * tf.random.normal([1],dtype='float32')

    w_shift = 100. * tf.random.normal([1],dtype='float32')

  

    # Get transformation matrix from random transformation values

    m = transformationMatrix(rot, shr, h_zoom, w_zoom, h_shift, w_shift) 



    # Get a list of destination pixel indices

    x = tf.repeat( tf.range(dim_x//2, -dim_x//2, -1), dim_y )

    y = tf.tile( tf.range(-dim_y//2,dim_y//2), [dim_x] )

    z = tf.ones([dim_x*dim_y], dtype='int32')

    idx = tf.stack( [x, y, z] )

    

    # Rotate the destination pixels onto the origin pixels

    idx2 = K.dot(m, tf.cast(idx,dtype='float32'))

    idx2 = K.cast(idx2, dtype='int32')

    idx2 = K.clip(idx2, -dim_x//2+1, dim_x//2)

    

    # Find origin pixel values

    idx3 = tf.stack( [dim_x//2-idx2[0,], dim_x//2-1+idx2[1,]] )

    d = tf.gather_nd(tensor, tf.transpose(idx3))

    

    # Return transformed image

    return tf.reshape(d, SHAPE)







def transformationMatrix(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):

    ##### Create a single 3x3 transformation matrix from 4 individual transformations #####

        

    # Convert degrees to radians

    rotation = math.pi * rotation / 180.

    shear = math.pi * shear / 180.

    

    # Rotation matrix

    c1 = tf.math.cos(rotation)

    s1 = tf.math.sin(rotation)

    one = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )

        

    # Shear matrix

    c2 = tf.math.cos(shear)

    s2 = tf.math.sin(shear)

    shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )    

    

    # Zoom matrix

    zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )

    

    # Shift matrix

    shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )

    

    # Combine all four transformation matrices into a single transformation matrix

    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))







def testingDataset(dataset):

    # Reshape image tensor

    dataset = dataset.map(reshapeImage, num_parallel_calls=AUTO)

    

    # Split image and mask

    dataset = dataset.map(splitImage, num_parallel_calls=AUTO)

    

    # Set batchsize

    dataset = dataset.batch(BATCH_SIZE)

    # Prepare the next batch while training

    dataset = dataset.prefetch(AUTO)

    return dataset







# Get the training dataset

train_dataset = trainingDataset(train_dataset, augmentation=True)



# Create iterator for dataset tensors

train_dataset_iterator = iter(train_dataset.unbatch())



# Create the subplots

fig, arr = plt.subplots(5, 4, figsize=(14, 14))



# Iterate over the first images

for i in range(10):

    

    # Get the next image and mask

    image, mask = next(train_dataset_iterator)

    

    

    # Populate the subplots

    arr[i//2][i*2%4].imshow(image)

    arr[i//2][i*2%4].set_title('Image: {}'.format(i))

    arr[i//2][i*2%4].axis('off')

    arr[i//2][i*2%4+1].imshow(mask, cmap=cm, norm=norm)

    arr[i//2][i*2%4+1].set_title('Segmentation: {}'.format(i))

    arr[i//2][i*2%4+1].axis('off')

plt.show()
def getModel():

    

    # Use the TPU strategy

    with strategy.scope():

        

        # Build the model

        input_img = Input((SHAPE[0], SHAPE[1], 3), name='img')



        ##### Convolutions #####

        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (input_img)

        p1 = MaxPooling2D((2, 2)) (c1)



        c2 = Conv2D(12, (3, 3), activation='relu', padding='same') (p1)

        p2 = MaxPooling2D((2, 2)) (c2)



        c3 = Conv2D(16, (3, 3), activation='relu', padding='same') (p2)

        p3 = MaxPooling2D((2, 2)) (c3)

        

        c4 = Conv2D(24, (3, 3), activation='relu', padding='same') (p3)

        p4 = MaxPooling2D((5, 5)) (c4)



        

        c5 = Conv2D(36, (3, 3), activation='relu', padding='same') (p4)



        ##### Deconvolutions #####

        u6 = Conv2DTranspose(24, (2, 2), strides=(5, 5), padding='same') (c5)

        u6 = concatenate([u6, c4])

        c6 = Conv2D(20, (3, 3), activation='relu', padding='same') (u6)



        

        u7 = Conv2DTranspose(20, (2, 2), strides=(2, 2), padding='same') (c6)

        u7 = concatenate([u7, c3])

        c7 = Conv2D(18, (3, 3), activation='relu', padding='same') (u7)



        u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)

        u8 = concatenate([u8, c2])

        c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)



        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)

        u9 = concatenate([u9, c1])

        c9 = Conv2D(16, (3, 3), activation='relu', padding='same') (u9)



        outputs = Conv2D(13, (1, 1), activation='softmax') (c9)





        model = Model(inputs=[input_img], 

                      outputs=[outputs])



    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    

    model.compile(optimizer=optimizer, 

                  loss='sparse_categorical_crossentropy',

                  metrics=['sparse_categorical_accuracy'])



    print(model.summary())

    return model







# Get the model

model = getModel()
# Get the train- and test-dataset

train_dataset = trainingDataset(loadDataset(train_files), augmentation=True)

test_dataset = testingDataset(loadDataset(test_files))





# Setup callbacks

learning_rate = ReduceLROnPlateau(patience=LR_PATIENCE, verbose=1, factor=0.5, min_delta=0.00001)

early_stopping = EarlyStopping(patience=STOPPING_PATIENCE, verbose=1)





# Fit the model to the data

history = model.fit(train_dataset, 

                    steps_per_epoch = int(n_train_images / BATCH_SIZE),

                    epochs = EPOCHS,

                    callbacks = [early_stopping, learning_rate],

                    validation_data = test_dataset,

                    validation_steps = int(n_test_images / BATCH_SIZE))
# Get the training results

acc = history.history['sparse_categorical_accuracy']

val_acc = history.history['val_sparse_categorical_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

lr = history.history['lr']







##### Plot Training And Validation Accuracy ######



# Create template for hovertool

hovertemplate = 'Sparse Accuracy: %{y:.3f}<br>Epoch: %{x}'



# Set up the data

trace1 = go.Scatter(x = list(range(1, len(acc)+1)), 

                    y = acc, 

                    name = 'Training',

                    marker = dict(color = '#cc3600'),

                    hovertemplate = hovertemplate)



trace2 = go.Scatter(x = list(range(1, len(acc)+1)), 

                    y = val_acc, 

                    name = 'Testing',

                    marker = dict(color = '#0033cc'),

                    hovertemplate = hovertemplate)



# Set up the layout

layout = go.Layout(title = 'Sparse Accuracy During The Training',

                   font = dict(family = 'sans-serif',

                               size = 14,

                               color = '#2f2f2f'),

                   xaxis = dict(title = 'Epoch'),

                   yaxis = dict(title = 'Accuracy',

                                type='log'),

                   plot_bgcolor = '#ffdacc',

                   hovermode='x')



# Create the plot

fig = go.Figure(data=[trace1, trace2], layout=layout)

iplot(fig)







##### Plot Training And Validation Loss ######



# Create template for hovertool

hovertemplate = 'Loss:   %{y:.3f}<br>Epoch: %{x}'



# Set up the data

trace1 = go.Scatter(x = list(range(1, len(loss)+1)), 

                    y = loss, 

                    name = 'Training',

                    marker = dict(color = '#cc3600'),

                    hovertemplate = hovertemplate)



trace2 = go.Scatter(x = list(range(1, len(loss)+1)), 

                    y = val_loss, 

                    name = 'Testing',

                    marker = dict(color = '#0033cc'),

                    hovertemplate = hovertemplate)



# Set up the layout

layout = go.Layout(title = 'Loss During The Training',

                   font = dict(family = 'sans-serif',

                               size = 14,

                               color = '#2f2f2f'),

                   xaxis = dict(title = 'Epoch'),

                   yaxis = dict(title = 'Loss',

                                type='log'),

                   plot_bgcolor = '#ffdacc',

                   hovermode='x')



# Create the plot

fig = go.Figure(data=[trace1, trace2], layout=layout)

iplot(fig)







##### Plot Learning Rate ######



# Create template for hovertool

hovertemplate = 'Learning Rate:  %{y:.3f}<br>Epoch: %{x}'



# Set up the data

trace1 = go.Scatter(x = list(range(1, len(lr)+1)), 

                    y = lr, 

                    name = 'Training',

                    marker = dict(color = '#cc3600'),

                    hovertemplate = hovertemplate)



# Set up the layout

layout = go.Layout(title = 'Learning Rate During The Training',

                   font = dict(family = 'sans-serif',

                               size = 14,

                               color = '#2f2f2f'),

                   xaxis = dict(title = 'Epoch'),

                   yaxis = dict(title = 'Learning Rate',

                                type='log'),

                   plot_bgcolor = '#ffdacc',

                   hovermode='x')



# Create the plot

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)
# Create iterator for dataset tensors

test_dataset_iterator = iter(test_dataset.unbatch())



# Create the subplots

fig, arr = plt.subplots(5, 3, figsize=(14, 18))



# Iterate over the first images

for i in range(5):

    

    # Get the next image and mask

    image, mask = next(test_dataset_iterator)

    

    prediction_r = model.predict(np.expand_dims(image.numpy(), axis=0))[0]

    prediction = tf.argmax(prediction_r, axis=2)

    

    

    # Populate the subplots

    arr[i][0].imshow(image)

    arr[i][0].set_title('Image: {}'.format(i))

    arr[i][0].axis('off')

    arr[i][1].imshow(mask, cmap=cm, norm=norm)

    arr[i][1].set_title('Segmentation: {}'.format(i))

    arr[i][1].axis('off')

    arr[i][2].imshow(prediction, cmap=cm, norm=norm)

    arr[i][2].set_title('Prediction: {}'.format(i))

    arr[i][2].axis('off')

plt.show()
# Get counter for all pixel-predictions

counter = Counter()



# Iterate over the whole test dataset

for images, masks in test_dataset:

    

    # Get a prediction for the images

    predictions_r = model.predict(images)

    # Get the pixel labels

    predictions = tf.argmax(predictions_r, axis=3)

    

    # Add the preditions to the counter

    counter.update(zip(masks.numpy().flatten(), predictions.numpy().flatten()))





# Extract data from counter

counter_data = [[i, j, value] for (i, j), value in counter.items()]



# Create dataframe from counter

df_counter = pd.DataFrame(counter_data, columns=['Truth', 'Prediction', 'Count'])



# Pivot dataframe to get confusion matrix

piv = pd.pivot_table(df_counter, index='Truth', columns='Prediction', values='Count').fillna(0)



# Add missing columns

for i in piv.index:

    if i not in piv.columns:

        piv[i] = 0



# Sort columns

piv = piv[piv.index]



# Rename the columns and indices

piv = piv.rename(CLASSES, axis=0)

piv = piv.rename(CLASSES, axis=1)





# Count correct pixel

num_correct_pixel = np.diag(piv).sum()



# Count all pixel

num_all_pixel = piv.sum().sum()



# Compute overall accuracy

accuracy = num_correct_pixel / num_all_pixel





# Normalize the dataframe

piv = piv.divide(piv.sum(axis=1), axis=0).round(3)





# Create a plot for the confusion matrix

plt.figure(figsize = (12, 12))

plt.title('Confusion Matrix With Overall Accuracy: {:.3f}'.format(accuracy))

sns.heatmap(piv, annot=True, cmap='binary')

plt.show()