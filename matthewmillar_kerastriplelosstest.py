from tensorflow.keras.applications.xception import preprocess_input, Xception

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.layers import Input, Concatenate

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.layers import Flatten

import tensorflow as tf

from tensorflow.python.ops import array_ops

from tensorflow.python.ops import math_ops

from tensorflow.python.framework import dtypes

import matplotlib.pyplot as plt
def pairwise_distance(feature, squared=False):

    pairwise_distances_squared = math_ops.add(

        math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),

        math_ops.reduce_sum(

            math_ops.square(array_ops.transpose(feature)),

            axis=[0],

            keepdims=True)) - 2.0 * math_ops.matmul(feature,array_ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.

    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)

    # Get the mask where the zero distances are at.

    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.

    if squared:

        pairwise_distances = pairwise_distances_squared

    else:

        pairwise_distances = math_ops.sqrt(

            pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

    # Undo conditionally adding 1e-16.

    pairwise_distances = math_ops.multiply(

        pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

    num_data = array_ops.shape(feature)[0]

    # Explicitly set diagonals to zero.

    mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(

        array_ops.ones([num_data]))

    pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)

    return pairwise_distances



def masked_maximum(data, mask, dim=1):



    axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)

    masked_maximums = math_ops.reduce_max(

        math_ops.multiply(data - axis_minimums, mask), dim,

        keepdims=True) + axis_minimums

    return masked_maximums



def masked_minimum(data, mask, dim=1):



    axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)

    masked_minimums = math_ops.reduce_min(

        math_ops.multiply(data - axis_maximums, mask), dim,

        keepdims=True) + axis_maximums

    return masked_minimums





def triplet_loss_keras(y_true, y_pred):

    del y_true

    margin = 1.

    labels = y_pred[:, :1]

    labels = tf.cast(labels, dtype='int32')

    embeddings = y_pred[:, 1:]



    ### Code from Tensorflow function [tf.contrib.losses.metric_learning.triplet_semihard_loss] starts here:

    # Build pairwise squared distance matrix.

    pdist_matrix = pairwise_distance(embeddings, squared=True)

    # Build pairwise binary adjacency matrix.

    adjacency = math_ops.equal(labels, array_ops.transpose(labels))

    # Invert so we can select negatives only.

    adjacency_not = math_ops.logical_not(adjacency)



    # global batch_size

    batch_size = array_ops.size(labels)  # was 'array_ops.size(labels)'



    # Compute the mask.

    pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])

    mask = math_ops.logical_and(

        array_ops.tile(adjacency_not, [batch_size, 1]),

        math_ops.greater(

            pdist_matrix_tile, array_ops.reshape(

                array_ops.transpose(pdist_matrix), [-1, 1])))

    mask_final = array_ops.reshape(

        math_ops.greater(

            math_ops.reduce_sum(

                math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True),

            0.0), [batch_size, batch_size])

    mask_final = array_ops.transpose(mask_final)



    adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)

    mask = math_ops.cast(mask, dtype=dtypes.float32)



    # negatives_outside: smallest D_an where D_an > D_ap.

    negatives_outside = array_ops.reshape(

        masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])

    negatives_outside = array_ops.transpose(negatives_outside)



    # negatives_inside: largest D_an.

    negatives_inside = array_ops.tile(

        masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])

    semi_hard_negatives = array_ops.where(

        mask_final, negatives_outside, negatives_inside)



    loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)



    mask_positives = math_ops.cast(

        adjacency, dtype=dtypes.float32) - array_ops.diag(

        array_ops.ones([batch_size]))



    # In lifted-struct, the authors multiply 0.5 for upper triangular

    #   in semihard, they take all positive pairs except the diagonal.

    num_positives = math_ops.reduce_sum(mask_positives)



    semi_hard_triplet_loss_distance = math_ops.truediv(

        math_ops.reduce_sum(

            math_ops.maximum(

                math_ops.multiply(loss_mat, mask_positives), 0.0)),

        num_positives,

        name='triplet_semihard_loss')



    ### Code from Tensorflow function semi-hard triplet loss ENDS here.

    return semi_hard_triplet_loss_distance
DATA_PATH = '../input/gtsplitfolders/Data/'
def tripleloss_generator(generator):

    while True:

        img, label = next(generator)

        train_set = [img, label]

        yield train_set, label
base_model = Xception(include_top=False, weights='imagenet')
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    shear_range=0,

    rotation_range=20,

    zoom_range=0.15,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest')



train_generator = train_datagen.flow_from_directory(

    DATA_PATH,

    target_size=(224, 224),

    batch_size=BATCH_SIZE,

    shuffle=True,

    class_mode='categorical')
triplet_generator = tripleloss_generator(train_generator)
# Start building model

image_input = Input(shape=(224,224,3), name='Image')

label_input = Input(shape=(1501,), name='Label')

# Get the embedding for the triple loss

img_embedding = base_model(image_input)

img_embedding = Flatten()(img_embedding)

triple_data = Concatenate(axis=1)([label_input, img_embedding])

model = Model(inputs=[image_input, label_input], outputs=triple_data)

model.compile(loss=triplet_loss_keras,

              optimizer=tf.keras.optimizers.Adam(0.001, decay=0.002))



print(model.summary())
filepath = "KerasTripleLoss_{epoch:02d}_{loss:04f}.h5"

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min',

                             save_weights_only=False)

callbacks_list = [checkpoint]

history = model.fit_generator(triplet_generator,

                                  steps_per_epoch=2000,

                                  epochs=2,

                                  verbose=1,

                                  callbacks=callbacks_list)
print("Logging results...")

plt.figure(figsize=(8, 8))

plt.plot(history.history['loss'], label='Triple loss')

plt.legend()

plt.title('Train loss')

plt.savefig("TripleTest.png")