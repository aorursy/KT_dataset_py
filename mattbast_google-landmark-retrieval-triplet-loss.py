import numpy as np 

import pandas as pd

import tensorflow as tf

import tensorflow_addons as tfa



import matplotlib.pyplot as plt

from PIL import Image

from scipy.spatial import distance

from tqdm.notebook import tqdm



from kaggle_datasets import KaggleDatasets
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() 



print("REPLICAS: ", strategy.num_replicas_in_sync)
PT1_GCS_DS_PATH = KaggleDatasets().get_gcs_path('google-landmarks-2020-tfrecords')

print(PT1_GCS_DS_PATH)



PT2_GCS_DS_PATH = KaggleDatasets().get_gcs_path('google-landmarks-2020-triplet-loss-tfrecords-pt2')

print(PT2_GCS_DS_PATH)



PT3_GCS_DS_PATH = KaggleDatasets().get_gcs_path('google-landmarks-2020-triplet-loss-tfrecords-pt3')

print(PT3_GCS_DS_PATH)



PT4_GCS_DS_PATH = KaggleDatasets().get_gcs_path('google-landmarks-2020-triplet-loss-tfrecords-pt4')

print(PT4_GCS_DS_PATH)



PT5_GCS_DS_PATH = KaggleDatasets().get_gcs_path('google-landmarks-2020-triplet-loss-tfrecords-pt5')

print(PT5_GCS_DS_PATH)



PT6_GCS_DS_PATH = KaggleDatasets().get_gcs_path('google-landmarks-2020-triplet-loss-tfrecords-pt6')

print(PT6_GCS_DS_PATH)



PT7_GCS_DS_PATH = KaggleDatasets().get_gcs_path('google-landmarks-2020-triplet-loss-tfrecords-pt7')

print(PT7_GCS_DS_PATH)



PT8_GCS_DS_PATH = KaggleDatasets().get_gcs_path('google-landmarks-2020-triplet-loss-tfrecords-pt8')

print(PT8_GCS_DS_PATH)



PT9_GCS_DS_PATH = KaggleDatasets().get_gcs_path('google-landmarks-2020-tripley-loss-tfrecords-pt9')

print(PT9_GCS_DS_PATH)
BATCH_SIZE = 64 * strategy.num_replicas_in_sync

EPOCHS = 20

STEPS_PER_EPOCH = 1451645 // BATCH_SIZE

RATE = 0.0001



IMAGE_SIZE = 128

EMBED_SIZE = 2048
filenames = tf.io.gfile.glob([

    PT1_GCS_DS_PATH + '/*.tfrec', 

    PT2_GCS_DS_PATH + '/*.tfrec', 

    PT3_GCS_DS_PATH + '/*.tfrec',

    PT4_GCS_DS_PATH + '/*.tfrec', 

    PT5_GCS_DS_PATH + '/*.tfrec', 

    PT6_GCS_DS_PATH + '/*.tfrec',

    PT7_GCS_DS_PATH + '/*.tfrec', 

    PT8_GCS_DS_PATH + '/*.tfrec',

    PT9_GCS_DS_PATH + '/*.tfrec',

])
train_data = tf.data.TFRecordDataset(

    filenames,

    num_parallel_reads = tf.data.experimental.AUTOTUNE

)
ignore_order = tf.data.Options()

ignore_order.experimental_deterministic = False 

train_data = train_data.with_options(ignore_order)
def get_triplet(example):

    tfrec_format = {

        "anchor_img": tf.io.FixedLenFeature([], tf.string),

        "positive_img": tf.io.FixedLenFeature([], tf.string),

        "negative_img": tf.io.FixedLenFeature([], tf.string),

    }

    

    example = tf.io.parse_single_example(example, tfrec_format)

        

    x = {

        'anchor_input': decode_image(example['anchor_img']),

        'positive_input': decode_image(example['positive_img']),

        'negative_input': decode_image(example['negative_img']),

    }

    

    return x, [0, 0, 0]





def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.

    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE), method='nearest')

    

    image = augment(image)

    

    return image





def augment(image):

    rand_aug = np.random.choice([0, 1, 2, 3])

    

    if rand_aug == 0:

        image = tf.image.random_brightness(image, max_delta=0.4)

    elif rand_aug == 1:

        image = tf.image.random_contrast(image, lower=0.2, upper=0.5)

    elif rand_aug == 2:

        image = tf.image.random_hue(image, max_delta=0.2)

    else:

        image = tf.image.random_saturation(image, lower=0.2, upper=0.5)

    

    rand_aug = np.random.choice([0, 1, 2, 3])

    

    if rand_aug == 0:

        image = tf.image.random_flip_left_right(image)

    elif rand_aug == 1:

        image = tf.image.random_flip_up_down(image)

    elif rand_aug == 2:

        rand_rot = np.random.randn() * 45

        image = tfa.image.rotate(image, rand_rot)

    else:

        image = tfa.image.transform(image, [1.0, 1.0, -50, 0.0, 1.0, 0.0, 0.0, 0.0])



    image = tf.image.random_crop(image, size=[100, 100, 3])

    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    

    return image
train_data = train_data.map(

    get_triplet, 

    num_parallel_calls=tf.data.experimental.AUTOTUNE

)
train_data = train_data.repeat()

train_data = train_data.shuffle(1024)

train_data = train_data.batch(BATCH_SIZE)

train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
fig, axes = plt.subplots(5, 3, figsize=(15, 15))



for images, landmark_id in train_data.take(1):

    anchors = images['anchor_input']

    positives = images['positive_input']

    negatives = images['negative_input']

    

    for i in range(5):

        axes[i, 0].set_title('Anchor')

        axes[i, 0].imshow(anchors[i])



        axes[i, 1].set_title('Positive')

        axes[i, 1].imshow(positives[i])



        axes[i, 2].set_title('Negative')

        axes[i, 2].imshow(negatives[i])
class GeMPoolingLayer(tf.keras.layers.Layer):

    def __init__(self, p=1., eps=1e-6):

        super().__init__()

        self.p = p

        self.eps = eps



    def call(self, inputs: tf.Tensor, **kwargs):

        inputs = tf.clip_by_value(

            inputs, 

            clip_value_min=self.eps, 

            clip_value_max=tf.reduce_max(inputs)

        )

        inputs = tf.pow(inputs, self.p)

        inputs = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)

        inputs = tf.pow(inputs, 1. / self.p)

        

        return inputs

    

    def get_config(self):

        return {

            'p': self.p,

            'eps': self.eps

        }
reg = tf.keras.regularizers



with strategy.scope():

    # backbone

    backbone = tf.keras.applications.Xception(

        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),

        weights='imagenet',

        include_top=False

    )

    

    backbone.trainable = False

    

    # embedding model

    x_input = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    x = backbone(x_input)

    x = GeMPoolingLayer()(x)

    x = tf.keras.layers.Dense(EMBED_SIZE, activation='softplus', kernel_regularizer=reg.l2(), dtype='float32')(x)



    embedding_model = tf.keras.models.Model(inputs=x_input, outputs=x, name="embedding")



    # anchor encoding

    anchor_input = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='anchor_input')

    anchor_x = embedding_model(anchor_input)



    # positive encoding

    positive_input = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='positive_input')

    positive_x = embedding_model(positive_input)



    # anchor encoding

    negative_input = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='negative_input')

    negative_x = embedding_model(negative_input)



    # construct model

    model = tf.keras.models.Model(

        inputs=[anchor_input, positive_input, negative_input], 

        outputs=[anchor_x, positive_x, negative_x]

    )
embedding_model.summary()
def triplet_loss(y_true, y_pred, alpha=0.2):     

    anchors = y_pred[0]

    positives = y_pred[1]

    negatives = y_pred[2]

    

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchors, positives)), axis=-1)

    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchors, negatives)), axis=-1)



    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)

    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))



    return loss
model.compile(

    optimizer=tf.keras.optimizers.Adam(learning_rate=RATE),

    loss = triplet_loss

)
callbacks = [

    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2, verbose=1),

    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True),

]
history = model.fit_generator(

    train_data,

    epochs = EPOCHS,

    steps_per_epoch = STEPS_PER_EPOCH,

    callbacks = callbacks,

)
plt.title('Model loss')

plt.plot(history.history['loss'])
def distance_test(anchors, positives, negatives):    

    pos_dist = []

    neg_dist = []

    

    anchor_encodings = embedding_model.predict(anchors)

    positive_encodings = embedding_model.predict(positives)

    negative_encodings = embedding_model.predict(negatives)

    

    for i in range(len(anchors)):

        pos_dist.append(

            distance.euclidean(anchor_encodings[i], positive_encodings[i])

        )

        

        neg_dist.append(

            distance.euclidean(anchor_encodings[i], negative_encodings[i])

        )

    

    return pos_dist, neg_dist
pos_dist, neg_dist = distance_test(anchors[0:5], positives[0:5], negatives[0:5])
fig, axes = plt.subplots(5, 3, figsize=(15, 20))



for i in range(5):

    axes[i, 0].set_title('Anchor')

    axes[i, 0].imshow(anchors[i])



    axes[i, 1].set_title('Positive dist: {:.2f}'.format(pos_dist[i]))

    axes[i, 1].imshow(positives[i])



    axes[i, 2].set_title('Negative dist: {:.2f}'.format(neg_dist[i]))

    axes[i, 2].imshow(negatives[i])
image_ids = pd.read_csv(

    '../input/landmark-retrieval-2020/train.csv',

    nrows=100

)
def get_image(img_id):    

    chars = [char for char in img_id]

    dir_1, dir_2, dir_3 = chars[0], chars[1], chars[2]

    

    image = Image.open('../input/landmark-retrieval-2020/train/' + dir_1 + '/' + dir_2 + '/' + dir_3 + '/' + img_id + '.jpg')

    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image = np.asarray(image) / 255.0

    

    return image
images = [get_image(img_id) for img_id in image_ids.id]

images = np.array(images)
embeddings = embedding_model.predict(images)
distances = distance.cdist(embeddings, embeddings, 'euclidean')

print(distances.shape)
predicted_positions = np.argpartition(distances, 10, axis=1)[:, :10]
anchor_img = get_image(

    image_ids.loc[41].id

)



plt.title('Landmark: {}'.format(image_ids.loc[40].landmark_id))

plt.imshow(anchor_img)
fig, ax = plt.subplots(3, 3, figsize=(12, 12))



top_ten_indexes = predicted_positions[41]



for i in range(3):

    for j in range(3):

        img_index = top_ten_indexes[j + (i*3)]

        

        landmark_id = image_ids.loc[img_index].landmark_id

        dist = distances[41, img_index]



        ax[i,j].set_title('landmark: {}, dist: {:.2f}'.format(landmark_id, dist))

        ax[i,j].imshow(

            get_image(image_ids.loc[img_index].id)

        )
anchor_img = get_image(

    image_ids.loc[80].id

)



plt.imshow(anchor_img)
fig, ax = plt.subplots(3, 3, figsize=(12, 12))



top_ten_indexes = predicted_positions[80]



for i in range(3):

    for j in range(3):

        img_index = top_ten_indexes[j + (i*3)]

        

        landmark_id = image_ids.loc[img_index].landmark_id

        dist = distances[80, img_index]



        ax[i,j].set_title('landmark: {}, dist: {:.2f}'.format(landmark_id, dist))

        ax[i,j].imshow(

            get_image(image_ids.loc[img_index].id)

        )
embedding_model.save(

    'embedding_model.h5', 

    save_format='h5',

    overwrite=True

)
# embedding_model = tf.keras.models.load_model(

#     'embedding_model.h5',

#     custom_objects={'GeMPoolingLayer': GeMPoolingLayer}

# )
# class MyModel(tf.keras.Model):

#     def __init__(self):

#         super(MyModel, self).__init__()

#         self.model = embedding_model

    

#     @tf.function(input_signature=[

#       tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8, name='input_image')

#     ])

    

#     def call(self, input_image):

#         output_tensors = {}

        

#         input_image = tf.cast(input_image, tf.float32) / 255.0

#         input_image = tf.image.resize(input_image, (IMAGE_SIZE, IMAGE_SIZE)) 

                

#         extracted_features = self.model(tf.convert_to_tensor([input_image], dtype=tf.float32))[0]

#         output_tensors['global_descriptor'] = tf.identity(extracted_features, name='global_descriptor')

#         return output_tensors
# m = MyModel()



# served_function = m.call



# tf.saved_model.save(

#     m, 

#     export_dir="./model", 

#     signatures={'serving_default': served_function}

# )
# from zipfile import ZipFile



# with ZipFile('submission.zip','w') as output_zip_file:

#     for filename in os.listdir('./model'):

#         if os.path.isfile('./model/'+filename):

#             output_zip_file.write('./model/'+filename, arcname=filename) 

    

#     for filename in os.listdir('./model/variables'):

#         if os.path.isfile('./model/variables/'+filename):

#             output_zip_file.write('./model/variables/'+filename, arcname='variables/'+filename)

    

#     for filename in os.listdir('./model/assets'):

#         if os.path.isfile('./model/assets/'+filename):

#             output_zip_file.write('./model/assets/'+filename, arcname='assets/'+filename)