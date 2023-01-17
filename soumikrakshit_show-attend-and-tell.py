import os

import numpy as np

from glob import glob

from tqdm import tqdm

from PIL import Image

from time import time

import tensorflow as tf

from matplotlib import pyplot as plt



print('Tensorflow Version:', tf.__version__)
images_dir = '../input/flickr8k/Flickr_Data/Flickr_Data/Images/'

annotation_dir = '../input/flickr8k/Flickr_Data/Flickr_Data/Flickr_TextData/'

train_images = [

    images_dir + file.strip()

    for file in open(

        annotation_dir + 'Flickr_8k.trainImages.txt', 'r'

    ).read().split('\n')

][:-1]

val_images = [

    images_dir + file.strip()

    for file in open(

        annotation_dir + 'Flickr_8k.devImages.txt', 'r'

    ).read().split('\n')

][:-1]

test_images = [

    images_dir + file.strip()

    for file in open(

        annotation_dir + 'Flickr_8k.testImages.txt', 'r'

    ).read().split('\n')

][:-1]

len(train_images), len(val_images), len(test_images)
annotation_dict = {}

for line in open(annotation_dir + 'Flickr8k.token.txt', 'r').read().split('\n'):

    try:

        image_file, caption = line.split('\t')

        image_file = images_dir + image_file.split('#')[0]

        caption = '<start> ' + caption + ' <end>'

        annotation_dict[image_file] = caption

    except:

        pass

len(annotation_dict.keys())
train_captions = [annotation_dict[image] for image in train_images]

val_captions = [annotation_dict[image] for image in val_images]

test_captions = [annotation_dict[image] for image in test_images]
plt.imshow(Image.open(train_images[5999]))

plt.xlabel(train_captions[5999])

plt.show()
BACKBONE_DICT = {

    'inception_v3': {

        'model': tf.keras.applications.InceptionV3,

        'prep_func': tf.keras.applications.inception_v3.preprocess_input

    },

    'xception': {

        'model': tf.keras.applications.Xception,

        'prep_func': tf.keras.applications.xception.preprocess_input

    },

    'inception_resnet_v2': {

        'model': tf.keras.applications.InceptionResNetV2,

        'prep_func': tf.keras.applications.inception_resnet_v2.preprocess_input

    },

    'mobilenet': {

        'model': tf.keras.applications.MobileNet,

        'prep_func': tf.keras.applications.mobilenet.preprocess_input

    },

    'mobilenet_v2': {

        'model': tf.keras.applications.MobileNetV2,

        'prep_func': tf.keras.applications.mobilenet_v2.preprocess_input

    }

}
class FeatureExtractor:

    

    def __init__(self, backbone='inception_v3'):

        self.backbone = BACKBONE_DICT[backbone]['model'](include_top=False, weights='imagenet')

        self.prep_func = BACKBONE_DICT[backbone]['prep_func']

        self.feature_extraction_model = self.create_feature_extractor()

    

    def load_image(self, image_path):

        image = tf.io.read_file(image_path)

        image = tf.image.decode_jpeg(image, channels=3)

        image = tf.image.resize(image, (299, 299))

        image = self.prep_func(image)

        return image, image_path

    

    def create_feature_extractor(self):

        _input = self.backbone.input

        _output = self.backbone.layers[-1].output

        return tf.keras.Model(_input, _output)

    

    def cache_extracted_features(self, image_files, cache_dir, batch_size=16):

        dataset = tf.data.Dataset.from_tensor_slices(image_files)

        dataset = dataset.map(

            self.load_image,

            num_parallel_calls=tf.data.experimental.AUTOTUNE

        )

        dataset = dataset.batch(16)

        for image, path in tqdm(dataset):

            batch_features = self.feature_extraction_model(image)

            batch_features = tf.reshape(

                batch_features, (

                    batch_features.shape[0], -1,

                    batch_features.shape[3]

                )

            )

            for _feature, _path in zip(batch_features, path):

                _path = _path.numpy().decode("utf-8")

                _path = os.path.join(cache_dir, _path.split('/')[-1])

                np.save(_path, _feature)
feature_exractor = FeatureExtractor(backbone='inception_v3')
try:

    os.mkdir('./train_features')

except:

    print('Directory exists')

feature_exractor.cache_extracted_features(

    train_images, batch_size=8,

    cache_dir='./train_features'

)
try:

    os.mkdir('./val_features')

except:

    print('Directory exists')

feature_exractor.cache_extracted_features(

    val_images, batch_size=8,

    cache_dir='./val_features'

)
try:

    os.mkdir('./test_features')

except:

    print('Directory exists')

feature_exractor.cache_extracted_features(

    test_images, batch_size=8,

    cache_dir='./test_features'

)
tokenizer = tf.keras.preprocessing.text.Tokenizer(

    num_words=5000, oov_token="<unk>",

    filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ '

)

tokenizer.fit_on_texts(train_captions + val_captions + test_captions)
tokenizer.word_index['<pad>'] = 0

tokenizer.index_word[0] = '<pad>'
train_sequences = tokenizer.texts_to_sequences(train_captions)

val_sequences = tokenizer.texts_to_sequences(val_captions)

test_sequences = tokenizer.texts_to_sequences(test_captions)
_max_len = 0

for seq in train_sequences + val_sequences + test_sequences:

    _max_len = max(_max_len, len(seq))

_max_len
padded_train_sequences = tf.keras.preprocessing.sequence.pad_sequences(

    train_sequences, padding='post', maxlen=_max_len

)

padded_val_sequences = tf.keras.preprocessing.sequence.pad_sequences(

    val_sequences, padding='post', maxlen=_max_len

)

padded_test_sequences = tf.keras.preprocessing.sequence.pad_sequences(

    test_sequences, padding='post', maxlen=_max_len

)

padded_train_sequences.shape, padded_val_sequences.shape, padded_test_sequences.shape
class DataLoader:

    

    def __init__(self, dataset='train'):

        self.dataset = dataset

    

    def load_data(self, image_file, caption):

        image_file = image_file.decode('utf-8').split('/')[-1]

        image_file = os.path.join(self.dataset + '_features', image_file)

        feature_tensor = np.load(image_file)

        return feature_tensor, caption

    

    def get_dataset(self, captions, batch_size=64, buffer_size=1000):

        dataset = tf.data.Dataset.from_tensor_slices(

            (glob(self.dataset + '_features/*'), captions)

        )

        dataset = dataset.map(

            lambda item1, item2: tf.numpy_function(

                self.load_data, [item1, item2], [tf.float32, tf.int32]),

            num_parallel_calls=tf.data.experimental.AUTOTUNE

        )

        dataset = dataset.shuffle(buffer_size).batch(batch_size)

        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset
train_dataloader = DataLoader(dataset='train')

train_dataset = train_dataloader.get_dataset(padded_train_sequences)

x, y = next(iter(train_dataset))

x.shape, y.shape
val_dataloader = DataLoader(dataset='val')

val_dataset = val_dataloader.get_dataset(padded_val_sequences)

x, y = next(iter(val_dataset))

x.shape, y.shape
test_dataloader = DataLoader(dataset='test')

test_dataset = val_dataloader.get_dataset(padded_test_sequences)

x, y = next(iter(test_dataset))

x.shape, y.shape
class Attention(tf.keras.Model):

    

    def __init__(self, units):

        super(Attention, self).__init__()

        self.w_1 = tf.keras.layers.Dense(units)

        self.w_2 = tf.keras.layers.Dense(units)

        self.v = tf.keras.layers.Dense(1)

    

    def call(self, features, hidden):

        hidden = tf.expand_dims(hidden, 1)

        score = tf.nn.tanh(self.w_1(features) + self.w_2(hidden))

        attention_weights = tf.nn.softmax(self.v(score), axis=1)

        context_vector = attention_weights * features

        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
class Encoder(tf.keras.Model):

    

    def __init__(self, embedding_dimension):

        super(Encoder, self).__init__()

        self.dense = tf.keras.layers.Dense(embedding_dimension)

    

    def call(self, inputs):

        return tf.nn.relu(self.dense(inputs))
class Decoder(tf.keras.Model):

    

    def __init__(self, embedding_dimension, units, vocabulary_size):

        super(Decoder, self).__init__()

        self.units = units

        self.embedding = tf.keras.layers.Embedding(

            vocabulary_size, embedding_dimension

        )

        self.gru = tf.keras.layers.GRU(

            units, return_sequences=True,

            return_state=True, recurrent_initializer='glorot_uniform'

        )

        self.dense_1 = tf.keras.layers.Dense(self.units)

        self.dense_2 = tf.keras.layers.Dense(vocabulary_size)

        self.attention = Attention(self.units)

    

    def reset_state(self, batch_size):

        return tf.zeros((batch_size, self.units))

    

    def call(self, inputs, features, hidden):

        context_vector, attention_weights = self.attention(features, hidden)

        x = self.embedding(inputs)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)

        x = self.dense_1(output)

        x = tf.reshape(x, (-1, x.shape[2]))

        x = self.dense_2(x)

        return x, state, attention_weights
encoder = Encoder(256)

decoder = Decoder(256, 512, 5001)
cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(

    from_logits=True, reduction='none'

)



def loss_function(y, y_pred):

    mask = tf.math.logical_not(tf.math.equal(y, 0))

    loss_ = cross_entropy_loss(y, y_pred)

    mask = tf.cast(mask, dtype=loss_.dtype)

    loss_ *= mask

    return tf.reduce_mean(loss_)
optimizer = tf.keras.optimizers.Adam()
checkpoint_path = "./checkpoints/"

ckpt = tf.train.Checkpoint(

    encoder=encoder,

    decoder=decoder,

    optimizer = optimizer

)

ckpt_manager = tf.train.CheckpointManager(

    ckpt, checkpoint_path, max_to_keep=5

)
start_epoch = 0

if ckpt_manager.latest_checkpoint:

    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

    ckpt.restore(ckpt_manager.latest_checkpoint)
@tf.function

def train_step(x, y):

    loss = 0

    hidden_state = decoder.reset_state(batch_size=y.shape[0])

    decoder_input = tf.expand_dims(

        [tokenizer.word_index['<start>']] * y.shape[0], 1

    )

    with tf.GradientTape() as tape:

        features = encoder(x)

        for i in range(1, y.shape[1]):

            predictions, hidden, _ = decoder(

                decoder_input, features, hidden_state

            )

            loss += loss_function(y[:, i], predictions)

            decoder_input = tf.expand_dims(y[:, i], 1)

    total_loss = (loss / int(y.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss
loss_history = []

num_steps = len(train_images) // 64

for epoch in range(start_epoch, 20):

    start = time()

    total_loss = 0

    print('Epoch {}:'.format(str(epoch)))

    for (batch, (x, y)) in enumerate(train_dataset):

        batch_loss, total_batch_loss = train_step(x, y)

        total_loss += total_batch_loss

    loss_history.append(total_loss / num_steps)

    if epoch % 5 == 0:

        ckpt_manager.save()

    print('Loss {:.6f}\n'.format(total_loss / num_steps))

    print ('Time taken: {} sec\n'.format(time() - start))
plt.plot([loss.numpy() for loss in loss_history])