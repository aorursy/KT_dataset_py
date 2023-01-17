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
!pip install kulc
!git clone https://github.com/ahmadelsallab/MultiCheXNet.git
import tensorflow as tf
import pandas as pd 
import cv2
import numpy as np
import os
from glob import glob
import math
import matplotlib.pyplot as plt

import re
import html
import string
import unicodedata
from nltk.tokenize import word_tokenize
df  =pd.read_csv("/kaggle/input/chest-xrays-indiana-university/indiana_reports.csv")
df.head()
len(df)
dff = pd.read_csv('../input/chest-xrays-indiana-university/indiana_projections.csv')
dff.head()
df['findings'].iloc[0:10].tolist()
df.shape
df['impression'].unique().shape
df['MeSH'].unique().tolist()[:20]
img = cv2.imread('/kaggle/input/chest-xrays-indiana-university/images/images_normalized/1_IM-0001-3001.dcm.png')
plt.imshow(img)
plt.show()
df2 = pd.read_csv("/kaggle/input/chest-xrays-indiana-university/indiana_projections.csv")
df2.head()
df2.projection.unique()
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow
def remove_special_chars(text):
    re1 = re.compile(r'  +')
    x1 = text.lower().replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x1))


def remove_non_ascii(text):
    """Remove non-ASCII characters from list of tokenized words"""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def to_lowercase(text):
    return text.lower()

def remove_punctuation(text):
    """Remove punctuation from list of tokenized words"""
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def replace_numbers(text):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    return re.sub(r'\d+', '', text)


def remove_whitespaces(text):
    return text.strip()


def remove_stopwords(words, stop_words):
    """
    :param words:
    :type words:
    :param stop_words: from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
    or
    from spacy.lang.en.stop_words import STOP_WORDS
    :type stop_words:
    :return:
    :rtype:
    """
    return [word for word in words if word not in stop_words]


def stem_words(words):
    """Stem words in text"""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]

def lemmatize_words(words):
    """Lemmatize words in text"""

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

def lemmatize_verbs(words):
    """Lemmatize verbs in text"""

    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word, pos='v') for word in words])

def text2words(text):
    return word_tokenize(text)

def normalize_text( text):
    text = remove_special_chars(text)
    text = remove_non_ascii(text)
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_numbers(text)
    #words = text2words(text)
    #stop_words = stopwords.words('english')
    #words = remove_stopwords(words, stop_words)
    #words = stem_words(words)# Either stem ovocar lemmatize
    #words = lemmatize_words(words)
    #words = lemmatize_verbs(words)

    return text
  
def normalize_corpus(corpus):
    return [normalize_text(t) for t in corpus]
  
df= df.dropna(subset=['findings'])
!pip install swifter
import swifter 
df['findings_cleaned'] = df['findings'].swifter.apply(normalize_text)

df['findings_cleaned'] = df['findings'].apply(normalize_text)
df['findings_cleaned'] = 'startseq '+df['findings_cleaned']+' endseq'
num_words = []
for row in df['findings_cleaned'].tolist():
    num_words.append(len(word_tokenize(row)))
num_words= np.array(num_words)
print("min length             : ", num_words.min())
print("max length             : ", num_words.max())
print("50th percentile length : ", np.percentile(num_words,50))
print("75th percentile length : ", np.percentile(num_words,75))
print("90th percentile length : ", np.percentile(num_words,90))
print("95th percentile length : ", np.percentile(num_words,95))
print("98th percentile length : ", np.percentile(num_words,98))
print("98th percentile length : ", np.percentile(num_words,99))

vocab_size = 10000
max_len = 100

tok = Tokenizer(num_words=vocab_size,  oov_token='UNK' )
tok.fit_on_texts(df['findings_cleaned'].tolist())

vocab_size = len(tok.word_index) + 1
vocab_size
from MultiCheXNet.data_loader.indiana_dataloader import get_train_validation_generator
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
max_vocab_size=10000
max_len=100

csv_path1  ="/kaggle/input/chest-xrays-indiana-university/indiana_reports.csv"
csv_path2  ="/kaggle/input/chest-xrays-indiana-university/indiana_projections.csv"
img_path   ="/kaggle/input/chest-xrays-indiana-university/images/images_normalized/"
batch_sz = 8
validation_split = 0.2

train_dataloader, val_dataloader, vocab_size, tok, df = get_train_validation_generator(csv_path1,csv_path2,img_path, max_vocab_size,max_len,preprocess=vgg_preprocess_input, batch_size=batch_sz, validation_split=validation_split)
len(df)
batch, (X,Y)  = next(enumerate(train_dataloader))
batch

#X,Y = next(enumerate(train_dataloader))[1]
len(X)

print(len(X[0]))# images
print(X[0].shape)
#Sample img
smpl_idx = 0
plt.imshow(X[0][smpl_idx])
plt.show()
print(len(X[1]))# Input Text w/o endseq
print(X[1].shape)

Y.shape# Target Text w/o startseq
print(X[1][smpl_idx])
print(Y[smpl_idx])
def tok2txt(tokens, tok):
    return " ".join([tok.index_word[token] for token in tokens if token!=0])
print(tok2txt(X[1][smpl_idx], tok))
print(tok2txt(Y[smpl_idx], tok))
'''
def data_gen(df):
    for i in range(len(df)):
        
        yield img, caption
'''
'''
import tensorflow as tf
def gen():
    #return next(enumerate(train_dataloader))[1]
    return train_dataloader
dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.float32))
'''
#X1,Y1 = next(enumerate(dataset))
class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # attention_hidden_layer shape == (batch_size, 64, units)
    attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                         self.W2(hidden_with_time_axis)))

    # score shape == (batch_size, 64, 1)
    # This gives you an unnormalized score for each image feature.
    score = self.V(attention_hidden_layer)

    # attention_weights shape == (batch_size, 64, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

'''
Densenet_model = tf.keras.applications.DenseNet121(
            include_top=False,
            #weights="imagenet",
            input_shape=(256,256,3),
        )
number_of_encoder_layers=  len(Densenet_model.layers)

encoder_output = Densenet_model(img_input)
encoder_output = layers.Flatten()(encoder_output)
encoder_output = layers.Dropout(0.2)(encoder_output)
encoder_output = layers.Dense(512,activation='relu')(encoder_output)
'''
class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.feat_ext = tf.keras.applications.DenseNet121(
            include_top=False,
            #weights="imagenet",
            input_shape=(256,256,3),
        )
        
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.feat_ext(x)
        #x = tf.keras.layers.Flatten()(x)
        #x = tf.keras.layers.Dropout(0.2)(x)
        #batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
        # shape after fc == (batch_size, 64, embedding_dim)
        x = tf.reshape(x, (x.shape[0], -1, x.shape[3]))
        # shape after fc == (batch_size, 64, embedding_dim)
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x
class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))
embedding_dim = 256
units = 512
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)
lr = 0.0001#0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
start_epoch = 0
if ckpt_manager.latest_checkpoint:
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
  # restoring the latest checkpoint in checkpoint_path
  ckpt.restore(ckpt_manager.latest_checkpoint)
# adding this in a separate cell because if you run the training cell
# many times, the loss_plot array will be reset
loss_plot = []
@tf.function
def train_step(img_tensor, target):
  loss = 0

  # initializing the hidden state for each batch
  # because the captions are not related from image to image
  hidden = decoder.reset_state(batch_size=target.shape[0])

  dec_input = tf.expand_dims([tok.word_index['startseq']] * target.shape[0], 1)


  with tf.GradientTape() as tape:
      features = encoder(img_tensor)

      for i in range(1, target.shape[1]):
          # passing the features through the decoder
          predictions, hidden, _ = decoder(dec_input, features, hidden)

          loss += loss_function(target[:, i], predictions)

          # using teacher forcing
          dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))

  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)

  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss
import time
EPOCHS = 20
num_steps = len(df)*validation_split // batch_sz

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0

    #for (batch, (img_tensor, target)) in enumerate(dataset):
    for (batch, (X,Y)) in enumerate(train_dataloader):
        img_tensor = X[0]
        target = Y
        #print(target.shape)
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
            #print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps)

    if epoch % 5 == 0:
      ckpt_manager.save()

    print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                         total_loss/num_steps))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.show()
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
# Shape of the vector extracted from DenseNet121 is (64, 1024)
# These two variables represent that vector shape
features_shape = 1024
attention_features_shape = 64
encoder.summary()
def evaluate(image):
    attention_plot = np.zeros((max_len, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(image, 0)
    #img_tensor_val = image_features_extract_model(temp_input)
    #img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(temp_input)

    dec_input = tf.expand_dims([tok.word_index['startseq']], 0)
    result = []

    for i in range(max_len):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tok.index_word[predicted_id])

        if tok.index_word[predicted_id] == 'endseq':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot
from PIL import Image
def plot_attention(image, result, attention_plot):
    temp_image = image#np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()
batch, (X,Y)  = next(enumerate(val_dataloader))
#Sample img
smpl_idx = 0

# Caption
#print(tok2txt(X[1][smpl_idx], tok))
real_caption = tok2txt(Y[smpl_idx], tok)
print(real_caption)

# Image
image = X[0][smpl_idx]
plt.imshow(image)
plt.show()



# captions on the validation set
#rid = np.random.randint(0, len(img_name_val))
#image = img_name_val[rid]

#real_caption = ' '.join([tok.index_word[i] for i in cap_val[rid] if i not in [0]])
result, attention_plot = evaluate(image)

print ('Real Caption:', real_caption)
print ('Prediction Caption:', ' '.join(result))
plot_attention(image, result, attention_plot)
from MultiCheXNet.evaluation.report_gen_evaluation import get_predictions_from_data_loader
from copy import deepcopy

val_dataloader_tmp = deepcopy(val_dataloader)
val_dataloader_tmp.nb_iteration  = 3
GT , preds = get_predictions_from_data_loader(val_dataloader_tmp,tok,encoder, decoder,max_len,decoder_type='GRU')



index=0
print(GT[index])
print(("====================================="))
print(preds[index])
index=1
print(GT[index])
print(("====================================="))
print(preds[index])
index=2
print(GT[index])
print(("====================================="))
print(preds[index])
index=3
print(GT[index])
print(("====================================="))
print(preds[index])
index=4 
print(GT[index])
print(("====================================="))
print(preds[index])
evaluate_from_dataloader(val_dataloader,tok,encoder_model,decoder_model,max_len,decoder_type='GRU')
evaluate_from_dataloader(val_dataloader,tok,encoder_model, decoder_model,max_len,decoder_type="LSTM")