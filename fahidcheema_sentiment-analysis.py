# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# !pip install urduhack tensorflow_gpu
import tensorflow as tf
from urduhack import normalize
from urduhack.preprocess import remove_punctuation, normalize_whitespace
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout
from tensorflow.keras.layers import Concatenate, Conv1D, MaxPool1D, LSTM, BatchNormalization
from transformers import XLMRobertaTokenizer, TFXLMRobertaModel
FILTER_SIZES = [3, 4]
NUM_FILTERS = 4
MAX_SEQUENCE_LENGTH = 512
imdb_df = pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-translated-urdu-reviews/urdu_imdb_dataset.csv")
imdb_df.head()
imdb_df.info()
imdb_df.describe()
imdb_df.columns
plt.hist(imdb_df["sentiment"]);
imdb_df["review"][0]
# Removing punctuations and unnecessary whitespaces from the text
imdb_df["review"] = imdb_df["review"].apply(remove_punctuation)
imdb_df["review"] = imdb_df["review"].apply(normalize_whitespace)
imdb_df["review"] = imdb_df["review"].apply(normalize)
#clean and normalized text
imdb_df["review"][0]
tokenizer = XLMRobertaTokenizer.from_pretrained("jplu/tf-xlm-roberta-base")
xlm_roberta = TFXLMRobertaModel.from_pretrained("jplu/tf-xlm-roberta-base")
xlm_roberta.trainable = False
tokens = tokenizer.tokenize(imdb_df["review"][0])
tokens
token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(imdb_df["review"][0]))
token_ids
#sentence output gives token level vector representation
#pooled output gives sentence level vector representation
sentence_output, pooled_output = xlm_roberta(np.array([token_ids]))

print(sentence_output.shape, pooled_output.shape)
for i in range(2):
    print(str(tokens[i]) + ": " + str(sentence_output[:, i, :]))
texts = imdb_df["review"].values.tolist()
#tokenizer.encode directly converts text to tokens to ids and adds special_tokens(cls_tokens, unk_token, sep_tokens) as well as padding
encoded = []
for text in texts:
    input_ids = tokenizer.encode(text, max_length=MAX_SEQUENCE_LENGTH, pad_to_max_length=True, add_special_tokens=True)
    encoded.append(input_ids)
data = np.array(encoded)
labels = imdb_df["sentiment"].values.tolist()
from sklearn.preprocessing import LabelEncoder
en = LabelEncoder()
labels = en.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels)
n_classes = len(np.unique(labels))
VALIDATION_SPLIT = 0.15
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
sequence_ouput, _ = xlm_roberta(sequence_input)

conv_blocks = []
for k_size in FILTER_SIZES:
    conv = Conv1D(filters=NUM_FILTERS,
                         kernel_size=k_size,
                         padding="valid",
                         activation="relu",
                         strides=1)(sequence_ouput)
    conv = Dropout(0.25)(conv)
    conv = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH - k_size + 1)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
concat = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
x = Dropout(0.2)(concat)
x = Dense(512, activation="relu")(x)
preds = Dense(n_classes, activation="softmax")(x)
model = tf.keras.Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
              metrics=['acc'])
model.summary()
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='auto')
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=10, batch_size=16, callbacks=[es])
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show();
    
plot_graphs(history, "acc");
plot_graphs(history, "loss");
text = "اگر آپ ہیل بوائے کے ساتھ کھوئے ہوئے صندوق کے چھاپوں کو عبور کرتے ہیں اور پرل ہاربر اور ہدایت کار جو جانسٹن کے اپنے دی روکیٹیئر کو شامل کرتے ہیں تو ، آپ اس فلم کی تخلیق کرنے کے راستے میں ٹھیک ہوجائیں گے۔ یہ اچھی طرح سے چل رہا ہے ، محب وطن جنگ عظیم دوم کی مہم جوئی سائنس فائی / فنتاسی تصوف کا ایک کم جمہوریہ اور ہیرو ہے جس کی جڑیں اکٹھا کرنا بہت آسان ہے۔ راجرز / کیپٹن امریکہ ، بہت اچھا آدمی ہے ، حقیقت میں ، کہ وہ کبھی کبھی تھوڑا سا کمزور ہوتا ہے - وہ یقینا a ایک بہت بڑا رول ماڈل ہے ، لیکن آپ اپنے آپ کو یہ خواہش پاسکتے ہیں کہ اس کے پاس تھوڑا سا ولورائن یا ٹونی اسٹارک کی سنیارک تھی۔ (فلم میں مجموعی طور پر مزاح سے تھوڑا سا مختصر ہے ، دراصل ، کیپٹن امریکہ کی رگ ٹیگ آف سپاہیوں کی ٹیم زیادہ تر قہقہوں دیتی ہے ، لیکن ان میں اتنا زیادہ سکرین ٹائم نہیں ملتا ہے۔)"
text = remove_punctuation(text)
text = normalize_whitespace(text)
text = normalize(text)
encoded = tokenizer.encode(text, add_special_tokens=True, max_length=MAX_SEQUENCE_LENGTH, pad_to_max_length=True)
encoded = np.asarray([encoded])
predictions = model.predict(encoded)
print(np.argmax(predictions))