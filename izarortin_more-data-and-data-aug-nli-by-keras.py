!pip install transformers==3.0.2
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
os.environ["WANDB_API_KEY"] = "0" ## Para silenciar el aviso que nos daría Colab al no darle una KEY válida.
#Las relacionadas con transformers de Hugging Face.#Las relacionadas con transformers de Hugging Face.



import transformers 

from transformers import BertTokenizer, TFBertModel, AutoTokenizer, TFAutoModel



#Las relacionadas con TensorFlow



import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam, SGD, Adamax

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

from kaggle_datasets import KaggleDatasets



#El train-test split de Scikit-learn.



import sklearn

from sklearn.model_selection import train_test_split



#Las herramientas gráficas necesarias.



import plotly.express as px

import matplotlib.pyplot as plt



#Para acceder a las librerías de datasets de Hugging Face.



!pip install nlp

import nlp



#Resto de funciones



import datetime

from tqdm.notebook import tqdm

import random

from PIL import Image

import requests

from io import BytesIO

import io

import PIL
transformers.__version__
Image.open("../input/tf-logo/tf_logo.png")
Image.open("../input/tpu-v38/tpu--sys-arch4 1.png")
#Detectaremos el hardware requerido, y nos devolverá la distribución adecuada.



try:

    #No es necesario introducir un valor en el "resolver", ya que la enviroment variable TPU_NAME ya está establecida en Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    

    #Le brindamos dicha información al tf.config.experimental_connect_to_cluster, que conecta con el cluster dado    

    tf.config.experimental_connect_to_cluster(tpu)

    

    #Inicializamos el TPU.

    tf.tpu.experimental.initialize_tpu_system(tpu)

    

    #Y definimos la estrategia de distribución.

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    

except ValueError:

    strategy = tf.distribute.get_strategy() 

    print('Number of replicas:', strategy.num_replicas_in_sync)
train = pd.read_csv("../input/contradictory-my-dear-watson/train.csv")



test = pd.read_csv("../input/contradictory-my-dear-watson/test.csv")



submission = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/sample_submission.csv')
response = requests.get('https://miro.medium.com/max/700/1*Nv2NNALuokZEcV6hYEHdGA.png')

image_bytes = io.BytesIO(response.content)



img = PIL.Image.open(image_bytes)

img
train.head()
train.premise.values[1]
train.hypothesis.values[1]
train.label.values[1]
labels, frequencies = np.unique(train.language.values, return_counts = True)



plt.figure(figsize = (10,10))

plt.pie(frequencies,labels = labels, autopct = '%1.1f%%')

plt.show()
train.isnull().sum()
#Definimos los dos modelos a estudiar

model_xml_roberta = 'jplu/tf-xlm-roberta-large'



model_bert = 'bert-base-multilingual-cased'



#Definimos el número de epochs a iterar y el max_len

n_epochs = 10

max_len = 80



# El tamaño de nuestro batch_size dependerá del número de réplicas en nuestra estrategia

batch_size = 16 * strategy.num_replicas_in_sync
Image.open("../input/table-accurancy-models/Table accurancy models.png")
response = requests.get('http://jalammar.github.io/images/bert-base-bert-large-encoders.png')

image_bytes = io.BytesIO(response.content)



img = PIL.Image.open(image_bytes)

img
response = requests.get('http://jalammar.github.io/images/bert-encoders-input.png')

image_bytes = io.BytesIO(response.content)



img = PIL.Image.open(image_bytes)

img
response = requests.get('http://jalammar.github.io/images/bert-output-vector.png')

image_bytes = io.BytesIO(response.content)



img = PIL.Image.open(image_bytes)

img
response = requests.get('https://cdn.analyticsvidhya.com/wp-content/uploads/2019/09/bert_emnedding.png')

image_bytes = io.BytesIO(response.content)



img = PIL.Image.open(image_bytes)

img
response = requests.get('https://cdn.analyticsvidhya.com/wp-content/uploads/2019/09/bert-vs-openai-.jpg')

image_bytes = io.BytesIO(response.content)



img = PIL.Image.open(image_bytes)

img
response = requests.get('https://miro.medium.com/max/2272/1*9cRchmIyxP4LUnONXLM82g.png')

image_bytes = io.BytesIO(response.content)



img = PIL.Image.open(image_bytes)

img
response = requests.get('https://images.deepai.org/converted-papers/1911.02116/x7.png')

image_bytes = io.BytesIO(response.content)



img = PIL.Image.open(image_bytes)

img
# Cargamos el tokenizador correspondiente

tokenizer_bert = AutoTokenizer.from_pretrained(model_bert)
# Transformaremos el texto en listas, para poder introducirlas en el batch_encode_plus

train_text_bert = train[['premise', 'hypothesis']].values.tolist()

test_text_bert = test[['premise', 'hypothesis']].values.tolist()



# Ahora utilizaremos el tokenizador que hemos preparado previamente.

train_encoded_bert = tokenizer_bert.batch_encode_plus(

    train_text_bert,

    pad_to_max_length=True,

    max_length=max_len

)



test_encoded_bert = tokenizer_bert.batch_encode_plus(

    test_text_bert,

    pad_to_max_length=True,

    max_length=max_len

)
x_train, x_valid, y_train, y_valid = train_test_split(

    train_encoded_bert['input_ids'], train.label.values, 

    test_size=0.2, random_state=2020

)



x_test = test_encoded_bert['input_ids']
auto = tf.data.experimental.AUTOTUNE



train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .shuffle(2048)

    .batch(batch_size)

    .prefetch(auto)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_valid, y_valid))

    .batch(batch_size)

    .cache()

    .prefetch(auto)

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .batch(batch_size)

)
with strategy.scope():

    # Primero cargamos la capa de codificador.

    transformer_encoder = TFAutoModel.from_pretrained(model_bert)



    # Definimos los inputs tokenizados

    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")



    # Ahora, codificamos los inputs segun el encoder que hemos definido anteriormente.

    sequence_output = transformer_encoder(input_ids)[0]



    # Extraemos los tokens utilizados para clasificar, en este caso [CLS]

    cls_token = sequence_output[:, 0, :]



    # La última capa es la que pasamos a través de softmax para la aplicación del uso de probabilidades. Con 3 niveles, ya que son 3 posibles resultados (0,1 y 2)

    out = Dense(3, activation='softmax')(cls_token)



    # Construimos y compilamos el modelo.

    model = Model(inputs=input_ids, outputs=out)

    model.compile(

        Adam(lr=1e-5), 

        loss='sparse_categorical_crossentropy', 

        metrics=['accuracy']

    )



model.summary()
n_steps = len(x_train) // batch_size



train_history_bert = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    epochs=n_epochs

)
# list all data in history

print(train_history_bert.history.keys())

# summarize history for loss

plt.plot(train_history_bert.history['loss'])

plt.plot(train_history_bert.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# summarize history for accuracy

plt.plot(train_history_bert.history['accuracy'])

plt.plot(train_history_bert.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
# Cargamos el tokenizador correspondiente

tokenizer_xml_roberta = AutoTokenizer.from_pretrained(model_xml_roberta)
# Transformaremos el texto en listas, para poder introducirlas en el batch_encode_plus

train_text_xml_roberta = train[['premise', 'hypothesis']].values.tolist()

test_text_xml_roberta = test[['premise', 'hypothesis']].values.tolist()



# Ahora utilizaremos el tokenizador que hemos preparado previamente.

train_encoded_xml_roberta = tokenizer_xml_roberta.batch_encode_plus(

    train_text_xml_roberta,

    pad_to_max_length=True,

    max_length=max_len

)



test_encoded_xml_roberta = tokenizer_xml_roberta.batch_encode_plus(

    test_text_xml_roberta,

    pad_to_max_length=True,

    max_length=max_len

)
x_train, x_valid, y_train, y_valid = train_test_split(

    train_encoded_xml_roberta['input_ids'], train.label.values, 

    test_size=0.2, random_state=2020

)



x_test = test_encoded_xml_roberta['input_ids']
auto = tf.data.experimental.AUTOTUNE



train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .shuffle(2048)

    .batch(batch_size)

    .prefetch(auto)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_valid, y_valid))

    .batch(batch_size)

    .cache()

    .prefetch(auto)

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .batch(batch_size)

)
with strategy.scope():

    # Primero cargamos la capa de codificador.

    transformer_encoder = TFAutoModel.from_pretrained(model_xml_roberta)



    # Definimos los inputs tokenizados

    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")



    # Ahora, codificamos los inputs segun el encoder que hemos definido anteriormente.

    sequence_output = transformer_encoder(input_ids)[0]



    # Extraemos los tokens utilizados para clasificar, en este caso <s>

    cls_token = sequence_output[:, 0, :]



    # La última capa es la que pasamos a través de softmax para la aplicación del uso de probabilidades. Con 3 niveles, ya que son 3 posibles resultados (0,1 y 2)

    out = Dense(3, activation='softmax')(cls_token)



    # Construimos y compilamos el modelo.

    model = Model(inputs=input_ids, outputs=out)

    model.compile(

        Adam(lr=1e-5), 

        loss='sparse_categorical_crossentropy', 

        metrics=['accuracy']

    )



model.summary()
n_steps = len(x_train) // batch_size



train_history_xml_roberta = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    epochs=n_epochs

)
# list all data in history

print(train_history_xml_roberta.history.keys())

# summarize history for loss

plt.plot(train_history_xml_roberta.history['loss'])

plt.plot(train_history_xml_roberta.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# summarize history for accuracy

plt.plot(train_history_xml_roberta.history['accuracy'])

plt.plot(train_history_xml_roberta.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
response = requests.get('https://miro.medium.com/max/398/1*BfvKeP4ykqi4J4C5g4EZzg.png')

image_bytes = io.BytesIO(response.content)



img = PIL.Image.open(image_bytes)

img
mnli = nlp.load_dataset(path='glue', name='mnli')
mnli_train_df = pd.DataFrame(mnli['train'])



mnli_train_df = mnli_train_df[['premise', 'hypothesis', 'label']]



mnli_train_df['lang_abv'] = 'en'
mnli_train_df.head(10)
mnli_sample= mnli_train_df.sample(n = 40000,random_state= 2020)
mnli_sample.head(10)
xnli = nlp.load_dataset(path='xnli')
buffer = {

    'premise': [],

    'hypothesis': [],

    'label': [],

    'lang_abv': []

}





for x in xnli['validation']:

    label = x['label']

    for idx, lang in enumerate(x['hypothesis']['language']):

        hypothesis = x['hypothesis']['translation'][idx]

        premise = x['premise'][lang]

        buffer['premise'].append(premise)

        buffer['hypothesis'].append(hypothesis)

        buffer['label'].append(label)

        buffer['lang_abv'].append(lang)

        

# convert to a dataframe and view

xnli_valid_df = pd.DataFrame(buffer)

xnli_valid_df = xnli_valid_df[['premise', 'hypothesis', 'label', 'lang_abv']]
xnli_valid_df.head(10)
response = requests.get('https://miro.medium.com/max/750/0*cZwrV8EyfJSeunag.png')

image_bytes = io.BytesIO(response.content)



img = PIL.Image.open(image_bytes)

img
response = requests.get('https://d3i71xaburhd42.cloudfront.net/f3ee8dcaaad5f47f347354fe5d740096097cbed5/1-Figure1-1.png')

image_bytes = io.BytesIO(response.content)



img = PIL.Image.open(image_bytes)

img
!pip -q install googletrans

import gc

from googletrans import Translator

from dask import bag, diagnostics

import seaborn as sns
SEED = 42

os.environ['PYTHONHASHSEED']=str(SEED)

random.seed(SEED)

np.random.seed(SEED)
def translate(words, dest):

    dest_choices = ['zh-cn',

                    'ar',

                    'fr',

                    'sw',

                    'ur',

                    'vi',

                    'ru',

                    'hi',

                    'el',

                    'th',

                    'es',

                    'de',

                    'tr',

                    'bg'

                    ]

    if not dest:

        dest = np.random.choice(dest_choices)

        

    translator = Translator()

    decoded = translator.translate(words, dest=dest).text

    return decoded



def trans_parallel(df, dest):

    premise_bag = bag.from_sequence(df.premise.tolist()).map(translate, dest)

    hypo_bag =  bag.from_sequence(df.hypothesis.tolist()).map(translate, dest)

    with diagnostics.ProgressBar():

        premises = premise_bag.compute()

        hypos = hypo_bag.compute()

    df[['premise', 'hypothesis']] = list(zip(premises, hypos))

    return df
eng = train.loc[train.lang_abv == "en"].copy().pipe(trans_parallel, dest=None)

non_eng =  train.loc[train.lang_abv != "en"].copy().pipe(trans_parallel, dest='en')

train_data_translate = train.append([eng, non_eng])
frames = [train_data_translate, xnli_valid_df, mnli_sample]

train_data_def = pd.concat(frames)
tf.tpu.experimental.initialize_tpu_system(tpu)
train_data_def.head()
labels, frequencies = np.unique(train_data_def.lang_abv.values, return_counts = True)



plt.figure(figsize = (10,10))

plt.pie(frequencies,labels = labels, autopct = '%1.1f%%')

plt.show()
# Transformaremos el texto en listas, para poder introducirlas en el batch_encode_plus

train_text_xml_roberta = train_data_def[['premise', 'hypothesis']].values.tolist()

test_text_xml_roberta = test[['premise', 'hypothesis']].values.tolist()



# Ahora utilizaremos el tokenizador que hemos preparado previamente.

train_encoded_xml_roberta = tokenizer_xml_roberta.batch_encode_plus(

    train_text_xml_roberta,

    pad_to_max_length=True,

    max_length=max_len

)



test_encoded_xml_roberta = tokenizer_xml_roberta.batch_encode_plus(

    test_text_xml_roberta,

    pad_to_max_length=True,

    max_length=max_len

)
x_train, x_valid, y_train, y_valid = train_test_split(

    train_encoded_xml_roberta['input_ids'], train_data_def.label.values, 

    test_size=0.2, random_state=2020

)



x_test = test_encoded_xml_roberta['input_ids']
auto = tf.data.experimental.AUTOTUNE



train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .shuffle(2048)

    .batch(batch_size)

    .prefetch(auto)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_valid, y_valid))

    .batch(batch_size)

    .cache()

    .prefetch(auto)

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .batch(batch_size)

)
response = requests.get('https://miro.medium.com/max/1100/1*zfdW5zAyQxge85gA_mFPYg.png')

image_bytes = io.BytesIO(response.content)



img = PIL.Image.open(image_bytes)

img
response = requests.get('https://paperswithcode.com/media/methods/Screen_Shot_2020-05-28_at_6.15.37_PM_apRrZCo.png')

image_bytes = io.BytesIO(response.content)



img = PIL.Image.open(image_bytes)

img
response = requests.get('https://www.jeremyjordan.me/content/images/2018/02/Screen-Shot-2018-02-24-at-11.47.09-AM.png')

image_bytes = io.BytesIO(response.content)



img = PIL.Image.open(image_bytes)

img
fine_tuning_lr = pd.read_csv("../input/fine-tuning/table tuning lr.csv", sep = ';')



fine_tuning_lr
fine_tuning_lr = pd.read_csv("../input/fine-tuning/epsilon table.csv", sep = ';')



fine_tuning_lr
with strategy.scope():

    # Primero cargamos la capa de codificador.

    transformer_encoder = TFAutoModel.from_pretrained(model_xml_roberta)



    # Definimos los inputs tokenizados

    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")



    # Ahora, codificamos los inputs segun el encoder que hemos definido anteriormente.

    sequence_output = transformer_encoder(input_ids)[0]



    # Extraemos los tokens utilizados para clasificar, en este caso <s>

    cls_token = sequence_output[:, 0, :]



    # La última capa es la que pasamos a través de softmax para la aplicación del uso de probabilidades. Con 3 niveles, ya que son 3 posibles resultados (0,1 y 2)

    out = Dense(3, activation='softmax')(cls_token)



    # Construimos y compilamos el modelo.

    model = Model(inputs=input_ids, outputs=out)

    model.compile(

        Adam(lr=2e-5,beta_1=0.9, beta_2=0.999, epsilon=5e-08), 

        loss='sparse_categorical_crossentropy', 

        metrics=['accuracy']

    )



model.summary()
n_steps = len(x_train) // batch_size



train_history_xml_roberta_def = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    epochs=3

)
# list all data in history

print(train_history_xml_roberta_def.history.keys())

# summarize history for loss

plt.plot(train_history_xml_roberta_def.history['loss'])

plt.plot(train_history_xml_roberta_def.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# summarize history for accuracy

plt.plot(train_history_xml_roberta_def.history['accuracy'])

plt.plot(train_history_xml_roberta_def.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
test_preds = model.predict(test_dataset, verbose=1)

submission['prediction'] = test_preds.argmax(axis=1)
submission.head()
submission.to_csv("submission.csv", index = False)