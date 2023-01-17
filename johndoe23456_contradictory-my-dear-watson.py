from pandas import read_csv, DataFrame



data = read_csv("../input/contradictory-my-dear-watson/train.csv")

data.head()
len(data)
!pip install nlp
from nlp import load_dataset



extra_data = load_dataset(path='glue', name='mnli')

mnli = []

for sample in extra_data['train']:

    mnli.append([sample['premise'], sample['hypothesis'], sample['label']])

del extra_data

mnli = DataFrame(mnli, columns=['premise','hypothesis','label'])

len(mnli)
from tensorflow.distribute.cluster_resolver import TPUClusterResolver

from tensorflow.config import experimental_connect_to_cluster

from tensorflow.tpu.experimental import initialize_tpu_system

from tensorflow.distribute.experimental import TPUStrategy

from tensorflow.distribute import get_strategy



try:

    tpu = TPUClusterResolver()

    experimental_connect_to_cluster(tpu)

    initialize_tpu_system(tpu)

    strategy = TPUStrategy(tpu)

    print('using TPU...')

except ValueError:

    strategy = get_strategy() # for CPU and single GPU

    print('not using TPU...')
from transformers import AutoTokenizer, TFAutoModel



pretrained = 'jplu/tf-xlm-roberta-large'

tokenizer = AutoTokenizer.from_pretrained(pretrained)
from tensorflow import data as d

from tensorflow.data.experimental import AUTOTUNE

from tensorflow.keras.preprocessing.sequence import pad_sequences



# average length of encoded sequences in test set is ~44

# bigger maximum length might need more than given 16GB of RAM

max_len = 256



def generator(dataset, batch_size):

        

    texts = dataset[['premise', 'hypothesis']].values.tolist()

    encoded = tokenizer.batch_encode_plus(texts, max_length=max_len, padding=True, truncation=True)['input_ids']

    label = dataset['label'].values

    

    # padding with 0

    data_tensor = d.Dataset.from_tensor_slices((pad_sequences(encoded, padding='post', value=0), label))

    

    return data_tensor.shuffle(2048).batch(batch_size).prefetch(AUTOTUNE)
# bigger batch size might need more RAM

train = generator(mnli, 256)

val = generator(data, 256)
from gc import collect



del data, mnli

collect()
from tensorflow.keras.layers import Input, Dropout, Dense, GlobalAveragePooling1D

from tensorflow.keras import Model

from tensorflow import int32

from tensorflow.keras.metrics import CategoricalAccuracy



def build(dropout_rate, optimizer, loss):

    inputs = Input(shape=(max_len,), dtype=int32)

    layers = TFAutoModel.from_pretrained(pretrained)(inputs)[0]

#     layers = Dropout(rate=dropout_rate)(layers)

    layers = GlobalAveragePooling1D()(layers)

    outputs = Dense(3, activation='softmax')(layers)

    model = Model(inputs=inputs, outputs=outputs)

    model.layers[1].trainable = False

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model
from tensorflow.keras.optimizers import Adam



with strategy.scope():

    dropout_rate = 0

    optimizer = Adam(lr=1e-5)

    loss = 'sparse_categorical_crossentropy'

    model = build(dropout_rate=dropout_rate, optimizer='adam',loss=loss)

    model.summary()
# Time limit is 2 hours

# Theoretically, one could train for infinite time by saving weights after each session then using the final weights for evaluation

hst = model.fit(train, epochs=20, verbose=1, validation_data=val)
import matplotlib.pyplot as plt



def visualize(metric):

    plt.plot(hst.history[metric])

    plt.plot(hst.history['val_' + metric])

    plt.title(metric)

    plt.ylabel(metric)

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
visualize('loss')
visualize('accuracy')
data_test = read_csv("../input/contradictory-my-dear-watson/test.csv")

texts = data_test[['premise', 'hypothesis']].values.tolist()

encoded = tokenizer.batch_encode_plus(texts, max_length=max_len, padding='max_length', truncation=True)['input_ids']

encoded_data_test = d.Dataset.from_tensor_slices(pad_sequences(encoded, padding='post', value=0))

encoded_data_test = encoded_data_test.batch(256)
prediction = model.predict(encoded_data_test, verbose=1)

prediction
submission = data_test.id.copy().to_frame()

submission['prediction'] = prediction.argmax(1)

submission.to_csv("submission.csv", index = False)