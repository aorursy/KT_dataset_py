!pip install flair

from flair.embeddings import Sentence

from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, ELMoEmbeddings, FlairEmbeddings

# FastText = WordEmbeddings('en-crawl')

# Komninos = WordEmbeddings('en-extvec')

# glove = WordEmbeddings('en-glove')

ELMo = ELMoEmbeddings('medium')

flair_forward = FlairEmbeddings('news-forward')

flair_backward = FlairEmbeddings('news-backward')



# initialize the document embeddings, mode = mean

document_embeddings = DocumentPoolEmbeddings([

#     glove,

#     FastText,

#     Komninos, 

    ELMo, 

    flair_forward, 

    flair_backward, 

])





def get_flaired(x_):

    try:

        x_ = Sentence(x_)

        document_embeddings.embed(x_)

    except TypeError:

        x_ = 'TypeError'

        x_ = Sentence(x_)

        document_embeddings.embed(x_)

    return x_.get_embedding().detach().cpu().numpy()

import pandas as pd

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

import numpy as np



def embed(data_):

    data_['Province_country'] = pd.DataFrame([

        data_.iloc[i, 1] + ' ' + data_.iloc[i, 2] 

        if isinstance(data_.iloc[i, 1], str)

        else data_.iloc[i, 2] 

        for i in range(len(data_))])

    embedding = np.array([get_flaired(i) for i in data_['Province_country']])

    return embedding
import datetime

def day_numbers(data_):

    day_number = [(

        datetime.datetime.strptime(data_.iloc[i, 5], '%Y-%m-%d').date()- datetime.date(2020, 1, 22)

    ).days for i in range(len(data_))]

    return np.array(day_number)
x1 = embed(train)

x2_ = day_numbers(train)

y1_ = train.ConfirmedCases.to_numpy()

y2_ = train.Fatalities.to_numpy()
x2 = x2_ / np.max(x2_)

y1 = y1_ / np.max(y1_)

y2 = y2_ / np.max(y2_)
!pip install tensorflow-gpu==2.1
from tensorflow_core.python.keras.layers import Input, Dense, Dropout, Concatenate

from tensorflow_core.python.keras.models import Model

from tensorflow_core.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

early = EarlyStopping(monitor='loss', min_delta=0,

                      patience=7, verbose=1, mode='auto',

                      baseline=None, restore_best_weights=False)

learn = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, verbose=1,

                          mode='auto', min_delta=0.0001,

                          cooldown=10, min_lr=0)

    
def create_model(batch_size, embedding_dimensions):

    units=512

    activations='relu'

    use_bias=True

    dropout=0.3

    kernel_init='he_normal'

    bias_init='he_uniform'

    optimizer='nadam'

    loss='mean_squared_error'

    batch_size_=batch_size

    embed_dims=embedding_dimensions

    embeddings = Input(name='embeddings', batch_shape=(batch_size_, embed_dims))

    day_numbers = Input(name='day_numbers', batch_shape=(batch_size_, 1))

    x = Concatenate(axis=1)([embeddings, day_numbers])

    x0 = Dense(units=units, activation=activations, use_bias=use_bias, kernel_initializer=kernel_init,

               bias_initializer=bias_init, name='Dense_1')(x)

    x0 = Dropout(dropout, name='Dropout_1')(x0)

    x0 = Dense(units=units, activation=activations, use_bias=use_bias, kernel_initializer=kernel_init,

               bias_initializer=bias_init, name='Dense_2')(x0)



    x0 = Dropout(dropout, name='Dropout_2')(x0)

    x0 = Dense(units=int(units / 2), activation=activations, use_bias=use_bias, kernel_initializer=kernel_init,

               bias_initializer=bias_init, name='Dense_3')(x0)

    x0 = Dropout(dropout, name='Dropout_3')(x0)

    

    x1 = Dense(units=int(units / 2), activation=activations, use_bias=use_bias, kernel_initializer=kernel_init,

               bias_initializer=bias_init, name='Dense_4')(x0)

    x1 = Dropout(dropout, name='Dropout_4')(x1)



    y1 = Dense(1, activation='linear', use_bias=use_bias, kernel_initializer=kernel_init,

              bias_initializer=bias_init, name='Output_1')(x1)

    

    x2 = Concatenate(axis=1)([y1, x0])



    x2 = Dense(units=int(units / 2), activation=activations, use_bias=use_bias, kernel_initializer=kernel_init,

               bias_initializer=bias_init, name='Dense_5')(x2)

    x2 = Dropout(dropout, name='Dropout_5')(x2)



    y2 = Dense(1, activation='linear', use_bias=use_bias, kernel_initializer=kernel_init,

              bias_initializer=bias_init, name='Output_2')(x2)





    model_ = Model(inputs=[embeddings, day_numbers], outputs=[y1, y2])

    model_.compile(optimizer=optimizer, loss=loss)

    return model_

model = create_model(batch_size=None, embedding_dimensions=x1.shape[1])

history = model.fit(x=[x1, x2], y=[y1, y2], 

                    epochs=1000, 

                    batch_size=32,

                    verbose=2,

                    callbacks=[early, learn])

test_x1 = embed(test)

test_x2 = day_numbers(test)

test_x2 = test_x2 / np.max(x2_)

ConfirmedCases_, Fatalities_ = model.predict([test_x1, test_x2])
ConfirmedCases = ConfirmedCases_ * np.max(y1_)

Fatalities = Fatalities_ * np.max(y2_)

for i in range(len(ConfirmedCases)):

    if ConfirmedCases[i] > 0:

        pass

    else:

        ConfirmedCases[i] = 0

for i in range(len(Fatalities)):

    if Fatalities[i] > 0:

        pass

    else:

        Fatalities[i] = 0
submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')

for i in range(len(ConfirmedCases)):

    submission.iloc[i, 1] = ConfirmedCases[i]

    submission.iloc[i, 2] = Fatalities[i]

submission.head
submission.to_csv('submission.csv', index=False)