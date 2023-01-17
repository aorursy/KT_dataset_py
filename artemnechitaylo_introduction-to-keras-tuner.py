!pip install keras-tuner
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from kerastuner.tuners import RandomSearch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
max_words = 15000
max_len = 150
train_data = '/kaggle/input/nlp-getting-started/train.csv'
test_data = '/kaggle/input/nlp-getting-started/test.csv'
df_train = pd.read_csv(train_data)
df_test = pd.read_csv(test_data)
df_train.head()
x_train = df_train['text']
y_train = df_train['target']
counts = y_train.value_counts()
groups = ['Fake', 'True']
colors = ['b', 'g']
plt.title('Amount of data')

width = len(counts.array) * 0.2
plt.bar(groups, counts.array, width=width, color=colors, alpha=0.6, bottom=2, linewidth=2)
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
proportion_data = len(x_train) - int(len(x_train) * 0.2)
x_test = x_train[proportion_data:]
y_test = y_train[proportion_data:]
x_train = x_train[:proportion_data]
y_train = y_train[:proportion_data]
test_ids = df_test['id']
x_test1 = df_test['text']
x_test1 = tokenizer.texts_to_sequences(x_test1)
x_train = pad_sequences(x_train, maxlen=max_len, padding='post')
x_test = pad_sequences(x_test, maxlen=max_len, padding='post')
x_test1 = pad_sequences(x_test1, maxlen=max_len, padding='post')
def build_model(hp):
    model = Sequential()
    activation_choice = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh', 'elu', 'selu'])
    model.add(Embedding(max_words, hp.Int('output_dim',
                                            min_value=4,
                                            max_value=64,
                                            step=4),
                       input_length=max_len))
    model.add(Conv1D(hp.Int('filters',
                         min_value=100,
                         max_value=250,
                         step=25),
                     hp.Int('kernel_size',
                         min_value=1,
                         max_value=5,
                         step=1),
                    activation=activation_choice))
    model.add(MaxPooling1D(hp.Int('pool_size',
                         min_value=1,
                         max_value=5,
                         step=1)))
    model.add(Conv1D(hp.Int('filters',
                         min_value=250,
                         max_value=500,
                         step=50),
                     hp.Int('kernel_size',
                         min_value=1,
                         max_value=5,
                         step=1),
                    activation=activation_choice))
    model.add(MaxPooling1D(hp.Int('pool_size',
                         min_value=1,
                         max_value=5,
                         step=1)))
    model.add(Dense(units=hp.Int('units_input',
                                   min_value=512,
                                   max_value=1024,
                                   step=32),
                    activation=activation_choice))
    model.add(Dropout(hp.Float('rate',
                              min_value=0.0,
                              max_value=0.3)))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam','RMSprop','SGD']),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model
models_dir = '/kaggle/working/models'
tuner = RandomSearch(
    build_model,                 
    objective='val_accuracy',     
    max_trials=10,               
    directory=models_dir)
tuner.search_space_summary()
tuner.search(x_train,
             y_train,
             batch_size=128,
             epochs=20,
             validation_split=0.2)
models = tuner.get_best_models(num_models=3)
evaluating = []
for model in models:
    model.summary()
    evaluating.append(model.evaluate(x_test, y_test, verbose=1))
    print()
print('Three best models:')
for e in evaluating:
    print(f'Loss: {e[0]}')
    print(f'Accuracy: {e[1]}')
    print('-----')
    print()