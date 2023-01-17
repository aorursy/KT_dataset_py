import numpy as np

import pandas as pd

from random import sample



from tensorflow import keras
columns = ['ID', 'Book length (mins)_overall',

           'Book length (mins)_avg', 'Price_overall',

           'Price_avg', 'Review',

           'Review 10/10', 'Minutes listed',

           'Completion', 'Support Requests',

           'Last visited minus Purchase date','Target']
data_audio_book = pd.read_csv('../input/audiobooks-data/Audiobooks_data.csv', names=columns, index_col='ID')
data_audio_book.Target.value_counts()
ID_converted_all = list(data_audio_book[data_audio_book.Target == 1].index)

ID_not_converted_all = list(data_audio_book[data_audio_book.Target == 0].index)
size = len(ID_converted_all)



ID_not_converted_equal = sample(ID_not_converted_all, size)
len(ID_not_converted_equal)
data_balanced = data_audio_book.loc[ID_not_converted_equal + ID_converted_all]
standardize_columns = ['Book length (mins)_overall',

                       'Book length (mins)_avg', 'Price_overall',

                       'Price_avg', 'Review 10/10', 'Minutes listed',

                       'Completion', 'Support Requests',

                       'Last visited minus Purchase date']





data_balanced[standardize_columns] = (data_balanced[standardize_columns] - data_balanced[standardize_columns].mean())/data_balanced[standardize_columns].std()
inputs = data_balanced.values[:, :-1]

targets = data_balanced.values[:, -1]
shuffled_indices = np.arange(inputs.shape[0])

np.random.shuffle(shuffled_indices)



shuffled_inputs = inputs[shuffled_indices]

shuffled_targets = targets[shuffled_indices]
samples_count = shuffled_inputs.shape[0]



train_samples_count = int(0.8*samples_count)

validation_samples_count = int(0.1*samples_count)

test_samples_count = samples_count - train_samples_count - validation_samples_count



train_inputs = shuffled_inputs[:train_samples_count]

train_targets = shuffled_targets[:train_samples_count]



validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]

validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]



test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]

test_targets = shuffled_targets[train_samples_count+validation_samples_count:]



print(train_targets.sum()/len(train_targets))

print(validation_targets.sum()/len(validation_targets))

print(test_targets.sum()/len(test_targets))
np.savez('Audiobooks_data_train', intpus = train_inputs, targets=train_targets)

np.savez('Audiobooks_data_validation', intpus = validation_inputs, targets=validation_targets)

np.savez('Audiobooks_data_test', intpus = test_inputs, targets=test_targets)
m = keras.Sequential([

    keras.layers.Dense(50, activation='relu'),

    keras.layers.Dense(50, activation='relu'),

    keras.layers.Dense(2, activation='softmax')    

])



m.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



bacth_size = 100



max_apochs = 100



early_stopping = keras.callbacks.EarlyStopping(patience=2)





m.fit(train_inputs, train_targets, 

      batch_size=bacth_size, 

      epochs = max_apochs,

      callbacks=[early_stopping],

      validation_data = (validation_inputs, validation_targets),

      verbose=2

     )