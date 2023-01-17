!pip install git+https://github.com/keras-team/keras-tuner.git -q
import pandas as pd

full_train_dataframe = pd.read_csv('../input/titanic/train.csv')
test_dataframe = pd.read_csv('../input/titanic/test.csv')
full_train_dataframe.head()
import numpy as np

def fill_nan(df, mean_age):
    df['Age'].fillna(value=mean_age, inplace=True)
    
# Create training and validation datasets
val_dataframe = full_train_dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = full_train_dataframe.drop(val_dataframe.index)
mean_age = np.mean(train_dataframe['Age'])
    
print("Total number of training samples: %d" % (len(full_train_dataframe)))
print("Total number of test samples: %d" % (len(test_dataframe)))
print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

fill_nan(train_dataframe, mean_age)
fill_nan(val_dataframe, mean_age)
fill_nan(full_train_dataframe, mean_age)
fill_nan(test_dataframe, mean_age)
import tensorflow as tf

def dataframe_to_dataset(dataframe, train=True):
    dataframe = dataframe.copy()

    # Drop useless features
    dataframe.pop("Cabin")
    dataframe.pop("Name")
    dataframe.pop("Ticket")
    dataframe.pop("Embarked")
    dataframe.pop("PassengerId")
    
    if train:
        # Set aside labels
        labels = dataframe.pop("Survived")
        # Create dataset
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        ds = ds.shuffle(buffer_size=len(dataframe))
    else:
        ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))
    return ds

train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)
test_ds = dataframe_to_dataset(test_dataframe, train=False)
full_train_ds = dataframe_to_dataset(full_train_dataframe)

# Visualize the names and types of the features in one sample
for sample in train_ds.take(1):
    for key in sample[0].keys():
        print('Feature:', key, '- dtype:', sample[0][key].dtype.name)
        
# Batch the datasets and configure prefetching
train_ds = train_ds.batch(32).prefetch(32)
val_ds = val_ds.batch(32).prefetch(32)
test_ds = test_ds.batch(32).prefetch(32)
full_train_ds = full_train_ds.batch(32).prefetch(32)
from tensorflow import keras

# Numerical features
age = keras.Input(shape=(1,), name='Age')
fare = keras.Input(shape=(1,), name='Fare')

# Integer categorical features
pclass = keras.Input(shape=(1,), name='Pclass', dtype='int64')
sibsp = keras.Input(shape=(1,), name='SibSp', dtype='int64')
parch = keras.Input(shape=(1,), name='Parch', dtype='int64')

# String categorical features
sex = keras.Input(shape=(1,), name='Sex', dtype='string')
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow.keras.layers.experimental.preprocessing import IntegerLookup


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset):
    # Create a Lookup layer which will turn strings into integer indices
    if feature.dtype.name == 'string':
        index = StringLookup()
    else:
        index = IntegerLookup()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible feature values and assign them a fixed integer index
    index.adapt(feature_ds)

    # Turn the values into integer indices
    encoded_feature = index(feature)

    # Create a CategoryEncoding for our integer indices
    encoder = CategoryEncoding(output_mode="binary")

    # Prepare a dataset of indices
    feature_ds = feature_ds.map(index)

    # Learn the space of possible indices
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices
    encoded_feature = encoder(encoded_feature)
    return encoded_feature

# Numerical features
encoded_age = encode_numerical_feature(age, name='Age', dataset=train_ds)
encoded_fare = encode_numerical_feature(fare, name='Fare', dataset=train_ds)

# Integer categorical features
encoded_pclass = encode_categorical_feature(pclass, name='Pclass', dataset=train_ds)
encoded_sibsp = encode_categorical_feature(sibsp, name='SibSp', dataset=train_ds)
encoded_parch = encode_categorical_feature(parch, name='Parch', dataset=train_ds)

# String categorical features
encoded_sex = encode_categorical_feature(sex, name='Sex', dataset=train_ds)
from tensorflow.keras import layers

inputs = [age, fare, pclass, sibsp, parch, sex]
features = layers.concatenate([encoded_age, encoded_fare, encoded_pclass, encoded_sibsp, encoded_parch, encoded_sex])

def make_model(hp):
    num_dense = hp.Int('num_dense', min_value=1, max_value=3, step=1)
    x = features
    for i in range(num_dense):
        units = hp.Int('units_{i}'.format(i=i), min_value=8, max_value=256, step=8)
        x = layers.Dense(units, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    learning_rate = hp.Float('learning_rate', min_value=3e-4, max_value=3e-3)
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=optimizer,
                  metrics=[keras.metrics.BinaryAccuracy(name='acc')])
    model.summary()
    return model
import kerastuner as kt

tuner = kt.tuners.RandomSearch(
    make_model,
    objective='val_acc',
    max_trials=100,
    overwrite=True)

callbacks=[keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', patience=3)]
tuner.search(train_ds, validation_data=val_ds, callbacks=callbacks, epochs=100)
best_hp = tuner.get_best_hyperparameters()[0]
model = make_model(best_hp)
history = model.fit(train_ds, validation_data=val_ds, epochs=100)
val_acc_per_epoch = history.history['val_acc']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))
model = make_model(best_hp)
model.fit(full_train_ds, epochs=best_epoch)
import numpy as np

predictions = tf.nn.sigmoid(model.predict(test_ds)).numpy()
passenger_ids = test_dataframe.pop("PassengerId")
submission = pd.DataFrame({"PassengerId": passenger_ids,
                           "Survived": np.ravel(np.round(predictions))})
submission.to_csv("submission.csv", index=False)
