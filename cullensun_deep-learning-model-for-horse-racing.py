import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection 
import matplotlib.pyplot as plt
races = pd.read_csv(r"../input/hkracing/races.csv", delimiter=",", header=0, index_col='race_id')
races_data = races[['venue', 'race_no', 'config', 'surface', 'distance', 'going', 'horse_ratings', 'race_class']]
runs = pd.read_csv(r"../input/hkracing/runs.csv", delimiter=",", header=0)
runs_data = runs[['race_id', 'result', 'won', 'horse_age', 'horse_country', 'horse_type', 'horse_rating',
                  'declared_weight', 'actual_weight', 'draw', 'win_odds', 'trainer_id', 'jockey_id']]
data = runs_data.join(races_data, on='race_id')
# drop race_id after join because it's not a feature
data = data.drop(columns=['race_id'])
print(data.head())
# remove rows with NaN
print('data shape before drop NaN rows', data.shape)
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)
print('data shape after drop NaN rows', data.shape)

# encode ordinal columns
encoder = preprocessing.OrdinalEncoder()
data['config'] = encoder.fit_transform(data['config'].values.reshape(-1, 1))
data['going'] = encoder.fit_transform(data['going'].values.reshape(-1, 1))
data['horse_ratings'] = encoder.fit_transform(data['horse_ratings'].values.reshape(-1, 1))

# encode nominal columns
horse_countries = sorted(data['horse_country'].unique())
horse_types = sorted(data['horse_type'].unique())
trainer_ids = sorted(data['trainer_id'].unique())
jockey_ids = sorted(data['jockey_id'].unique())
venues = sorted(data['venue'].unique())
onehot = preprocessing.OneHotEncoder(dtype=np.int, sparse=True)
nominal_columns = ['horse_country', 'horse_type', 'venue', 'trainer_id', 'jockey_id']
nominals = pd.DataFrame(onehot.fit_transform(data[nominal_columns]).toarray(),
                        columns=np.concatenate((horse_countries, horse_types, venues, trainer_ids, jockey_ids)))
data = data.drop(columns=nominal_columns)

data = pd.concat([data, nominals], axis=1)
print('numberic data frame', data.shape)
print(data.head())
# result and won are outputs, the rest are inputs
X = data.drop(columns=['result', 'won'])
y = data['won']
#y = pd.DataFrame(np.where(data['result'] <= 3, 1, 0), columns=['top_3'])

# standardize the inputs to similar scale
ss = preprocessing.StandardScaler()
X = pd.DataFrame(ss.fit_transform(X),columns = X.columns)
print(X.head())

# split data into train and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)
print('X_train', X_train.shape)
print('y_train', y_train.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(120, activation='relu', input_shape=(402,)),
    tf.keras.layers.Dense(80, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.Precision(name='precision')])

print(model.summary())
dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
train_dataset = dataset.shuffle(len(X_train)).batch(1000)
dataset = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))
validation_dataset = dataset.shuffle(len(X_test)).batch(1000)

print("Start training..\n")
history = model.fit(train_dataset, epochs=200, validation_data=validation_dataset)
print("Done.")
precision = history.history['precision']
val_precision = history.history['val_precision']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(precision) + 1)

plt.plot(epochs, precision, 'b', label='Training precision')
plt.plot(epochs, val_precision, 'r', label='Validation precision')
plt.title('Training and validation precision')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()