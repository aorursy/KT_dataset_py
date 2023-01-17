import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Reshape, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

np.random.seed(3)
def draw(index, x):
    plt.imshow(x[index][:,:,0], cmap='gray')
    plt.show()
def draw_multiple(df):
    n_cols = 10
    n_rows = int(np.ceil(df.shape[0] / 10.0))
    col_i = 0
    row_i = 0
    fig, ax = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(25, 25 * n_rows / 10))
    for index, row in df.iterrows():
        x = row[1:]
        y = row[0]
        x = x.values.reshape((28,28))
        p = ax[row_i, col_i] if n_rows > 1 else ax[col_i]
        p.imshow(x, cmap='gray')
        p.set_title("i:{}, l:{}".format(index, y))
        
        col_i += 1
        if (col_i >= n_cols):
            col_i = 0
            row_i += 1
df_train = pd.read_csv("../input/train.csv")
x_unlabeled = pd.read_csv("../input/test.csv")
bad_sample_ids = [73,125,2013,4226,5288,5747,8566,14032,15219,16301,17300,24477,25946,34274,36018]
df_bad_samples = df_train.loc[bad_sample_ids, :]
df_train_cleaned = df_train.drop(bad_sample_ids, errors='ignore')
print('dropped')
draw_multiple(df_bad_samples)
def prepare_x(x):
    x_prepared = x.values / 256.0
    x_prepared = x_prepared.reshape(-1, 28, 28, 1)
    return x_prepared

def prepare_df(df):
    x = df_train.iloc[:, 1:]
    y = df_train.iloc[:, 0]
    x_prepared = prepare_x(x)
    y_prepared = to_categorical(y, num_classes=10)
    return x_prepared, y_prepared

x_test = prepare_x(x_unlabeled)
x_train_all, y_train_all = prepare_df(df_train)
x_train_all_cleaned, y_train_all_cleaned = prepare_df(df_train_cleaned)
print(x_train_all.shape)
print(x_train_all_cleaned.shape)
x_train, x_val, y_train, y_val = train_test_split(x_train_all_cleaned, y_train_all_cleaned, test_size = 0.02, random_state=3)
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
image_gen = ImageDataGenerator(
    rotation_range=15.0, 
    width_shift_range=0.15, 
    height_shift_range=0.15, 
    zoom_range=0.15, 
    fill_mode='nearest'
)
image_gen.fit(x_train)
def build_model(optimizer):
    CONV_FILTERS_1 = 80
    CONV_KERNEL_1 = (7, 7)

    CONV_FILTERS_2 = 80
    CONV_KERNEL_2 = (5, 5)
    MAX_POOL_SIZE_2 = (2, 2)

    CONV_FILTERS_3 = 160
    CONV_KERNEL_3 = (5, 5)
    MAX_POOL_SIZE_3 = (2, 2)

    CONV_FILTERS_4 = 160
    CONV_KERNEL_4 = (3, 3)

    model = Sequential()

    model.add(Conv2D(filters=CONV_FILTERS_1, kernel_size=CONV_KERNEL_1, activation='relu', input_shape=(28,28,1)))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=CONV_FILTERS_2, kernel_size=CONV_KERNEL_2, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=MAX_POOL_SIZE_2))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=CONV_FILTERS_3, kernel_size=CONV_KERNEL_3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=MAX_POOL_SIZE_3))

    model.add(Conv2D(filters=CONV_FILTERS_4, kernel_size=CONV_KERNEL_4, activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dense(80, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()
    return model
optimizer_rmsprop = 'rmsprop'
optimizer_adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model_rmsprop = build_model(optimizer_rmsprop)
model_adam = build_model(optimizer_adam)
def fit(model, x_train, y_train, x_val, y_val, batch_size, epochs, history):
    history_new = model.fit_generator(
        image_gen.flow(x_train, y_train, batch_size=batch_size), 
        steps_per_epoch=len(x_train) / batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_val, y_val)
    ).history
    history['acc'].extend(history_new['acc'])
    history['val_acc'].extend(history_new['val_acc'])
    return history

def train(model): 
    history = {'acc':[], 'val_acc':[]}
    history = fit(model, x_train, y_train, x_val, y_val, 128, 8, history)
    history = fit(model, x_train, y_train, x_val, y_val, 256, 8, history)
    history = fit(model, x_train, y_train, x_val, y_val, 512, 8, history)
    history = fit(model, x_train, y_train, x_val, y_val, 1024, 8, history)
    history = fit(model, x_train, y_train, x_val, y_val, 2048, 8, history)
    return history
# history_rmsprop = train(model_rmsprop)
history_adam = train(model_adam)
# pd.DataFrame.from_dict(history_rmsprop)[['acc','val_acc']].plot.line()
pd.DataFrame.from_dict(history_adam)[['acc','val_acc']].plot.line()
model = model_adam
y_test_one_hot = model.predict(x_test)
print(y_test_one_hot.shape)
y_test = np.argmax(y_test_one_hot, axis=1)
print(y_test.shape)
y_test
Submission = pd.DataFrame()
Submission['ImageId'] = range(1, x_test.shape[0] + 1)
Submission['Label'] = y_test
print(Submission.head())
Submission.set_index('ImageId', inplace=True)
Submission.to_csv('cnn_submission1.csv', sep=',')
print('Saved')
y_train_predicted_one_hot = model.predict(x_train_all)
y_train_predicted = np.argmax(y_train_predicted_one_hot, axis=1)
y_train_actual = np.argmax(y_train_all, axis=1)
train_digits_df = pd.DataFrame.from_dict({ "predicted": y_train_predicted, "actual": y_train_actual })
mislabeled_df = train_digits_df.loc[train_digits_df['predicted'] != train_digits_df['actual']]
print(mislabeled_df.shape)
mislabeled_pairs_table = pd.DataFrame(0, index=range(0,10), columns=range(0,10))
for index, row in mislabeled_df.iterrows():
    actual = row['actual']
    predicted = row['predicted']
    mislabeled_pairs_table[actual][predicted] += 1
mislabeled_pairs_table
draw_multiple(df_train.loc[mislabeled_df.index.values, :].head(40))
print("Samples marked as bad (excluded from training): {}", bad_sample_ids)
