import warnings  

warnings.filterwarnings('ignore')



import os

import h5py

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.cm as cm

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score



from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv3D, BatchNormalization, MaxPooling3D
base_dir = '/kaggle/input/3d-mnist/'



with h5py.File(os.path.join(base_dir, 'full_dataset_vectors.h5'), 'r') as dataset:

    X_train = dataset["X_train"][:]

    X_test  = dataset["X_test"][:]

    y_train = dataset["y_train"][:]

    y_test  = dataset["y_test"][:]

    

print(f'X_train shape: {X_train.shape}, X_test shape: {X_test.shape}')
with h5py.File(os.path.join(base_dir, 'train_point_clouds.h5'), 'r') as hf:

    n_rows, n_cols = 1, 5

    fig = plt.figure(figsize=(20, 4))

    

    for i in range(n_cols):

        a = hf[str(i)]

        digit = (a["img"][:], a["points"][:], a.attrs["label"]) 

        ax = fig.add_subplot(1, n_cols, i+1, projection='3d')

    

        X = a["points"][:][:,0]

        Y = a["points"][:][:,1]

        Z = a["points"][:][:,2]



        ax.scatter(X, Y, Z)

        ax.view_init(5, 5)

        ax.axis('off')
with h5py.File(os.path.join(base_dir, 'train_point_clouds.h5'), 'r') as hf:

    n_rows, n_cols = 1, 5

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 7))

    axs = axs.flatten()

    

    for i in range(n_cols):

        a = hf[str(i)]

        digit = (a["img"][:], a["points"][:], a.attrs["label"]) 

        axs[i].imshow(digit[0])

        axs[i].set_title(f'Label: {digit[2]}')
sns.distplot(y_train)



label_counts = np.bincount(y_train)

print(f"Label count mean: {label_counts.mean()}, std: {label_counts.std()}")
X_train_shaped = np.ndarray((X_train.shape[0], 4096, 3))

X_test_shaped  = np.ndarray((X_test.shape[0], 4096, 3))



def array_to_color(array):

    scaler_map = cm.ScalarMappable(cmap="Oranges")

    array = scaler_map.to_rgba(array)[:, : -1]

    return array



for i in range(X_train_shaped.shape[0]):

    X_train_shaped[i] = array_to_color(X_train[i])

    

for i in range(X_test_shaped.shape[0]):

    X_test_shaped[i] = array_to_color(X_test[i])



X_train = np.reshape(X_train_shaped, (X_train_shaped.shape[0], 16, 16, 16, 3))

X_test  = np.reshape(X_test_shaped, (X_test_shaped.shape[0], 16, 16, 16, 3))

    

print(f'X_train shape: {X_train.shape}, X_test shape: {X_test.shape}')

y_train = to_categorical(y_train, 10)

y_test  = to_categorical(y_test, 10)



print(f'y_train shape: {y_train.shape}, y_test shape: {y_test.shape}')
num_models = 5

net_models = [0] * num_models



for i in range(num_models):

    net_models[i] = Sequential([

            Conv3D(16, (3,3,3), strides=1, padding='same', activation='relu', input_shape = (16, 16, 16, 3)),

            BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"),

            Dropout(0.1),



            Conv3D(32, (3,3,3), strides=1, activation='relu', padding='same'),

            BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform"),

            Dropout(0.2),

            MaxPooling3D(pool_size=2, strides=2, padding="same"),



            Conv3D(64, (5,5,5), strides=1, activation='relu', padding='same'),

            BatchNormalization(momentum=0.17, epsilon=1e-5, gamma_initializer="uniform"),

            MaxPooling3D(pool_size=2, strides=2, padding="same"),



            Conv3D(128, (5,5,5), strides=1, activation='relu', padding='same'),

            BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform"),

            Dropout(0.2),



            Conv3D(64, (5,5,5), strides=1, activation='relu', padding='same'),

            BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform"),

            Dropout(0.2),



            Conv3D(32, (3,3,3), strides=1, activation='relu', padding='same'),

            BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform"),

            Dropout(0.05),



            Flatten(),

            Dense(50, activation='relu'),

            Dropout(0.05),

            Dense(25, activation='relu'),

            Dropout(0.03),

            Dense(10, activation='softmax')

        ])



    adam = Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, amsgrad=True)

    net_models[i].compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    
epochs = 50

batch_size = 32
def step_decay_schedule(initial_lr, decay_factor, step_size):

    def schedule(epoch):

        return initial_lr * (decay_factor ** np.floor(epoch/step_size))

    return LearningRateScheduler(schedule, verbose=0)



lr_scheduler   = step_decay_schedule(initial_lr=0.002, decay_factor=0.5, step_size=5)

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=0, factor=0.5, min_lr=0.00001)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5,

                                 verbose=0, mode='auto', restore_best_weights=True)



histories = [0] * num_models

total_confusion_matrix = np.zeros((10, 10))



for i in range(num_models):

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

    

    histories[i] = net_models[i].fit(x=X_train, y=y_train,

                                     batch_size=batch_size, epochs=epochs, verbose=0,

                                     callbacks=[lr_scheduler, early_stopping],

                                     validation_data=(X_valid, y_valid))



    print('Trained model no. {}, val_acc: {}, val_loss: {}'.format(

        i+1,

        round(max(histories[i].history['val_acc']), 2),

        round(max(histories[i].history['val_loss']), 2),

    ))

    

    y_test_true = [np.argmax(x) for x in y_test]

    y_test_pred = [np.argmax(x) for x in net_models[i].predict(X_test, batch_size=16)]

    total_confusion_matrix += confusion_matrix(y_test_true, y_test_pred)
plt.style.use('seaborn')

fig, ax = plt.subplots(figsize=(20,10), nrows=2, ncols=1)

for i in range(num_models):

    ax[0].plot(histories[i].history['val_acc'], label='Model no. '+str(i))

    ax[0].set_title('Validation accuracy')

    ax[0].legend()



    ax[1].plot(histories[i].history['val_loss'], label='Model no. '+str(i))

    ax[1].set_title('Validation loss')

    ax[1].legend()
plt.figure(figsize=(20, 5))

for i in range(num_models):

    plt.plot(histories[i].history['lr'])

plt.ylabel('Learning rate')

plt.xlabel('Epoch')

plt.title('Learning rate over epochs')

plt.show()
plt.figure(figsize=(20, 6))

df_cm = pd.DataFrame(total_confusion_matrix.astype(int), range(0, 10), range(0, 10))

ax = sns.heatmap(df_cm, annot=True, linewidths=.5, fmt='d')

plt.show()
y_test_pred = np.zeros((X_test.shape[0], 10))

for i in range(num_models):

    y_test_pred = y_test_pred + net_models[i].predict(X_test)

    

y_test_true = [np.argmax(x) for x in y_test]

y_test_pred = np.argmax(y_test_pred, axis=1)



print("Accuracy:", accuracy_score(y_test_true, y_test_pred))