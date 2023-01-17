# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler,scale

from sklearn import metrics

from sklearn.utils import class_weight

import keras 

from keras import layers

from keras.layers import LSTM, Input, Dense

from keras import optimizers

from keras import losses

from keras.utils import to_categorical



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



%matplotlib inline

# Any results you write to the current directory are saved as output.
X_train = pd.read_csv('../input/X_train.csv')

y_train = pd.read_csv('../input/y_train.csv')
X_train.head()

# y_train.surface.unique()


# x = X_train.values #returns a numpy array

# min_max_scaler = MinMaxScaler()

# x_scaled = min_max_scaler.fit_transform(x)

# X_train = pd.DataFrame(x_scaled)
X = pd.merge(X_train, y_train,  how='left', left_on=['series_id'], right_on=['series_id'])
X = X.drop(columns=['row_id'])
# categorical_attr = ['surface']

# label_encoder = LabelEncoder()

# X[categorical_attr] = X[categorical_attr].apply(label_encoder.fit_transform)

# X[categorical_attr].count_value()

# origin_y = X['surface']

# origin_x = X.drop(['surface'])



X['surface'].value_counts().plot('bar')
y = X['surface']

X = X.drop(columns=['surface', 'group_id','series_id'])

X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train),y_train)

class_weights


encoder = LabelEncoder()

encoder.fit(y_train)

y_train = encoder.transform(y_train)

y_train = to_categorical(y_train)

y_test = encoder.transform(y_test)

y_test = to_categorical(y_test)
X_train.head()



def prepare_data(df):

    scalable_cols = ['measurement_number', 'orientation_X', 'orientation_Y', 'orientation_Z',

       'orientation_W', 'angular_velocity_X', 'angular_velocity_Y',

       'angular_velocity_Z', 'linear_acceleration_X', 'linear_acceleration_Y',

       'linear_acceleration_Z']

# print(X_train.measurement_number)

    df[scalable_cols] = scale(df[scalable_cols])



    return df



def perform_feature_engineering(actual):

    new = pd.DataFrame()

    actual['total_angular_velocity'] = (actual['angular_velocity_X'] ** 2 + actual['angular_velocity_Y'] ** 2 + actual['angular_velocity_Z'] ** 2) ** 0.5

    actual['total_linear_acceleration'] = (actual['linear_acceleration_X'] ** 2 + actual['linear_acceleration_Y'] ** 2 + actual['linear_acceleration_Z'] ** 2) ** 0.5

    

    actual['acc_vs_vel'] = actual['total_linear_acceleration'] / actual['total_angular_velocity']

    

    x, y, z, w = actual['orientation_X'].tolist(), actual['orientation_Y'].tolist(), actual['orientation_Z'].tolist(), actual['orientation_W'].tolist()

    nx, ny, nz = [], [], []

    for i in range(len(x)):

        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])

        nx.append(xx)

        ny.append(yy)

        nz.append(zz)

    

    actual['euler_x'] = nx

    actual['euler_y'] = ny

    actual['euler_z'] = nz

    

    actual['total_angle'] = (actual['euler_x'] ** 2 + actual['euler_y'] ** 2 + actual['euler_z'] ** 2) ** 5

    actual['angle_vs_acc'] = actual['total_angle'] / actual['total_linear_acceleration']

    actual['angle_vs_vel'] = actual['total_angle'] / actual['total_angular_velocity']

    

    def mean_change_of_abs_change(x):

        return np.mean(np.diff(np.abs(np.diff(x))))



    def mean_abs_change(x):

        return np.mean(np.abs(np.diff(x)))

    

    for col in actual.columns:

        if col in ['row_id', 'series_id', 'measurement_number']:

            continue

        new[col + '_mean'] = actual.groupby(['series_id'])[col].mean()

        new[col + '_min'] = actual.groupby(['series_id'])[col].min()

        new[col + '_max'] = actual.groupby(['series_id'])[col].max()

        new[col + '_std'] = actual.groupby(['series_id'])[col].std()

        new[col + '_max_to_min'] = new[col + '_max'] / new[col + '_min']

        

        # Change. 1st order.

        new[col + '_mean_abs_change'] = actual.groupby('series_id')[col].apply(mean_abs_change)

        

        # Change of Change. 2nd order.

        new[col + '_mean_change_of_abs_change'] = actual.groupby('series_id')[col].apply(mean_change_of_abs_change)

        

        new[col + '_abs_max'] = actual.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))

        new[col + '_abs_min'] = actual.groupby('series_id')[col].apply(lambda x: np.min(np.abs(x)))



    return new
# y_train = y_train.values.reshape(1, -1)

# cols_to_norm = ['orientation_X', 'orientation_Y',

#        'orientation_Z', 'orientation_W', 'angular_velocity_X',

#        'angular_velocity_Y', 'angular_velocity_Z', 'linear_acceleration_X',

#        'linear_acceleration_Y', 'linear_acceleration_Z']

# X_train[cols_to_norm] = X_train[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

X_train = prepare_data(X_train)






model = keras.models.Sequential()

model.add(layers.Dense(2048,activation='relu',input_dim=11))

model.add(layers.Dense(2048,activation='relu'))

model.add(layers.Dense(2048,activation='relu'))

model.add(layers.Dense(2048,activation='relu'))

model.add(layers.Dense(1024,activation='relu'))

model.add(layers.Dense(1024,activation='relu'))

model.add(layers.Dense(1024,activation='relu'))

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dropout(0.8))

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dense(9, activation='softmax'))

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])



model.compile(optimizer=optimizers.Adam() , loss=losses.categorical_crossentropy, metrics=['acc',])

# model.summary()
history = model.fit(X_train,y_train,batch_size=10000,epochs=80,class_weight=class_weights)

# model = keras.models.load_modelo('acc-93')
plt.plot(history.history['loss'])

# plt(history.history['loss'])

y_preds = model.predict(prepare_data(X_test))

# metrics.f1_score(y_test, y_preds, average='weighted')  

metrics.f1_score(y_test.argmax(axis=1), y_preds.argmax(axis=1), average='weighted')  
confusion_matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_preds.argmax(axis=1))
def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=True,

                          title=None,

                          cmap=plt.cm.OrRd):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = metrics.confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    # Only use the labels that appear in the data

#     classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



#     print(cm)



    fig, ax = plt.subplots()



    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label',)



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.set_size_inches(8, 8, forward=True)



    return ax



plot_confusion_matrix(y_test, y_preds, classes=['fine_concrete', 'concrete', 'soft_tiles', 'tiled', 'soft_pvc',

       'hard_tiles_large_space', 'carpet', 'hard_tiles', 'wood'],

                      title='Confusion matrix, with normalization')

plt.show()
test = pd.read_csv('../input/X_test.csv')

test.head()
test = test.drop(columns=['row_id','series_id'])
preds = model.predict(prepare_data(test))
predictions = np.argmax(preds, axis=1)
predictions
df = pd.DataFrame({'series_id': range(len(predictions)), 'surface' :encoder.inverse_transform(predictions)})
df.head(50)


submission = pd.read_csv('../input/sample_submission.csv')

submission['surface'] = df['surface']

submission.head()
submission.to_csv('submit.csv', index = False)