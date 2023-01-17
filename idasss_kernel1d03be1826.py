# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import KFold, train_test_split
import keras

from keras.utils import to_categorical

from keras.engine.input_layer import Input

from keras.models import Model

from keras.layers import Dense, Dropout, Flatten, Activation, Reshape

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.layers import LeakyReLU

from keras.optimizers import Adam, Adadelta

from keras import backend as K

from keras.callbacks import EarlyStopping

from keras.preprocessing.image import ImageDataGenerator
df_train = pd.read_csv('../input/digit-recognizer/train.csv')

df_test = pd.read_csv('../input/digit-recognizer/test.csv')

df_train.head()
train_data_data = df_train.drop("label", axis = 1).values / 255

train_data_data = np.expand_dims((train_data_data.reshape((train_data_data.shape[0], 28, 28))), axis = -1)



train_data_labels = to_categorical(df_train["label"].values, 10)



test_data_data = df_test.values / 255

test_data_data = np.expand_dims(test_data_data.reshape((test_data_data.shape[0], 28, 28)), axis = -1)
def build_model():

    input_layer = Input(shape = (28, 28, 1))

    

    x = BatchNormalization()(input_layer)

    x = Activation(activation = 'relu')(x)

    

    x = Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same')(x)

    x = LeakyReLU()(x)

    x = Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same')(x)

    x = LeakyReLU()(x)

    

    x = MaxPooling2D(pool_size = (2, 2))(x)

    

    x = Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same')(x)

    x = LeakyReLU()(x)

    

    x = MaxPooling2D(pool_size = (2, 2))(x)

    

    x = Conv2D(filters = 128, kernel_size = (3, 3))(x)

    x = LeakyReLU()(x)

    

    x = Flatten()(x)

    x = Dense(units = 256)(x)

    x = LeakyReLU()(x)

    x = Dense(units = 128)(x)

    x = LeakyReLU()(x)

    x = Dense(units = 10)(x)

    x = Activation(activation = 'softmax')(x)

    

    model = Model(inputs = input_layer, outputs = x)

    model.compile(loss = 'categorical_crossentropy', optimizer = Adadelta(), metrics = ['acc'])

    return model
data_generator = ImageDataGenerator(

    featurewise_center = False,

    samplewise_center = False,

    featurewise_std_normalization = False,

    samplewise_std_normalization = False,

    zca_whitening = False,

    rotation_range = 10,

    width_shift_range = 0.1,

    height_shift_range = 0.1,

    shear_range = 0.1,

    zoom_range = 0.1,

    channel_shift_range = False,

    horizontal_flip = False,

    vertical_flip = False

)
max_epochs = 300

batch_size = 64

train_finish_ratio = 0.005

ratio_to_judge_retrain = 0.9
weights_list = []
X_loop_train_base = train_data_data

Y_loop_train_base = train_data_labels



while True:

    X_loop_train, X_loop_test, Y_loop_train, Y_loop_test = train_test_split(X_loop_train_base, Y_loop_train_base, test_size = 0.7)

    

    data_generator.fit(X_loop_train)

    

    earlystopping_callback = EarlyStopping(monitor = 'val_loss', patience = 3, verbose = 1, mode = 'auto')



    model = build_model()

    history = model.fit_generator(

        data_generator.flow(X_loop_train, Y_loop_train, batch_size = batch_size),

        epochs = max_epochs,

        verbose = 1,

        steps_per_epoch = len(X_loop_train) / batch_size,

        callbacks = [earlystopping_callback],

        validation_data = (X_loop_test, Y_loop_test)

    )

    

    weights_list.append(model.get_weights())

    prediction = model.predict(X_loop_test)

    

    train_base_flag = 0

    for i in range(prediction.shape[0]):

        add_next_flag = 0

        if prediction[i].argmax(axis = -1) != Y_loop_test[i].argmax(axis = -1):

            add_next_flag = 1

        

        if prediction[i, prediction[i].argmax(axis = -1)] < ratio_to_judge_retrain:

            add_next_flag = 1

        

        if add_next_flag == 1:

            if train_base_flag == 0:

                X_loop_train_base = np.expand_dims(X_loop_test[i], axis = 0)

                Y_loop_train_base = np.expand_dims(Y_loop_test[i], axis = 0)

                if_train_base_flag = 1

            else:

                X_loop_train_base = np.concatenate([X_loop_train_base, np.expand_dims(X_loop_test[i], axis = 0)])

                Y_loop_train_base = np.concatenate([Y_loop_train_base, np.expand_dims(Y_loop_test[i], axis = 0)])

            train_base_flag = 1

                

    print(X_loop_train_base.shape[0])

    

    if X_loop_train_base.shape[0] < train_data_data.shape[0] * train_finish_ratio:

        break
submission_test_data = test_data_data

submission_fail_data_id = []



first_time_flag = 1

for weights in weights_list:

    print('predict!')

    fail_data_id = submission_fail_data_id

    

    model = build_model()

    model.set_weights(weights)

    

    raw_prediction = model.predict(submission_test_data)

    

    new_data_flag = 1

    for i in range(raw_prediction.shape[0]):

        if raw_prediction[i, raw_prediction[i].argmax(axis = -1)] < ratio_to_judge_retrain:

            if new_data_flag == 1:

                if first_time_flag == 1:

                    submission_test_data = np.expand_dims(test_data_data[i], axis = 0)

                    submission_fail_data_id = [i]

                else:

                    submission_test_data = np.expand_dims(test_data_data[i], axis = 0)

                    submission_fail_data_id = [fail_data_id[i]]

                new_data_flag = 0

            else:

                if first_time_flag == 1:

                    submission_test_data = np.concatenate([submission_test_data, np.expand_dims(test_data_data[i], axis = 0)])

                    submission_fail_data_id.append(i)

                else:

                    submission_test_data = np.concatenate([submission_test_data, np.expand_dims(test_data_data[i], axis = 0)])

                    submission_fail_data_id.append(fail_data_id[i])

        

        

    if first_time_flag == 1:

        submit_prediction = np.expand_dims(raw_prediction.argmax(axis = -1).astype(np.uint8), axis = -1)

        na_imageId = np.expand_dims(np.arange(1, test_data_data.shape[0] + 1), axis = -1)

        first_time_flag = 0

    else:

        for i in range(raw_prediction.shape[0]):

            submit_prediction[fail_data_id[i]] = np.array([raw_prediction[i].argmax(axis = -1).astype(np.uint8)])
df_submission = pd.DataFrame(np.concatenate([na_imageId, submit_prediction], axis = 1), columns = ["ImageId", "Label"])

df_submission.head()
df_submission.to_csv("submission.csv", index = False)