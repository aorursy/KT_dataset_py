## Read in training data (from https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction)



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook



%time train = pd.read_csv("../input/train.csv", dtype={"acoustic_data": np.int16, "time_to_failure": np.float32})
# Split the whole training segments into five cross-folds.

# There are 17 pre-failure periods in the training data, so samples for five cross folds were randomly selected 

# from pre-failure periods with index 0 - 3, 4 - 6, 7 - 9, 10 - 12, and 13 - 16.



sample_num = 150

input_len = 150000

np.random.seed(7898)



# From https://www.kaggle.com/friedchips/how-to-reduce-the-training-data-to-400mb

ttf = train["time_to_failure"].values

index_start = np.nonzero(np.diff(ttf) > 0)[0] + 1

index_start = np.insert(index_start, 0, 0) # insert 1st period start manually

chunk_length = np.diff(np.append(index_start, train.shape[0]))



X_series = []

y_series = []

# Split 17 pre-failure periods into five cross-folds, where the last four folds starts at index of 4, 7, 10 and 13th pre-failure period

cv_assign = [4, 7, 10, 13]

X, y = None, None

for i in range(len(index_start)):

    if i in cv_assign:

        X_series.append(X)

        y_series.append(y)

        X, y = None, None

    index_set = np.random.randint(low=index_start[i], high=index_start[i] + chunk_length[i] - input_len, size=sample_num)

    ac_data = np.zeros((sample_num,input_len, 1), dtype=np.int16)

    ac_label = np.zeros((sample_num,), dtype=np.float32)

    for j in range(sample_num):

        ac_data[j, :, 0] = train["acoustic_data"].values[index_set[j]:index_set[j] + input_len]

        ac_label[j] = train["time_to_failure"].values[index_set[j] + input_len]

    if X is None: X, y = ac_data, ac_label

    else:

        X, y = np.concatenate((X, ac_data), axis=0), np.concatenate((y, ac_label), axis=0)

X_series.append(X)

y_series.append(y)

del train # Just to save space



# From https://www.kaggle.com/tandonarpit6/lanl-earthquake-prediction-fast-ai

submission = pd.read_csv("../input/sample_submission.csv", index_col="seg_id")

X_test = np.zeros((2624, 150000, 1), dtype=np.int32)

i = 0

for seg_id in tqdm_notebook(submission.index):

    seg = pd.read_csv("../input/test/{}.csv".format(seg_id))

    X_test[i, :, :] = seg.values

    i += 1
#Generate CNN model with a squeeze-and-excitation mechanism

from keras.models import Sequential, Model

from keras.layers import Input, Conv1D, MaxPooling1D, Dense, Flatten, Multiply, Dropout, Lambda, GlobalAveragePooling1D

from keras.optimizers import Adam



def squeeze_block(x, filter_num):

    squeeze = GlobalAveragePooling1D()(x)

    squeeze = Dense(int(filter_num / 2), activation="relu", kernel_initializer="he_normal")(squeeze)

    squeeze = Dense(filter_num, activation="sigmoid", kernel_initializer="he_normal", name="squeeze_coef")(squeeze)

    c = Multiply()([x, squeeze])

    return c



def model_gen(InputDim):

    print("Building model ...")

    inputs = Input((InputDim, 1))

    c = Lambda(lambda x: x / 100.0)(inputs)

    c = Conv1D(filters=16, kernel_size=(10,), strides=1, padding="valid", 

               kernel_initializer="he_normal", activation="relu", name="conv_before_squeeze")(c)

    c = squeeze_block(c, 16)

    c = MaxPooling1D(pool_size=10, strides=10, padding="valid")(c)

    c = Conv1D(filters=16, kernel_size=(10,), strides=5, padding="valid", 

               kernel_initializer="he_normal", activation='relu', name="conv_after_squeeze")(c)

    f = Flatten()(c)

    d = Dense(64, activation="relu", kernel_initializer="he_normal")(f)

    d = Dropout(rate=0.5)(d)

    outputs = Dense(1, activation="elu", kernel_initializer="he_normal")(d)

    model = Model(inputs=[inputs], outputs=[outputs])

    optim = Adam(lr=0.005)

    model.compile(loss="mean_absolute_error", optimizer=optim)

    return model

model = model_gen(150000)

model.summary()
# Model training with 5-cv



import numpy as np

from keras.callbacks import LearningRateScheduler



batch_size = 50

epochs_number = 30



def step_decay(epoch):

    initial_lrate = 0.005

    drop = 0.8

    epochs_drop = 5.0

    lrate = initial_lrate * np.power(drop, np.floor((1 + epoch) / epochs_drop))

    return lrate

lrate = LearningRateScheduler(step_decay)



model_loss = np.zeros((5,), dtype=np.float32)

y_pred_series = []

y_test = np.zeros((2624, 1), dtype=np.float32)



for i in range(len(X_series)):

    model = model_gen(input_len)

    X_fit, y_fit = None, None

    for j in range(len(X_series)):

        if j != i:

            if X_fit is None: X_fit, y_fit = X_series[j], y_series[j]

            else:

                X_fit, y_fit = np.concatenate((X_fit, X_series[j]), axis=0), np.concatenate((y_fit, y_series[j]), axis=0)

    X_val, y_val = X_series[i], y_series[i]

    model.fit(X_fit, y_fit, batch_size=batch_size, epochs=epochs_number, 

              validation_data=(X_val, y_val), shuffle=True, callbacks=[lrate])

    model_loss[i] = model.evaluate(X_val, y_val)

    y_pred_series.append(model.predict(X_val))

    y_test += model.predict(X_test)

model.save_weights(filepath="model_param.hdf5")

    

model_loss_mean = np.mean(model_loss)

model_loss_std = np.std(model_loss)

print("CV Score for the model is {:.3f} +/- {:.3f}".format(model_loss_mean, model_loss_std))

submission.time_to_failure = y_test / 5

submission.to_csv("submission.csv", index=True)
print("CV Score for the model is {:.3f} +/- {:.3f}".format(model_loss_mean, model_loss_std))

fig, ax = plt.subplots(5, 1, figsize=(15, 10))

for i in range(len(y_pred_series)):

    index = np.argsort(y_series[i])

    ax[i].plot(np.arange(y_series[i].shape[0]), y_series[i][index], "b-")

    ax[i].plot(np.arange(y_pred_series[i].shape[0]), y_pred_series[i][index], "r.")

    ax[i].set_ylabel("Time in CV {}".format(i + 1))

    if i == 0:

        ax[i].set_title("Truth (Blue) versus Prediction (Red)")

    ax[-1].set_xlabel("Samples ranked with increasing failure time")
from keras import backend as K



I = y_val.argsort()[-1] # Pick some sample with the largest failure time

x = X_val[I, :, :]

y = y_val[I]

x = np.expand_dims(x, axis=0)

y_pred = model.predict(x)[0][0]



get_pre_squeeze_conv_layer_output = K.function([model.layers[0].input],

                                  [model.get_layer("conv_before_squeeze").output])

get_post_squeeze_conv_layer_output = K.function([model.layers[0].input],

                                      [model.get_layer("conv_before_squeeze").output])

get_squeeze_coefficients = K.function([model.layers[0].input],

                                      [model.get_layer("squeeze_coef").output])



conv_output = np.array(get_pre_squeeze_conv_layer_output([x])[0])

conv2_output = np.array(get_post_squeeze_conv_layer_output([x])[0])

squeeze_coefs = np.array(get_squeeze_coefficients([x])[0])



r, c = 5, 8

fig, ax = plt.subplots(r, c, figsize=(30, 10))

for i in range(r):

    for j in range(c):

        ax[i][j].set_xticks([])

        if i == 0:

            if j == 0:

                ax[i][j].plot(np.arange(x.shape[1]), x[0, :, 0], "b-")

                ax[i][j].set_title("True/Pred = {:.2f}/{:.2f}".format(y, y_pred))

            else:

                ax[i][j].set_visible(False)

        else:

            if j <= 3:

                k = c // 2 * (i - 1) + j

                ax[i][j].plot(np.arange(conv_output.shape[1]), conv_output[0, :, k], "b-")

                ax[i][j].set_title("Squeeze Coef: {:.2f}".format(squeeze_coefs[0, k]))

            else:

                k = c // 2 * (i - 1) + j - 4

                ax[i][j].plot(np.arange(conv2_output.shape[1]), conv2_output[0, :, k], "b-")

ax[2][0].set_ylabel("First Conv Outputs Before Squeezing")

ax[2][4].set_ylabel("Second Conv Outputs After Squeezing")
# From https://www.kaggle.com/rtatman/download-a-csv-file-from-a-kernel

from IPython.display import HTML

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe

create_download_link(submission)