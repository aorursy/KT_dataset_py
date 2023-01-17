import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
#plot / img libs
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# load data set
X = np.load('../input/Sign-language-digits-dataset/X.npy')
Y = np.load('../input/Sign-language-digits-dataset/Y.npy')
X.shape, Y.shape
plt.imshow(X[0,:,:], cmap='gray')
Y[0,:]
def plot_digits_colidx(X, Y):
    plt.figure(figsize=(10,10))
    plt.plot([5, 2, 11])
    for i in col_idx:
        ax = plt.subplot(5, 2, i+1)
        ax.set_title("Column_idx: " + str(i))
        plt.axis('off')
        plt.imshow(X[np.argwhere(Y[:,i]==1)[0][0],:], cmap='gray')
N_classes = Y.shape[1]
col_idx = [i for i in range(N_classes)]
plot_digits_colidx(X, Y)
#dictionary that handles the column index - digit relatinship
colidx_digit = {0: 9,
                1: 0,
                2: 7,
                3: 6,
                4: 1,
                5: 8,
                6: 4,
                7: 3,
                8: 2,
                9: 5}

#digit - column index relationship dictionary
digit_colidx = {v: k for k, v in colidx_digit.items()}
#create empty matrix
Y_ordered = np.zeros(Y.shape)
#fill the matrix so that the columns index also corresponds to the digit
for i in range(N_classes):
    Y_ordered[:, i] = Y[:, digit_colidx[i]]
plot_digits_colidx(X, Y_ordered)
Y.sum(axis=0)
#N images per row
N_im_lab = 5
plt.figure(figsize=(11,11))
plt.plot([N_classes, N_im_lab, (N_im_lab * N_classes) + 1])

#for every label
for lab in range(N_classes):
    #show N_im_lab first samples
    for i in range(N_im_lab):
        ax = plt.subplot(N_classes, N_im_lab, 1 + (i + (lab*N_im_lab)))
        plt.axis('off')
        plt.imshow(X[np.argwhere(Y_ordered[:,lab]==1)[i][0],:], cmap='gray')
import keras
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten
def keras_lr(input_shape):
    #input layer
    X_input = Input(input_shape)
    #flatten
    X = Flatten()(X_input)
    #dense layer
    X = Dense(N_classes, activation='softmax')(X)
    model = Model(inputs = X_input, outputs = X, name='keras_lr')
    return model
lr_model = keras_lr((64, 64, 1))
lr_model.summary()
#set the optimization
lr_model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
#reshape the data, to adapt the shape to the keras expectation
X = X.reshape(X.shape[0], 64, 64, 1)

#train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y_ordered, random_state=4)
#fit the model
lr_fit_hist = lr_model.fit(x = X_train , y = y_train, validation_data = (X_test, y_test), epochs = 500, batch_size = 128, verbose=0)
#show the train-test accuracy depending on the epoch
def plot_acc_vs_epoch(fit_history):
    plt.plot(fit_history.history['acc'])
    plt.plot(fit_history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
plot_acc_vs_epoch(lr_fit_hist)
#evaluate the model performance in the validation set
evs = lr_model.evaluate(x = X_test, y = y_test)
#show the accuracy metric
print(evs[1])
#fit the baseline model
base_model = keras_lr((64, 64, 1))
base_model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
basline_fit_hist = base_model.fit(x = X_train , y = y_train, validation_data = (X_test, y_test), epochs = 200, batch_size = 128, verbose=0)
#baseline model accuracy
base_model.evaluate(x = X_test, y = y_test)[1]
#compute confusion matrix
import seaborn as sn
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

def plot_conf_matrix(y_true, y_pred, set_str):
    """
    This function plots a basic confusion matrix
    """
    conf_mat = confusion_matrix(y_true, y_pred)
    df_conf = pd.DataFrame(conf_mat, index = ['Digit - ' + str(i) for i in range(N_classes)],
                           columns = ['Digit - ' + str(i) for i in range(N_classes)])

    plt.figure(figsize = (12, 12))
    sn.heatmap(df_conf, annot=True, cmap="YlGnBu")
#class estimation
base_y_test_pred = base_model.predict(X_test)
plot_conf_matrix(y_test.argmax(axis=1), base_y_test_pred.argmax(axis=1), '')
from sklearn.metrics import classification_report
print(classification_report(y_test.argmax(axis=1), base_y_test_pred.argmax(axis=1)))
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, ZeroPadding2D, Dropout
#CONV-> BatchNorm-> RELU block
def conv_bn_relu_block(X, n_channels, kernel_size=(3, 3)):
    X = Conv2D(n_channels, kernel_size)(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    return X
def keras_cnn_v1(input_shape):
    #input layer
    X_input = Input(input_shape)
    #32 filters, with 5x5 kernel size
    X = conv_bn_relu_block(X_input, 10, (5, 5))
    #Maxpooling and dropout
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.5)(X)
    #run another CONV -> BN -> RELU block
    X = ZeroPadding2D((1, 1))(X)
    X = conv_bn_relu_block(X, 20)
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.5)(X)
    #flatten
    X = Flatten()(X)
    #dense layer
    X = Dense(N_classes, activation='softmax')(X)
    model = Model(inputs = X_input, outputs = X, name='keras_lr')
    return model
cnn_v1 = keras_cnn_v1((64, 64, 1))
cnn_v1.summary()
cnn_v1.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
cnn_v1_hist = cnn_v1.fit(x = X_train , y = y_train, validation_data = (X_test, y_test), epochs = 300, batch_size = 128, verbose=0)
plot_acc_vs_epoch(cnn_v1_hist)
print (cnn_v1.evaluate(x = X_test, y = y_test)[1])
def keras_cnn_v2(input_shape):
    #input layer
    X_input = Input(input_shape)
    #32 filters, with 5x5 kernel size
    X = conv_bn_relu_block(X_input, 10, (5, 5))
    #Maxpooling and dropout
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.5)(X)
    #run another CONV -> BN -> RELU block
    X = ZeroPadding2D((1, 1))(X)
    X = conv_bn_relu_block(X, 15)
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.5)(X)
    #run another CONV -> BN -> RELU block
    X = ZeroPadding2D((1, 1))(X)
    X = conv_bn_relu_block(X, 20)
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.5)(X)
    #flatten
    X = Flatten()(X)
    #dense layer
    X = Dense(N_classes, activation='softmax')(X)
    model = Model(inputs = X_input, outputs = X, name='keras_lr')
    return model
cnn_v2 = keras_cnn_v2((64, 64, 1))
cnn_v2.summary()
cnn_v2.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
cnn_v2_hist = cnn_v2.fit(x = X_train , y = y_train, validation_data = (X_test, y_test), epochs = 300, batch_size = 128, verbose=0)
plot_acc_vs_epoch(cnn_v2_hist)
print (cnn_v2.evaluate(x = X_test, y = y_test)[1])
cnn2_y_test_pred = cnn_v2.predict(X_test)
plot_conf_matrix(y_test.argmax(axis=1), cnn2_y_test_pred.argmax(axis=1), '')
print(classification_report(y_test.argmax(axis=1), cnn2_y_test_pred.argmax(axis=1)))
def visual_err_inspection(y_true, y_pred, lab_eval, N_samples=6):
    """
    This function runs a visual error inspection. It plots two rows of images,
    the first row shows true positive predictions, while the second one shows
    flase positive predictions
    """
    df_y = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    idx_y_eval_tp = df_y.loc[(df_y.y_true == lab_eval) & (df_y.y_pred == lab_eval)].index.values[:N_samples]
    idx_y_eval_fp = df_y.loc[(df_y.y_true != lab_eval) & (df_y.y_pred == lab_eval)].index.values[:N_samples]

    #capture number of false positives
    N_fp = idx_y_eval_fp.shape[0]

    N_plts = min(N_samples, N_fp)

    fig, axs = plt.subplots(2, N_plts, figsize=(15,6))
    for i in range(N_plts):
        #set plot for true positive sample
        axs[0, i].set_title("OK: " + "Digit - " + str(lab_eval))
        axs[0, i].axis('off')
        axs[0, i].imshow(X_test[idx_y_eval_tp[i], :, :, 0], cmap='gray')
        
        #set plot for false positive sample
        lab_ = df_y.iloc[idx_y_eval_fp[i]].y_true
        axs[1, i].set_title("KO: " + "Digit - " + str(lab_))
        axs[1, i].axis('off')
        axs[1, i].imshow(X_test[idx_y_eval_fp[i], :, :, 0], cmap='gray')
       

    plt.show()
visual_err_inspection(y_test.argmax(axis=1), cnn2_y_test_pred.argmax(axis=1), 1)
visual_err_inspection(y_test.argmax(axis=1), cnn2_y_test_pred.argmax(axis=1), 4)
visual_err_inspection(y_test.argmax(axis=1), cnn2_y_test_pred.argmax(axis=1), 7)