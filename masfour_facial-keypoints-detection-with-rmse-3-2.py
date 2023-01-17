import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split



from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Activation, BatchNormalization, Flatten, Lambda

from keras.optimizers import Adam

from keras.models import Model

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau

from keras.initializers import he_normal

from keras import backend as K



from IPython.display import clear_output
# first let's make a directory to extract the data

!mkdir /root/data
# extract the data in the created directory

!unzip ../input/training.zip -d /root/data

!unzip ../input/test.zip -d /root/data
train_df = pd.read_csv('/root/data/training.csv')
train_df.shape
# let's see what we are working with here

train_df.head(5)
def list2img(df_data):

    """

    converts img from the provided list type to a proper ndarray

    :param df_data: the image in its list type

    :return: the image in a proped ndarray shape

    """

    int_arr = np.array(df_data.split(' ')).astype(int)

    img = int_arr.reshape(96, 96)

    return img



# let's check a random image from the dataframe to check that it works as needed

rnd_idx = np.random.randint(0, train_df.shape[0])  # select a random idx

sample = train_df.Image.iloc[rnd_idx]  # select the corresponding row and get its image

sample_img = list2img(sample)  # convert it to an ndarray



plt.imshow(sample_img, cmap='gray')
train_df.Image = train_df.Image.apply(lambda x: list2img(x));  # apply the function to all image entries
# let's check that the dataframe images column is now converted to the required shape

plt.imshow(train_df.Image.iloc[rnd_idx], cmap='gray')
def plot_keypoints(df_row, plot_img=True, ax=None, color='blue'):

    """

    :param df_row: a row of the dataframe with the image entrey converted

    :param plot_img: boolean for plotting the image as well or just the keypoints

    :param ax: provided axes to plot on - the function will create a new one if not provided

    :param color: color of th plotted keypoints

    :return: the axes that was used in plotting 

    """

    if not ax:

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    if plot_img:

        ax.imshow(df_row[-1], cmap='gray')

    

    for i in range(0, (df_row.size-1), 2):  # loop on every two entries since data is stored as (x, y)

        # get the x and y coordinates for each of the 15 features

        x_coord = df_row[i]

        y_coord = df_row[i+1]

        if not np.isnan(x_coord) and not np.isnan(y_coord):  # check if th coordinates aren't missing

            circle = plt.Circle((x_coord, y_coord), 1, color=color)  # create circle

            ax.add_artist(circle)  # add the circle to the axes

    return ax



rnd_idx = np.random.randint(0, train_df.shape[0])

print(train_df.iloc[rnd_idx][:-1])

plot_keypoints(train_df.iloc[rnd_idx])
train_df.isna().sum()
# fill_vals = train_df.iloc[:, :-1].mean().to_dict()

# fill_vals
# train_df.fillna(value=fill_vals, axis=0, inplace=True)

# train_df.fillna(method='ffill', axis=0, inplace=True)

train_df.dropna(axis=0, how='any', inplace=True)  # drop the rows with missing values
train_df.shape
train_df.isna().sum()  # we no longer have missing vlues
faces = train_df.Image.values  # let's extract the images on their own outside the dataframe

keypoints = np.array(train_df.iloc[:, :-1])  # extract the keypoints

print(faces.shape, keypoints.shape)
# faces are now an array of array we'll convert it to a 2d array

faces = np.apply_along_axis(lambda x: x.tolist(), axis=0, arr=faces)  # convert to 2d array 

faces = faces.reshape(*faces.shape, 1)  # add a dimension for color channels because convolution layers require it

faces.shape
faces = faces.astype(float) / 255  # normalize the images
fit_x, test_x, fit_y, test_y = train_test_split(faces, keypoints, test_size=0.1)

train_x, valid_x, train_y, valid_y = train_test_split(fit_x, fit_y, test_size=0.2)

print(f'training set shape (x, y): {train_x.shape, train_y.shape}')

print(f'validation set shape (x, y): {valid_x.shape, valid_y.shape}')

print(f'test set shape (x, y): {test_x.shape, test_y.shape}')
# because we defined plot_keypoints to take a row of the dataframe as an input we use this function to combine the

# image with its keypoints as a row before sending it to the function

def wrap_as_row(keypoints, img):

    return np.array(list(keypoints) + [img])
# used to plot training curves (accuracy, loss) while model is training

class Plotter(Callback):

    def plot(self):  # Updates the graph

        clear_output(wait=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        

        # plot the losses

        ax1.plot(self.epochs, self.losses, label='train_loss')

        ax1.plot(self.epochs, self.val_losses, label='val_loss')

        

        # plot the accuracies

        ax2.plot(self.epochs, self.acc, label='train_acc')

        ax2.plot(self.epochs, self.val_acc, label='val_acc')

    

        ax1.set_title(f'Loss vs Epochs')

        ax1.set_xlabel("Epochs")

        ax1.set_ylabel("Loss")

        

        ax2.set_title(f'Accuracy vs Epochs')

        ax2.set_xlabel("Epoches")

        ax2.set_ylabel("Accuracy")

        

        ax1.legend()

        ax2.legend()

        plt.show()

        

        # print out the accuracies at each epoch

        print(f'Epoch #{self.epochs[-1]+1} >> train_acc={self.acc[-1]*100:.3f}%, train_loss={self.losses[-1]:.5f}')

        print(f'Epoch #{self.epochs[-1]+1} >> val_acc={self.val_acc[-1]*100:.3f}%, val_loss={self.val_losses[-1]:.5f}')

        

    def on_train_begin(self, logs={}):

        # initialize lists to store values from training

        self.losses = []

        self.val_losses = []

        self.epochs = []

        self.batch_no = []

        self.acc = []

        self.val_acc = []

    

    def on_epoch_end(self, epoch, logs={}):

        # append values from the last epoch

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))

        self.acc.append(logs.get('acc'))

        self.val_acc.append(logs.get('val_acc'))

        self.epochs.append(epoch)

        self.plot()  # update the graph

        

    def on_train_end(self, logs={}):

        self.plot()

               

plotter = Plotter()
# used to decrease the learning rate if val_acc doesn't enhance

plateau_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5,

                              patience=1, min_lr=1e-25)
# not used to stop the training but just to rollback to the best weights encountered during training

e_stop = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)
callbacks = [plotter, plateau_reduce, e_stop]
# a Convolution block with optional pooling

def conv_block(x, filters, kernel_size, strides, layer_no, add_pool=False):

    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, name=f'conv{layer_no}', 

               padding='same', kernel_initializer=he_normal(layer_no))(x)

    x = Activation('relu', name=f'activation{layer_no}')(x)

    x = BatchNormalization(name=f'bn{layer_no}')(x)

    if add_pool:

        x = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same')(x)

    return x
# a Fully connected layer with activation, batchnorm and dropout

def dense_block(x, neurons, layer_no):

    x = Dense(neurons, kernel_initializer=he_normal(layer_no), name=f'topDense{layer_no}')(x)

    x = Activation('relu', name=f'Relu{layer_no}')(x)

    x = BatchNormalization(name=f'BatchNorm{layer_no}')(x)

    x = Dropout(0.5, name=f'Dropout{layer_no}')(x)

    return x
def create_model(shape):

    input_layer = Input(shape, name='input_layer')  # input layer with given shape

    

    conv1 = conv_block(input_layer, filters=32, kernel_size=[6, 6], strides=[2, 2], layer_no=1)

    conv2 = conv_block(conv1, filters=64, kernel_size=[6, 6], strides=[3, 3], layer_no=2)

    conv3 = conv_block(conv2, filters=128, kernel_size=[6, 6], strides=[3, 3], layer_no=3)

   

    flat1 = Flatten(name='Flatten1')(conv3)

     

    output_layer = Dense(30, name='Dense1')(flat1)

    

    model = Model(inputs=[input_layer], outputs=[output_layer])



    return model
# hyperparameters

height, width, channels_num = 96, 96, 1

learning_rate = 0.01

epochs = 40

batch_size = 32
# define the proper loss function - same function used for evaluation

def rmse(y_true, y_pred):

    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
model = create_model((height, width, channels_num))

optimizer = Adam(learning_rate)



model.compile(optimizer=optimizer, loss=rmse, metrics=['acc'])

model.summary()
model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, verbose=1, 

          callbacks=callbacks, validation_data=(valid_x, valid_y), shuffle=True)
metrics = model.evaluate(test_x, test_y)  # evaluate on the labelled test set we kept aside early on
m_names = model.metrics_names

print(f'{m_names[0]} = {metrics[0]}\n{m_names[1]} = {metrics[1]}')
preds = model.predict(test_x)  # let's predict the same labelled test data
fig, axs = plt.subplots(2, 4, figsize=(20, 10))

axs = axs.ravel()  # flatten the axs array from 2d to 1d 



for ax in axs:

    rnd_idx = np.random.randint(0, test_x.shape[0])  # select a random image

    img = test_x[rnd_idx].reshape(96, 96)

    pred = preds[rnd_idx]  # get the corresponding prediction

    true_label = test_y[rnd_idx]  # get the corresponding label

    

    pred_row = wrap_as_row(pred, img)  # wrap them together as if they are a row from the dataframe

    true_row = wrap_as_row(true_label, img) 

    ax = plot_keypoints(pred_row, plot_img=True, ax=ax, color='red')  # plot the prediction as red

    plot_keypoints(true_row, plot_img=False, ax=ax, color='blue')  # plot the true keypoints as blue
test_df = pd.read_csv('/root/data/test.csv')

id_df = pd.read_csv('../input/IdLookupTable.csv')

sample_df = pd.read_csv('../input/SampleSubmission.csv')
test_df.head(5)
id_df.head(5)
sample_df.head(5) 
test_df.shape
# let's transform the test images into an ndarray just like we did with the training images

test_df.Image = test_df.Image.apply(list2img)

test_img = test_df.Image.values

test_img = np.apply_along_axis(lambda x: x.tolist(), axis=0, arr=test_img)

test_img.shape
# normalize the test images just like the training images used in training the model

test_img = test_img.astype(float) / 255
test_preds = model.predict(test_img.reshape(*test_img.shape, 1))  # our predictions
fig, axs = plt.subplots(2, 4, figsize=(20, 10))

axs = axs.ravel()



for i in range(axs.size):

    rnd_idx = np.random.randint(0, test_img.shape[0])



    img = test_img[rnd_idx].reshape(96, 96)

    pred = test_preds[rnd_idx]

    pred_row = wrap_as_row(pred, img)



    plot_keypoints(pred_row, plot_img=True, ax=axs[i], color='red')
# create a dataframe using the prediction because we'll be accessing certain features by feature/column name to

# make our submission file so this way it'll be easier

preds_df = pd.DataFrame(columns=train_df.columns[:-1], data=test_preds)
preds_df.head(5)
id_df.shape
# create an empty dataframe with the required columns

submission = pd.DataFrame(columns=sample_df.columns, data=[])

submission.head()
# iterate over the ids and get the name of the required feature name at each row

for columns, (row_id, img_id, feature, location) in id_df.iterrows():

    submission.loc[row_id-1] = [row_id, preds_df[feature].iloc[img_id-1]]
submission.RowId = submission.RowId.astype(int)

submission.head(5)
submission.shape
id_df.head(5)  # check the format before saving
submission.to_csv('submission.csv', index=False)  # save to csv file