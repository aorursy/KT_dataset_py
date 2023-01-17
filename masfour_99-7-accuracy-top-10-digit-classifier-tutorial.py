# import NumPy and Panas

import numpy as np 

import pandas as pd



# import Keras and its layers

import keras

from keras import Model

from keras.layers import Dense, Conv2D, pooling, Activation, BatchNormalization, Input, Flatten, Dropout, MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



# import preprocessing functions and metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score



# import matplotlib and seaborn for visualization

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import clear_output



# import time for checkpoint saving

import time
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
pd.set_option('display.max_columns', 6)

print(train_data.head(10), '\n')

print(test_data.head(10), '\n')



print(f'train data shape: {train_data.shape}')

print(f'test data shape: {test_data.shape}')
# storing the labeled dataset in numpy arrays

dataset_x = np.array(train_data.iloc[:, 1:]).reshape(42000, 28, 28, 1)

dataset_y = np.array(train_data.iloc[:, 0]).ravel()



# one-hot encoded labels instead of digits to use in categorical/multi-class classification

dataset_one_hot = keras.utils.to_categorical(dataset_y, num_classes=10)



# storing the unlabelled dataset in a numpy array for manual model testing

unlabelled_test_x = np.array(test_data.iloc[:, :]).reshape(28000, 28, 28, 1)



# make sure that the number of examples equals the number of labels after reshaping

assert(dataset_x.shape[0] == dataset_y.size)
# To get more control over visualization we'll define our figure instead of simply using plt.

fig = plt.figure(figsize=(8, 8))  # create figure

ax = fig.add_subplot(111)  # add subplot



sns.countplot(dataset_y);

ax.set_title(f'Labelled Digits Count ({dataset_y.size} total)', size=20);

ax.set_xlabel('Digits');



# writes the counts on each bar

for patch in ax.patches:

        ax.annotate('{:}'.format(patch.get_height()), (patch.get_x()+0.1, patch.get_height()-150), color='w')
dataset_x = dataset_x / 255.0

unlabelled_test_x = unlabelled_test_x / 255.0
fig, axs = plt.subplots(1, 5, figsize=(25, 5))

indexes = np.random.choice(range(dataset_x.shape[2]), size=5)  # returns random 5 indexes



fig.suptitle('Original Images of MNIST', size=32)

for idx, ax in zip(indexes, axs):

    ax.imshow(dataset_x[idx,:, :, 0], cmap='gray');

    ax.set_title(f'Label: {dataset_y[idx]}', size= 20);
train_x, test_x, train_y, test_y = train_test_split(dataset_x, dataset_one_hot, test_size=0.1)
# Augmentation Ranges

# Large values might lead to insignificant outliers or noisy examples

transform_params = {

    'featurewise_center': False,

    'featurewise_std_normalization': False,

    'samplewise_center': False,

    'samplewise_std_normalization': False,

    'rotation_range': 10, 

    'width_shift_range': 0.1, 

    'height_shift_range': 0.1,

#     'shear_range': 0.15, 

    'zoom_range': 0.1,

    'validation_split': 0.15

}

img_gen = ImageDataGenerator(**transform_params) 
# used only to visualize the augmentations

visualizaion_flow = img_gen.flow(train_x, train_y, batch_size=1, shuffle=False)



# used to feed the model augmented training data

train_gen = img_gen.flow(x=train_x, y=train_y, subset='training', batch_size=96)



# used to feed the model augmented validation data

valid_gen = img_gen.flow(x=train_x, y=train_y, subset='validation', batch_size=96)
fig, axs = plt.subplots(2, 4, figsize=(20,10))  # let's see 4 augmentation examples

fig.suptitle('Augmentation Results', size=32)



for axs_col in range(axs.shape[1]):

    idx = np.random.randint(0, train_x.shape[0])  # get a random index

    img = train_x[idx,:, :, 0]  # the original image

    aug_img, aug_label = visualizaion_flow[idx]  # the same image after augmentation

    

    axs[0, axs_col].imshow(img, cmap='gray');

    axs[0, axs_col].set_title(f'example #{axs_col} - Label: {np.argmax(train_y[idx])}', size=20)

    

    axs[1, axs_col].imshow(aug_img[0, :, :, 0], cmap='gray');

    axs[1, axs_col].set_title(f'Augmented example #{axs_col}', size=20)
def conv_block(x, filters, kernel_size, strides, layer_no, use_pool=False, padding='same'):

    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, name=f'conv{layer_no}',

               padding=padding)(x)

    

    x = BatchNormalization(name=f'bn{layer_no}')(x)

    x = Activation('relu', name=f'activation{layer_no}')(x)

    if use_pool:

        x = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], name=f'pool{layer_no}', padding='same')(x)

    return x
def conv_model(X):

    h, w, c = X.shape[1:]  # get shape of input (height, width, channels)

    X = Input(shape=(h, w, c))

    conv1 = conv_block(X, filters=8, kernel_size=[3, 3], strides=[1, 1], layer_no=1)

    conv2 = conv_block(conv1, filters=16, kernel_size=[2, 2], strides=[1, 1], layer_no=2)

    conv3 = conv_block(conv2, filters=32, kernel_size=[2, 2], strides=[1, 1], layer_no=3, use_pool=True)

    

    conv4 = conv_block(conv3, filters=64, kernel_size=[3, 3], strides=[2, 2], layer_no=4)

    conv5 = conv_block(conv4, filters=128, kernel_size=[2, 2], strides=[1, 1], layer_no=5)

    conv6 = conv_block(conv5, filters=256, kernel_size=[2, 2], strides=[1, 1], layer_no=6, use_pool=True)

    

    flat1 = Flatten(name='flatten1')(conv6)

    drop1 = Dropout(0.35, name='Dopout1')(flat1)

    

    dens1 = Dense(128, name='dense1')(drop1)

    bn7 = BatchNormalization(name='bn7')(dens1)

    drop2 = Dropout(0.35, name='Dopout2')(bn7)

    relu1 = Activation('relu', name='activation7')(drop2)

    

    dens1 = Dense(256, name='dense01')(relu1)

    bn7 = BatchNormalization(name='bn07')(dens1)

    drop2 = Dropout(0.5, name='Dopout02')(bn7)

    relu1 = Activation('relu', name='activation07')(drop2)

    

    dens2 = Dense(10, name='dense2')(relu1)

    bn8 = BatchNormalization(name='bn8')(dens2)

    output_layer = Activation('softmax', name='softmax')(bn8)

    

    model = Model(inputs=X, outputs=output_layer)

    return model
# create the model

model = conv_model(train_x)
def build_model(learning_rate=0.00065):

    optimizer = keras.optimizers.RMSprop(lr=learning_rate, decay=learning_rate/2)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# not used - add to callbacks if wanted

plateau_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.5,

                              patience=6, min_lr=0.0000001)
# build the model computational graph and print out its summary

build_model()

model.summary()
class Plotter(keras.callbacks.Callback):

    def plot(self):  # Updates the graph

        clear_output(wait=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        fig.suptitle('Training Curves', size=32)

        

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

        ax2.set_xlabel("Epochs")

        ax2.set_ylabel("Accuracy")

        

        ax1.legend()

        ax2.legend()

        plt.show()

        

        # print out the accuracies at each epoch

        print(f'Epoch #{self.epochs[-1]+1} >> train_acc={self.acc[-1]:.5f}, val_acc={self.val_acc[-1]:.5f}')

    

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
use_generator = True  # toggles between using the augmentation generator or the original images
callbacks = [plotter]
if use_generator:

    model.fit_generator(train_gen, validation_data=valid_gen, epochs=120, 

                        steps_per_epoch=train_x.shape[0]*0.85//96, 

                        validation_steps=train_x.shape[0]*0.15//96, callbacks=callbacks)

else:

    model.fit(x=train_x, y=train_y, epochs=80, batch_size=32, callbacks=callbacks,

              validation_split=0.15)
# save model

keras.models.save_model(model, f'model_{time.time()}.h5')
def calculate_performance(labels, pred, dataset):

    pred_cat = np.argmax(pred, axis=1)  # categorical predictions 0-9

    labels_cat = np.argmax(labels, axis=1)  # categorical labels 0-9

    

    # a boolean vector of element-wise comparison between prediction and label

    corrects = (pred_cat == labels_cat)

    

    # get the falses data

    falses = dataset[~corrects]  # the falses images

    falses_labels = labels_cat[~corrects]  # true labels of the falsely classified images - categorical

    falses_preds = pred[~corrects]  # the false predictions of the images - 10-dim prediction

     

    examples_num = labels.shape[0]  # total numbers of examples

    accuracy = np.count_nonzero(corrects) / examples_num



    return accuracy, [falses, falses_labels, falses_preds], [labels_cat, pred_cat]
train_pred = model.predict(train_x)  # predict the whole training dataset



# calculate the accuracy over the whole dataset and get information about falses

train_accuracy, train_falses_data, (true_labels, pred_labels) = calculate_performance(train_y, train_pred ,train_x)



print(f'Don\'t use as a metric - Original Training Dataset Accuracy: {np.round(train_accuracy*100, 3)}%')



plt.figure(figsize=(10, 10))



# Calculate the confusion matrix and visualize it

train_matrix = confusion_matrix(y_pred=pred_labels, y_true=true_labels)

sns.heatmap(data=train_matrix, annot=True, cmap='Blues', fmt=f'.0f')



plt.title('Confusion Matrix - Training Dataset', size=24)

plt.xlabel('Predictions', size=20);

plt.ylabel('Labels', size=20);

test_pred = model.predict(test_x)  # predict the whole test dataset



# calculate the accuracy over the whole dataset and get information about falses

test_accuracy, test_falses_data, (true_labels, pred_labels) = calculate_performance(test_y, test_pred, test_x)



print(f'Test Dataset Accuracy: {np.round(test_accuracy*100, 3)}%')



plt.figure(figsize=(10, 10))



# Calculate the confusion matrix and visualize it

test_matrix = confusion_matrix(y_pred=pred_labels, y_true=true_labels)

sns.heatmap(data=test_matrix, annot=True, cmap='Blues', fmt=f'.0f')



plt.title('Confusion Matrix - Test Dataset', size=24)

plt.xlabel('Predictions', size=20);

plt.ylabel('Labels', size=20);
random_idx = np.random.choice(range(test_y.shape[0]), size=10)  # get random 10 indexes



examples = test_x[random_idx]  # the images

preds = model.predict(examples)  # the predictions - 10-dim probabilities

labels = np.argmax(test_y[random_idx], axis=1)  # the labels - categorical

preds_cat = np.argmax(preds, axis=1)  # the predictions - categorical
fig, axs = plt.subplots(2, 5, figsize=(28, 12));

fig.suptitle('Model Predictions - Test Dataset', size=32)



for ax, pred, pred_prob, label, example in zip(axs.ravel(), preds_cat, preds, labels, examples):

    ax.imshow(example[:, :, 0] ,cmap='gray');

    ax.set_title(f'Label: {label}\nPrediction: {pred} with {np.round(np.max(pred_prob)*100, 3)}% Confidence.',

                size = 15)

    ax.axis('off')
random_idx = np.random.choice(range(test_y.shape[0]), size=10)  # get random 10 indexes



examples = unlabelled_test_x[random_idx]  # the images

preds = model.predict(examples)  # the predictions - 10-dim probabilities

preds_cat = np.argmax(preds, axis=1)  # the predictions - categorical
fig, axs = plt.subplots(2, 5, figsize=(30, 12));

fig.suptitle('Predictions of Unlabelled Images', size=32)



for ax, pred, pred_prob, example in zip(axs.ravel(), preds_cat, preds, examples):

    ax.imshow(example[:, :, 0] ,cmap='gray');

    ax.set_title(f'Prediction: {pred} with {np.round(np.max(pred_prob)*100, 3)}% Confidence.', size = 15)

    ax.axis('off')
# checking the falses of the test dataset

falses_data = test_falses_data # set the dataset to check

falses_examples, falses_labels, falses_preds = falses_data

falses_idx = np.argmax(falses_preds, axis=1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

fig.suptitle('False Classifications Plots', size=32)



sns.countplot(falses_labels, ax=ax1);

ax1.set_title(f'Falses ({falses_labels.size} total)',size=20);

ax1.set_xlabel('Digits', size=15);

ax1.set_ylabel('Count', size=15);



for patch in ax1.patches:

    bar_height = patch.get_height()

    ax1.annotate(f'{bar_height}', (patch.get_x()+0.25, bar_height-0.2), color='w', size=15);

    



falses_matrix = confusion_matrix(y_pred=falses_idx, y_true=falses_labels)

sns.heatmap(data=falses_matrix, annot=True, cmap='Blues', fmt=f'.0f')



plt.xlabel('Predictions', size=20);

plt.ylabel('Labels', size=20);

random_falses = np.random.choice(range(falses_labels.size), size=np.min((10, falses_labels.size)), replace=False)



examples = falses_examples[random_falses]

preds_probs = falses_preds[random_falses]

labels = falses_labels[random_falses]

preds_binary = np.argmax(preds_probs, axis=1)
fig, axs = plt.subplots(2, 5, figsize=(30, 12));

fig.suptitle('Misclassified Images', size=32)



for ax, pred, pred_prob, label, example in zip(axs.ravel(), preds_binary, preds_probs, labels, examples):

    ax.imshow(example[:, :, 0] ,cmap='gray');

    ax.set_title(f'Label: {label}\nPrediction: {pred} with {np.round(np.max(pred_prob)*100, 3)}% Confidence.',

                size = 15)

    ax.axis('off')
# cateforical predictions of the dataset

pred = model.predict(unlabelled_test_x)

pred_cat = np.argmax(pred,axis = 1)



pred_series = pd.Series(pred_cat,name='Label')  # pandas series of predictions

id_series = pd.Series(range(1, unlabelled_test_x.shape[0]+1), name='ImageId') # pandas series of IDs



submission = pd.concat((id_series, pred_series), axis=1)



print(submission.head(5))

# save to csv

submission.to_csv('submission.csv', index=False)