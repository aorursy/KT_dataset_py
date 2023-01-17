#####################################
# Libraries
#####################################
# Common libs
import pandas as pd
import numpy as np
import sys
import os
import os.path
import random
from pathlib import Path

# Image processing
import imageio
import skimage
import skimage.io
import skimage.transform
#from skimage.transform import rescale, resize, downscale_local_mean

# Charts
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns


# ML
import scipy
from sklearn.model_selection import train_test_split
from sklearn import metrics

#from sklearn.preprocessing import OneHotEncoder
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
import tensorflow

#####################################
# Settings
#####################################
plt.style.use('ggplot')
# Set random seed to make results reproducable
np.random.seed(42)
tensorflow.set_random_seed(42)
os.environ['PYTHONHASHSEED'] = '42'

# Global variables
img_folder='../input/all/All/'
img_width=100
img_height=100
img_channels=1
# Read ground truth labels
data = pd.read_csv('../input/all/All/GTruth.csv')

data['cat']=data['Ground_Truth'].astype('category').cat.rename_categories(['Healthy', 'Pneumonia'])
#data['img']=data['Id'].apply(read_img)
plt.show()
data.head()
# Show count by category in barplot
data['cat'].value_counts().plot(kind='bar',title='Pheumonia counts')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()
# Draw 4 columns per category
cols=4

# Draw samples for each category: healthy, pneumonia
for cat in data['cat'].cat.categories:
    # Plot with ncols for this category
    f, axs = plt.subplots(1,cols, figsize=(12,3))
    cat_sample = data[data['cat']==cat]['Id'].sample(cols)
    i=0
    # Draw columns
    for fid in cat_sample.values:
        file = img_folder+str(fid)+'.jpeg'
        im=imageio.imread(file)
        axs[i].imshow(im)
        # Hide grid lines came from matplotlib style
        axs[i].grid(False)
        i+=1
    plt.suptitle(cat)
    plt.tight_layout()
    plt.show()

# Split to train_data, val_data, test_data
train_data, test_data = train_test_split(data)
train_data, val_data = train_test_split(train_data, test_size=0.1)

# # ncat_bal items per category after balanced
ncat_bal = train_data['cat'].value_counts().max()
# # Pandas construction to up/downresample and get ncat_bal items in each category
train_data_bal = train_data.groupby('cat', as_index=False).apply(lambda g: g.sample(ncat_bal, replace=True)).reset_index(drop=True)

# Plot balancing results
f, axs = plt.subplots(1,2, figsize=(8,4))

# Before
ax = train_data['cat'].value_counts().plot(kind='bar', ax=axs[0])
ax.set_title('Before balancing')
ax.set_ylabel('Count')

# After
ax = train_data_bal['cat'].value_counts().plot(kind='bar', ax=axs[1])
ax.set_title('After balancing')
ax.set_ylabel('Count')

plt.tight_layout()
plt.show()

# Train data is balanced data from now
train_data = train_data_bal

# # ncat_bal items per category after balanced
ncat_bal = val_data['cat'].value_counts().max()
# # Pandas construction to up/downresample and get ncat_bal items in each category
val_data = val_data.groupby('cat', as_index=False).apply(lambda g: g.sample(ncat_bal, replace=True)).reset_index(drop=True)

def read_img(fileid):
    """
    Read and resize img, adjust channels. 
    Caution: This function is not independent, it uses global vars: img_folder, img_channels
    @param file: file id, int
    """
    img = skimage.io.imread(img_folder + str(fileid) + '.jpeg')
    img = skimage.transform.resize(img, (img_width, img_height), mode='reflect')
    # A few image are grey, duplicate them for to have 3 alpha channels.
    if(len(img.shape) < 3):
        img = np.dstack([img, img, img])
    return img
                        
# Train data
train_X = np.stack(train_data['Id'].apply(read_img))
val_X = np.stack(val_data['Id'].apply(read_img))
test_X = np.stack(test_data['Id'].apply(read_img))
 # Data augmentation - a little bit rotate, zoom and shift input images.
generator = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)

generator.fit(train_X)
train_y = pd.get_dummies(train_data['Ground_Truth'], drop_first=False)
val_y = pd.get_dummies(val_data['Ground_Truth'], drop_first=False)
test_y = pd.get_dummies(test_data['Ground_Truth'], drop_first=False)

# Build CNN model
model=Sequential()

# Convolutional layers
model.add(Conv2D(32, kernel_size=3, input_shape=(img_width, img_height,3), activation='relu', padding='same'))
model.add(MaxPool2D(2))
model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPool2D(2))
model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))

model.add(Flatten())

#Dense layers
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(train_y.columns.size, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

#Convolutional layers
# model.add(Conv2D(64, kernel_size=3, input_shape=(img_width, img_height,3), activation='relu', padding='same'))
# model.add(MaxPool2D(2))
# model.add(Dropout(0.2))
# model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
# model.add(MaxPool2D(2))
# model.add(Dropout(0.2))
# model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
# model.add(Flatten())

# # Dense layers
# model.add(Dense(500, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(200, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(train_y.columns.size, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


weights_file='best_weights.h5'

# We'll stop training if no improvement after some epochs
earlystopper = EarlyStopping(monitor='loss', patience=10, verbose=1)

# Low, avg and high score training will be saved here
# Save the best model during the traning
checkpointer = ModelCheckpoint(weights_file
    ,monitor='loss'
    ,verbose=1
    ,save_best_only=True
    ,save_weights_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)

# Train
training = model.fit_generator(generator.flow(train_X,train_y, batch_size=60)
                                ,epochs=100
                                ,validation_data=[val_X, val_y]
                                ,steps_per_epoch=100
                                ,callbacks=[earlystopper, checkpointer, reduce_lr])
# Load best weights saved
model.load_weights(weights_file)
## Trained model analysis and evaluation
f, axs = plt.subplots(1,2, figsize=(10,3))
axs=axs.flatten()
ax = axs[0]
ax.plot(training.history['loss'], label="Loss")
ax.plot(training.history['val_loss'], label="Validation loss")
ax.set_title('Train/validation loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Accuracy
ax = axs[1]
ax.plot(training.history['acc'], label="Accuracy")
ax.plot(training.history['val_acc'], label="Validation accuracy")
ax.set_title('Train/validation accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.show()
# Prepare groung truth/predicted data for evaluation
pred_onehot = model.predict(test_X)
pred = np.argmax(pred_onehot, axis=1)
gtruth = np.argmax(test_y.values, axis=1)

f, axs = plt.subplots(1,2, figsize=(12,4))
# F1 score p
m = metrics.f1_score(gtruth, pred, average=None)
#m = metrics.precision_score(test_truth, test_pred, average=None)
ax = sns.barplot(test_data['cat'].cat.categories,m, ax=axs[0])
ax.set_title("F1 score")

# sklearn.metrics.confusion_matrix result: y - true labels, x = predicted labels
cm = metrics.confusion_matrix(gtruth, pred)
# Normalize
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
conf_matrix = pd.DataFrame(cm
                           ,index = test_data['cat'].cat.categories
                           ,columns = test_data['cat'].cat.categories)
# Visualize confusion matrix
ax = sns.heatmap(conf_matrix, annot=True, ax=axs[1])
ax.set_title("Pneumonia confusion matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Ground truth")
plt.show()

# Print classification report
print(metrics.classification_report(gtruth,pred, target_names = test_data['cat'].cat.categories))
print('Accuracy: %s' % metrics.accuracy_score(gtruth, pred))
class CnnVisualizer:
    """
    Visualization. How do images look inside CNN.
    """
    # Common function for visualization of kernels
    def visualize_layer_kernels(self, img, conv_layer, title):
        """
        Displays how input sample image looks after convolution by each kernel
        :param img: Sample image array
        :param conv_layer: Layer of Conv2D type
        :param title: Text to display on the top 
        """
        # Extract kernels from given layer
        weights1 = conv_layer.get_weights()
        kernels = weights1[0]
        kernels_num = min(kernels.shape[3], 5)

        # Each row contains 3 images: kernel, input image, output image
        f, ax = plt.subplots(kernels_num, 3, figsize=(7, kernels_num*2))
        for a in ax.flatten(): a.grid(False)

        for i in range(0, kernels_num):
            # Get kernel from the layer and draw it
            kernel=kernels[:,:,:3,i]
            ax[i][0].imshow((kernel * 255).astype(np.uint8), vmin=0, vmax=255)
            ax[i][0].set_title("Kernel %d" % i, fontsize = 9)

            # Get and draw sample image from test data
            ax[i][1].imshow((img * 255).astype(np.uint8), vmin=0, vmax=255)
            ax[i][1].set_title("Before", fontsize=8)

            # Filtered image - apply convolution
            img_filt = scipy.ndimage.filters.convolve(img, kernel)
            ax[i][2].imshow((img_filt * 255).astype(np.uint8), vmin=0, vmax=255)
            ax[i][2].set_title("After", fontsize=8)

        plt.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.show()     
        
    def convolve_layer(self, img, layer):
        """
        Convolve images through all layer filters
        """
        # Extract kernels from given layer
        weights = layer.get_weights()
        kernels = weights[0]
        # Pass the image through all kernels in the layer
        res_img = img
        for kernel in kernels:
            # Filtered image - apply convolution
            kernel_img = scipy.ndimage.filters.convolve(img, kernel)  
            res_img = res_img + kernel_img
        return res_img / (len(kernels) + 1)
        
    def visualize_layer(self, img, layer, title):
        """
        Displays how input sample looks after given Conv2D layer
        @param img: img to display
        @param layer: Conv2D layer to process img
        @param title: text to display
        """
        # Apply layer's filters to the image
        res_img = self.convolve_layer(img, layer)
        
        f, axs = plt.subplots(1, 2, figsize=(7, 4))
        for ax in axs.flatten(): ax.grid(False)

        # Get and draw sample image from test data
        axs[0].imshow((img * 255).astype(np.uint8), vmin=0, vmax=255)
        axs[0].set_title("Before", fontsize=8)
        
        # After
        axs[1].imshow((res_img * 255).astype(np.uint8), vmin=0, vmax=255)
        axs[1].set_title("After", fontsize=8)
        
        plt.suptitle('Layer: %s' % title)
        plt.tight_layout()
        plt.show()  
        
        return(res_img)
    
    def visualize_layers(self, img, model):
        """
        Push input image through each layer with visualization.
        """
        # Get Conv2D layers from the model
        layers = list(filter(lambda l : isinstance(l, Conv2D), model.layers))
        
        # Filter input image layer by layer sequentually and display the output
        res_img = img
        for l in layers:
            res_img = self.visualize_layer(res_img, l, l.name)
        return(res_img)
            

# Visualizer class instance
cnn_vis = CnnVisualizer()
# Get random image with Pneumonia
pneumonia_data = test_data[test_data['cat'] == 'Pneumonia']
idx = random.randint(0,len(pneumonia_data)-1)
#idx = random.randint(0,len(test_X)-1)
# Pneumonia input image
p_in_img = test_X[idx,:,:,:]

# Visualize
p_out_img = cnn_vis.visualize_layers(p_in_img, model)
# Get random image with Pneumonia
pneumonia_data = test_data[test_data['cat'] == 'Healthy']
idx = random.randint(0,len(pneumonia_data)-1)
#idx = random.randint(0,len(test_X)-1)
# Healthy input image
h_in_img = test_X[idx,:,:,:]

# Visualize
h_out_img = cnn_vis.visualize_layers(h_in_img, model)
f, axs = plt.subplots(1,2, figsize=(12,4))

# Input distribution
sns.distplot(p_in_img.flatten(), label="Pneumonia", ax=axs[0])
sns.distplot(h_in_img.flatten(), label="Healthy", ax=axs[0])
axs[0].legend()
axs[0].set_title('Input')

# Otput distribution
sns.distplot(p_out_img.flatten(), label="Pneumonia", ax=axs[1])
sns.distplot(h_out_img.flatten(), label="Healthy", ax=axs[1])
axs[1].legend()
axs[1].set_title("Output")
plt.suptitle("Distribution")
plt.show()