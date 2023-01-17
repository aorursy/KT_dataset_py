# importing required libraries and dependcies



#for unziping files

import zipfile

#fro finding file of same extensions

import glob

#for going into file to foleder

import os

#for loading data as dataframes

import pandas as pd

#for converitng to numpy arrays 

import numpy as np

# for data visulaization and plotting

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns 



#importing deeplearing dependicies

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img

from sklearn.model_selection import train_test_split
# extraction of train and test data from zipfiles to data folder

print('Reading zip files in the dataset')

zip_files = glob.glob('/kaggle/input/dogs-vs-cats/*.zip')





print('{} files found in the input directory'.format(len(zip_files)))

for file in zip_files:

    with zipfile.ZipFile(file, 'r') as Z:

        Z.extractall('data')

    print ('{} extraction is done'.format(file.split('/')[-1]))

      

print('Extraction is completed')

    

  
# Total number of images in train and test datasets



train_dir = '/kaggle/working/data/train/'

test_dir  = '/kaggle/working/data/test1/'



print('Train directory contains {} images'.format(len(os.listdir(train_dir))))

print('Test directory contains {} images'.format(len(os.listdir(test_dir))))            
# feature and label generation funtions





def category(path): 

    label = []

    for file in os.listdir(path):

        lab = file.split('.')[0]

        label.append(lab)

    return label







def filename(path):

    fname = []

    for file in os.listdir(path):

        fname.append(file)

    return fname

# using function to extract image names and category they belong

x_train = filename(train_dir) 

y_train = category(train_dir)

x_test = filename(test_dir)





#creation of total dataframe and submission dataframe

print('Image data is loading into dataframes...')

total_df = pd.DataFrame({ 'filename': x_train, 'category': y_train})

sub_df = pd.DataFrame({'filename': x_test})



print('Dataframe are created.')



print('All images from train set are loaded into total_df dataframe from which train,validation, test set will be derived.')

print('All images from test1 set are loaded into sub_df (which is a submission dataframe), we will save this for our best model')



# function to extract Image ID, Image locations and category 

print('This function gives IDs of category , paths of images in dataframes and categorry')

def img_path(directory):

    paths = []

    cate = []

    ID_no = []

    for file in os.listdir(directory):

        path = os.path.join(directory, file)

        paths.append(path)

        cate.append(file.split('.')[0])

        ID_no.append(file.split('.')[1])

    return ID_no, paths, cate

# Implementing above function over train data set

ID_no, img_paths, train_images = img_path(train_dir)



print('For data visulaization a data frame is created with output from img_path function')

#creating new dataframe for data visulaization

visual_df = pd.DataFrame({'ID_no':ID_no,'Category':train_images, 'img_paths': img_paths})



#sneak peak on visual_df

visual_df.head(10)
# Function for showing the images in the given data frame 

# num_row  --- no.of rows in a grid of images

# num_col  --- no.of columns in a grid of images

# what  --- cat, dog, or any



print('showImages is a gallery style images showcase fuction to view images in the dataset')



print('num_row is number of rows required for the gallery')

print('num_col is number of columns required for the gallery')

print('what = cat, dog, or dogs and cats')

def showImages(num_row,num_col,data, what ):

    import matplotlib.pyplot as plt

    %matplotlib inline

    

    cat_df = data[data['Category'] == 'cat']

    dog_df = data[data['Category'] == 'dog']



    if what == 'dog':

        X = dog_df['img_paths']

        Y = dog_df['ID_no']

    elif what == 'cat':

        X = cat_df['img_paths']

        Y = cat_df['ID_no']

    else:

        X = data['img_paths']

        Y = data['ID_no']





    from sklearn.utils import shuffle

    (X_rand, Y_rand) = shuffle(X, Y)

    

    fig, axes = plt.subplots(num_row,num_col,figsize = (12,12))

    fig.suptitle(' ',fontsize=10)



    axes = axes.ravel()

    for i in range(0, num_row*num_col):

        x = load_img(X_rand.iloc[i],target_size= (150, 150))

        axes[i].imshow(x)

        axes[i].set_title("{}".format(Y_rand.iloc[i]))

        axes[i].axis('off')

        plt.subplots_adjust(wspace =0)

    fig.tight_layout()

    print("{} samples of {}s from dataset with their respective ID number".format((num_row * num_col),what))

    

    return
showImages( 10,10,visual_df, 'dog')
showImages(10, 10, visual_df, 'cat')
showImages(10,10,visual_df, 'Dogs and Cat')
# Data split into train data and validation data

train_valid_df, test_df = train_test_split(total_df, test_size = 0.04)

train_df, valid_df = train_test_split(train_valid_df, test_size = 0.2)





train_images = train_df.shape[0]

valid_images = valid_df.shape[0]

holdon_images = test_df.shape[0]

test_images = sub_df.shape[0]



print('Total number of images in training dataset is {}'.format(train_images))

print('Total number of images in validating dataset is {}'.format(valid_images))

print('Total number of images in holdon dataset is {}'.format(holdon_images))

print('Total number of images in testing dataset or submission dataset is {}'.format(test_images))

# Data distribution count plots



data  = [train_df['category'] ,  valid_df['category'], test_df['category']]



fig, axis = plt.subplots(1,3, figsize = (25,6))

axis = axis.ravel()

sns.countplot(train_df['category'], ax = axis[0])

axis[0].set_title('Distribution of training data')

axis[0].set_xlabel('Classes')

sns.countplot(valid_df['category'], ax = axis[1])

axis[1].set_title('Distribution of validating data')

axis[1].set_xlabel('Classes')

sns.countplot(test_df['category'], ax = axis[2])

axis[2].set_title('Distribution of holdon test data')

axis[2].set_xlabel('Classes')

plt.show
img_size = 224

batch_size = 128

# dataframeiterators without data agumnetation



train_map = ImageDataGenerator()

valid_map = ImageDataGenerator()

test_map =  ImageDataGenerator()



        

#Creatinga a dataframe iterators for fitting

vani_train_data = train_map.flow_from_dataframe(

            train_df,train_dir,

            x_col = 'filename',

            y_col = 'category',

            target_size = (img_size, img_size),

            batch_size = batch_size,

            class_mode = 'categorical')



vani_valid_data = valid_map.flow_from_dataframe(

             valid_df, train_dir,

             x_col = 'filename',

             y_col = 'category',

             target_size = (img_size, img_size),

             batch_size = batch_size,

             class_mode = 'categorical')





vani_test_data = test_map.flow_from_dataframe(

             test_df, train_dir,

             x_col = 'filename',

             y_col = None,

             target_size = (img_size, img_size),

             batch_size = batch_size,

             class_mode = None,

             shuffle = False)







from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

    

    

#Building model computational graph

vani_model = Sequential()

vani_model.add(Conv2D(16, (3,3), activation = 'relu', padding = 'same', input_shape = (224,224,3)))

vani_model.add(Conv2D(16, (3,3), activation = 'relu', padding = 'same'))

vani_model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))

vani_model.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same', input_shape = (224,224,3)))

vani_model.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same'))

vani_model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))

vani_model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))

vani_model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))

vani_model.add(MaxPooling2D(pool_size = (2,2),strides=(2,2)))

vani_model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))

vani_model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))

vani_model.add(MaxPooling2D(pool_size = (2,2),strides=(2,2)))

vani_model.add(Dropout(0.3))

vani_model.add(Conv2D(256, (3,3), activation = 'relu', padding = 'same'))

vani_model.add(Conv2D(256, (3,3), activation = 'relu', padding = 'same'))

vani_model.add(MaxPooling2D(pool_size = (2,2),strides=(2,2)))

vani_model.add(Dropout(0.3))

vani_model.add(Flatten())

vani_model.add(Dense(512, activation = 'relu'))

vani_model.add(Dropout(0.5))

vani_model.add(Dense(2, activation = 'softmax'))



vani_model.summary()





from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model



plot_model(vani_model, to_file='vani_model.png')

SVG(model_to_dot(vani_model).create(prog='dot', format='svg'))
#compiling model with loss, opt, metrics

loss = 'categorical_crossentropy'

opt = tf.keras.optimizers.Adam(learning_rate= 0.0001,beta_1=0.9, beta_2=0.999,epsilon=1e-07)

metrics = ['accuracy']



vani_model.compile(loss = loss, optimizer = opt, metrics = metrics)



# fitting the model for training dataset

vani_history = vani_model.fit(vani_train_data, epochs = 15,

                          validation_data = vani_valid_data,

                          validation_steps= valid_images//batch_size,

                          steps_per_epoch= train_images//batch_size)

hist1 = vani_history.history

fig = plt.figure(figsize = (15,5))



Epochs =  range(len(hist1['loss']))



fig.add_subplot(1, 2 ,1)

sns.lineplot(x = Epochs, y = hist1['val_loss'], label = 'Validation Loss')

sns.lineplot(x = Epochs, y = hist1['loss'], label = 'Training_loss')



fig.add_subplot(1,2,2)

sns.lineplot(x = Epochs, y = hist1['val_accuracy'], label = 'Validation accuracy')

sns.lineplot(x = Epochs, y = hist1['accuracy'], label = 'Training_accuracy')



import warnings

warnings.filterwarnings('ignore')



vani_pred = vani_model.predict_generator(vani_test_data)

test_df['vani_pred'] = np.argmax(vani_pred, axis = -1)

labels = dict((v,k) for k,v in vani_train_data.class_indices.items())



test_df['vani_pred'] = test_df['vani_pred'].map(labels)

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



def make_confusion_matrix(cf,

                          group_names=None,

                          categories='auto',

                          count=True,

                          percent=True,

                          cbar=True,

                          xyticks=True,

                          xyplotlabels=True,

                          sum_stats=True,

                          figsize=None,

                          cmap='Blues',

                          title=None):





    # CODE TO GENERATE TEXT INSIDE EACH SQUARE

    blanks = ['' for i in range(cf.size)]



    if group_names and len(group_names)==cf.size:

        group_labels = ["{}\n".format(value) for value in group_names]

    else:

        group_labels = blanks



    if count:

        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]

    else:

        group_counts = blanks



    if percent:

        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]

    else:

        group_percentages = blanks



    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]

    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])





    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS

    if sum_stats:

        #Accuracy is sum of diagonal divided by total observations

        accuracy  = np.trace(cf) / float(np.sum(cf))



        #if it is a binary confusion matrix, show some more stats

        if len(cf)==2:

            #Metrics for Binary Confusion Matrices

            precision = cf[1,1] / sum(cf[:,1])

            recall    = cf[1,1] / sum(cf[1,:])

            f1_score  = 2*precision*recall / (precision + recall)

            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(

                accuracy,precision,recall,f1_score)

        else:

            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)

    else:

        stats_text = ""





    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS

    if figsize==None:

        #Get default figure size if not set

        figsize = plt.rcParams.get('figure.figsize')



    if xyticks==False:

        #Do not show categories if xyticks is False

        categories=False





    # MAKE THE HEATMAP VISUALIZATION

    plt.figure(figsize=figsize)

    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)



    if xyplotlabels:

        plt.ylabel('True label')

        plt.xlabel('Predicted label' + stats_text)

    else:

        plt.xlabel(stats_text)

    

    if title:

        plt.title(title)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



vani_cf_matrix = confusion_matrix(test_df['category'],test_df['vani_pred'])





labels = [ 'True Neg','False Pos','False Neg','True Pos']

categories = ['Cat', 'Dog']

make_confusion_matrix(vani_cf_matrix, 

                      group_names=labels,

                      categories=categories, 

                      title = 'Vanila CNN comfusion matrix')



vani_matrix = classification_report(test_df['category'],test_df['vani_pred'])

print('Classification report : \n',vani_matrix)
def data_argumentation_show(n, grid_size):

    sample_aug_map = ImageDataGenerator(

            zoom_range = 0.1,

            rotation_range = 25,

            horizontal_flip = True,

            height_shift_range =0.2,

            width_shift_range = 0.2,

            fill_mode='nearest',

            rescale = 1/255)

    sample_data = sample_aug_map.flow_from_dataframe(

            (train_df.sample(n)),

            train_dir,

            x_col = 'filename',

            y_col = 'category',

            target_size = (img_size, img_size),

            class_mode = 'categorical')

  

  #subplot grid 

    plt.figure(figsize = (15,15))

    for i in range(0,grid_size*grid_size):

        plt.subplot(grid_size,grid_size, i+1)

        for x,y in sample_data:

            img = x[0]

            plt.imshow(img)

            break

            plt.tight_layout()

            plt.show()



    return 



# To visulalize the effect of data argumentation 

#select number of samples to argument----> n = 

# total number of argumentation is grid_Size**2



data_argumentation_show(1, 5)
data_argumentation_show(2, 5)
print('setting a decay learning rate for learning rate schedule')

epoch = 50

learning_rate = 3e-5 

lr_start = 0.00000001

lr_min = 0.000001

lr_max = 3e-5 

lr_rampup_epochs = 1

lr_sustain_epochs = 1

lr_exp_decay = .8



def lrfn(epoch):

    if epoch < lr_rampup_epochs:

        lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start

    elif epoch < lr_rampup_epochs + lr_sustain_epochs:

        lr = lr_max

    else:

        lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min

    return lr

    



epochs = 20

epochs_range = [i for i in range(50 if epochs<50 else epochs)]

learn_rate = [lrfn(x) for x in epochs_range]





print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(learn_rate[0], max(learn_rate), learn_rate[-1]))





fig = plt.figure()

plt.plot(epochs_range, learn_rate, 'sr-') 

plt.xlabel('Range of epochs')

plt.ylabel('Learning rate in 10^-5')

plt.title('Learning Rate Schedule')





print('This schedule ensure that learing rate to be maximum strating preiod and decrease exponentially')
# using standard data augumentation



from keras.applications.resnet50 import preprocess_input



train_aug_map = ImageDataGenerator(

                    rotation_range=10,

                    zoom_range=0.1,

                    horizontal_flip=True,

                    fill_mode='nearest',

                    width_shift_range=0.1,

                    height_shift_range=0.1,

                    preprocessing_function = preprocess_input)

res_train_data = train_aug_map.flow_from_dataframe(

            train_df, train_dir,

            x_col = 'filename',

            y_col = 'category',

            target_size = (img_size, img_size),

            batch_size = batch_size,

            class_mode = 'categorical')



#one should validate the generality of model on the actcual target images

#so not supposed agumentation

valid_aug_map = ImageDataGenerator(preprocessing_function = preprocess_input)



res_valid_data = valid_aug_map.flow_from_dataframe(

             valid_df, train_dir,

             x_col = 'filename',

             y_col = 'category',

             target_size = (img_size, img_size),

             batch_size = batch_size,

             class_mode = 'categorical')





#test data rescaling images



test_aug_map = ImageDataGenerator(preprocessing_function = preprocess_input)



res_test_data = test_aug_map.flow_from_dataframe(

             test_df, train_dir,

             x_col = 'filename',

             y_col = None,

             class_mode = None,

             target_size = (img_size, img_size),

             shuffle = False)

           

from keras.applications import resnet

from keras.applications.resnet50 import preprocess_input

from keras.layers import  *

from keras.models import Model, Sequential

from keras import optimizers

from keras import regularizers



from keras import backend as K

K.clear_session()



#loading resent 

resNet = resnet.ResNet50(weights = 'imagenet',

                        include_top = False,

                        input_shape = (224,224, 3))



resNet.trainable = False # Freeze layers

resNet_model = Sequential([

        resNet,

        Flatten(),

        Dense(1024, activation = 'relu'),

        Dropout(0.4),

        Dense(2, activation = 'softmax')])

     



optimizer = optimizers.Adam(1e-5)



resNet_model.summary()
plot_model(resNet_model, to_file='resNet_model.png')

SVG(model_to_dot(resNet_model).create(prog='dot', format='svg'))
print('Setting early stopping factor and learning rate schedule')



from keras.callbacks import EarlyStopping, LearningRateScheduler



earlystop = EarlyStopping(patience= 5)

    

lr_callback = LearningRateScheduler(lrfn, verbose = True)



callbacks = [earlystop, lr_callback]
resNet_model.compile(optimizer = optimizer,

             loss = 'categorical_crossentropy',

             metrics = ['accuracy'])





resnet_history = resNet_model.fit_generator(res_train_data, epochs = 15,

                          validation_data = res_valid_data,

                          validation_steps= valid_images//batch_size,

                          steps_per_epoch= train_images//batch_size,

                          callbacks = callbacks)

hist2 = resnet_history.history

fig = plt.figure(figsize = (15,5))



Epochs =  range(len(hist2['loss']))



fig.add_subplot(1, 2 ,1)

sns.lineplot(x = Epochs, y = hist2['val_loss'], label = 'Validation Loss')

sns.lineplot(x = Epochs, y = hist2['loss'], label = 'Training_loss')



fig.add_subplot(1,2,2)

sns.lineplot(x = Epochs, y = hist2['val_accuracy'], label = 'Validation accuracy')

sns.lineplot(x = Epochs, y = hist2['accuracy'], label = 'Training_accuracy')

res_pred = resNet_model.predict_generator(res_test_data)

test_df['res_pred'] = np.argmax(res_pred, axis = -1)

labels = dict((v,k) for k,v in res_train_data.class_indices.items())



test_df['res_pred'] = test_df['res_pred'].map(labels)

test_df.head(50)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



res_cf_matrix = confusion_matrix(test_df['category'],test_df['res_pred'])





labels = [ 'True Neg','False Pos','False Neg','True Pos']

categories = ['Cat', 'Dog']

make_confusion_matrix(res_cf_matrix, 

                      group_names=labels,

                      categories=categories, 

                      title = 'ResNet comfusion matrix')



res_matrix = classification_report(test_df['category'],test_df['res_pred'])

print('Classification report : \n',res_matrix)
# generating an dataframe iterator for test dataset



vani_sub_aug_map = ImageDataGenerator()

res_sub_aug_map = ImageDataGenerator(preprocessing_function = preprocess_input)



vani_sub_data = vani_sub_aug_map.flow_from_dataframe(

             sub_df, test_dir,

             x_col = 'filename',

             y_col = None,

             class_mode = None,

             target_size = (img_size, img_size),

             shuffle = False)





res_sub_data = res_sub_aug_map.flow_from_dataframe(

             sub_df, test_dir,

             x_col = 'filename',

             y_col = None,

             class_mode = None,

             target_size = (img_size, img_size),

             shuffle = False)
vani_pred_sub = vani_model.predict_generator(vani_sub_data)

sub_df['vani_pred_sub'] = np.argmax(vani_pred_sub, axis = -1)

labels = dict((v,k) for k,v in res_train_data.class_indices.items())

sub_df['vani_pred_sub'] = sub_df['vani_pred_sub'].map(labels)





res_pred_sub = resNet_model.predict_generator(res_sub_data)

sub_df['res_pred_sub'] = np.argmax(res_pred_sub, axis = -1)

labels = dict((v,k) for k,v in res_train_data.class_indices.items())

sub_df['res_pred_sub'] = sub_df['res_pred_sub'].map(labels)



sub_df.head()
pred_sample = sub_df.sample(18)

pred_sample.reset_index(drop = True, inplace = True)

plt.figure(figsize=(12,24))

for index, row in pred_sample.iterrows():

    filename = row['filename']

    vani_pred = row['vani_pred_sub']

    res_pred = row['res_pred_sub']

    img = load_img( test_dir + filename, target_size= (img_size, img_size))

    plt.subplot(6,3, index+1)

    plt.imshow(img)

    plt.text(130, 175, 'vanila_pred: {}'.format(vani_pred), color='lightgreen',fontsize= 11, bbox=dict(facecolor='black', alpha=0.9))

    plt.text(130, 200, 'resNet_pred: {}'.format(res_pred), color='red',fontsize= 11, bbox=dict(facecolor='black', alpha=0.9))

    plt.title(filename.split('.')[0])

plt.tight_layout()

#plt.subplots_adjust( wspace=0, hspace= 1)

plt.show()

   
sub_df
sub_df['res_pred_sub'].replace({'cat': 0, 'dog': 1},  inplace = True)

sub_df.rename(columns = {'res_pred_sub': 'label'}, inplace = True)

sub_df['filename'] = sub_df['filename'].str.split('.').str[0]

sub_df.rename(columns = {'filename': 'id'}, inplace = True)

sub_df.drop(['vani_pred_sub'], axis=1, inplace=True)

sub_df.to_csv('submission.csv', index=False)
sub_df