import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
# io related
#from skimage.io import imread


import os
from glob import glob
# not needed in Kaggle, but required in Jupyter
%matplotlib inline 
#train and valid csv files path
humornous_train_path = r'../input/humorspn/MURA-v1.1/train'
humornous_valid_path = r'../input/humorspn/MURA-v1.1/valid'

finger_train_path = r'../input/fingerclassification/FINGER/XR_FINGER train'
finger_valid_path = r'../input/fingerclassification/FINGER/XR_FINGER Valid'
files = [f for f in glob(os.path.join(finger_train_path, "**/*.png"), recursive=True)]
print(np.shape(files))
train_images_paths = pd.DataFrame({'image_path':files})
train_images_paths['label'] = train_images_paths['image_path'].map(lambda x:'positive' if 'positive' in x else 'negative')
train_images_paths['category']  = train_images_paths['image_path'].apply(lambda x: x.split('/')[3])#[5])  
train_images_paths['patientId']  = train_images_paths['image_path'].apply(lambda x: x.split('/')[5].replace('patient',''))
train_images_paths.head()
total_number_of_training_images = np.shape(train_images_paths)[0]
print("total number of images:",total_number_of_training_images )
print ("\n\nnumber of null values", train_images_paths.isnull().sum())
print("\n\nnumber of training images:",np.shape(train_images_paths['image_path'])[0])

#df.groupby('a').count()
#values counts table 
categories_counts = pd.DataFrame(train_images_paths['category'].value_counts())
print ('\n\ncategories:\n',categories_counts )
print('\n\nnumber of patients:',train_images_paths['patientId'].nunique())
print('\n\nnumber of labels:',train_images_paths['label'].nunique())
print ('\n\npositive casses:',len(train_images_paths[train_images_paths['label']=='positive']))
print ('\n\nnegative casses:',len(train_images_paths[train_images_paths['label']=='negative']))
files = [f for f in glob(os.path.join(finger_valid_path, "**/*.png"), recursive=True)]
valid_data_paths = pd.DataFrame({'image_path':files})
print (valid_data_paths.head(5))

valid_data_paths['label'] = valid_data_paths['image_path'].map(lambda x:'positive' if 'positive' in x else 'negative')
valid_data_paths['category']  = valid_data_paths['image_path'].apply(lambda x: x.split('/')[2])  
# valid_data_paths['dir'] =  valid_data_paths['image_path'].apply(lambda x: x.split('/')[1])
valid_data_paths['patientId']  = valid_data_paths['image_path'].apply(lambda x: x.split('/')[3].replace('patient',''))
valid_data_paths.head()
total_number_of_valid_images = np.shape(valid_data_paths)[0]
print("total number of validation samples:",total_number_of_valid_images)
print("data_shape:",np.shape(valid_data_paths))
print ("\n\nnumber of null values", valid_data_paths.isnull().sum())
print("\n\nnumber of training images:",np.shape(valid_data_paths['image_path']))

validaton_categories_counts = pd.DataFrame(valid_data_paths['category'].value_counts())
print ('\n\ncategories:\n',validaton_categories_counts)
print('\n\nnumber of patients:',valid_data_paths['patientId'].nunique())
print('\n\nnumber of labels:',valid_data_paths['label'].nunique())
print ('\n\npositive casses:',len(valid_data_paths[valid_data_paths['label']=='positive']))
print ('\n\nnegative casses:',len(valid_data_paths[valid_data_paths['label']=='negative']))


dataset_size =total_number_of_training_images+ total_number_of_valid_images
nof_training_samples =math.ceil(dataset_size*0.8)
nof_valid_samples = math.floor(dataset_size*0.2)

train_images_paths.append(valid_data_paths)
train_images_paths=train_images_paths.sample(frac=1.0)

valid_data_paths=train_images_paths.tail(nof_valid_samples)
train_images_paths=train_images_paths.head(nof_training_samples)
total_number_of_valid_images = np.shape(valid_data_paths)[0]
print("total number of validation samples:",total_number_of_valid_images)
total_number_of_training_images = np.shape(train_images_paths)[0]
print("total number of training samples:",total_number_of_training_images )
print(dataset_size)
# from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import ImageDataGenerator

idg_train_settings = dict(samplewise_center = True,
                         samplewise_std_normalization = True,
                          rotation_range = 0, 
                          width_shift_range = 0, 
                         height_shift_range = 0,
                         zoom_range = 0, 
                         horizontal_flip = False,
                         vertical_flip = False)
idg_train = ImageDataGenerator(**idg_train_settings)

idg_valid_settings = dict(samplewise_center = True,
                         samplewise_std_normalization = True,
                          rotation_range = 0, 
                          width_shift_range = 0., 
                         height_shift_range = 0.,
                         zoom_range = 0.0, 
                         horizontal_flip = False,
                         vertical_flip = False)
idg_valid = ImageDataGenerator(**idg_valid_settings)
# put in train_df_of_cat all train_images_paths if you want to train on all dataset
# we will choose category which is not too much images or too many 
# XR_ELBOW or XR_FINGER both have some moderate nomber of examples 
# for train only on XR_ELBOW category we use this mask
# category = 'XR_WRIST'
# train_mask = train_images_paths['category'] == category
# valid_mask = valid_data_paths['category']==category
train_df_of_cat = train_images_paths
valid_df_of_cat = valid_data_paths
train_df_of_cat['label'].hist()
valid_df_of_cat['label'].hist()
import math
images_path_dir = "../input/mura-v11"
out_dir = "./"

train_generator = idg_train.flow_from_dataframe(
    dataframe = train_df_of_cat,
#     directory = images_path_dir,
    x_col = 'image_path',
    y_col = 'label',
    batch_size =64,
    shuffle = True,
    class_mode = "categorical",
    target_size = (128, 224),
    save_to_dir = out_dir,
    save_format = "png",
    color_mode = 'rgb' )#'grayscale')

valid_generator = idg_valid.flow_from_dataframe(
    dataframe = valid_df_of_cat,
#     directory = images_path_dir,
    x_col = 'image_path',
    y_col = 'label',
    batch_size =64,
    shuffle = True,
    class_mode = "categorical",
    target_size = (128, 224),
    save_to_dir = out_dir,
    save_format = "png",
    color_mode = 'rgb')

STEP_SIZE_TRAIN=math.ceil(train_generator.n / train_generator.batch_size)
STEP_SIZE_VALID=math.ceil(valid_generator.n / valid_generator.batch_size)

a, b = next(train_generator)
i,l = next(valid_generator)
print("training input images patch shape  : ", a.shape)
print("training input labels patch shape  : ",b.shape)
print("training labels:",train_generator.class_indices)
print ("________________________________________")
print("validation input images patch shape: ", a.shape)
print("validation input labels patch shape: ",b.shape)
print("validation labels:",valid_generator.class_indices)
# tg_class = {v:k for k,v in train_generator.class_indices.items()}
# fig, m_axs = plt.subplots(3, 3, figsize = (128, 128))
# for c_ax, c_img, c_lab in zip(m_axs.flatten(), a, b):
#     c_ax.imshow(c_img[:,:,0], cmap = 'bone')
#     c_ax.axis('off')
#     c_ax.set_title(tg_class[np.argmax(c_lab)])
# vg_class = {v:k for k,v in valid_generator.class_indices.items()}
# fig, m_axs = plt.subplots(3, 3, figsize = (12, 12))
# for c_ax, c_img, c_lab in zip(m_axs.flatten(), i, l):
#     c_ax.imshow(c_img[:,:,0], cmap = 'bone')
#     c_ax.axis('off')
#     c_ax.set_title(tg_class[np.argmax(c_lab)])
input_shape =a.shape[1:]
classes = b.shape[1] #binary classification normal vs upnormal 
loss_func = 'categorical_crossentropy'
optimizer = 'adam'
epochs = 40

# from keras.applications import MobileNet
# base_model = MobileNet(classes=classes,  weights=None,include_top=True, input_shape=input_shape)
# base_model.summary()

from keras.applications import DenseNet121
from keras import Sequential,Model
from keras.layers import Input,Dense,Flatten
from keras import layers
base_model = DenseNet121(classes=classes,  weights='imagenet',include_top=False, input_shape=input_shape ,input_tensor=Input(shape=(128, 224, 3)))

# remove the last fully connected layer
# base_model.layers.pop()
# freeze all the weights of the model except the last 4 layers
# for layer in base_model.layers[:-100]:
#     layer.trainable = False
#adding output layer  
model =Flatten()(base_model.output) 
model=Dense(2,activation='softmax')(model) 

model = Model(inputs=base_model.input, outputs=model)
model.summary()

# from keras.applications import MobileNet
# from keras import Sequential,Model
# from keras.layers import Input,Dense,Flatten
# #dropout=0.213
# base_model = MobileNet(classes=classes,  weights="imagenet",dropout=0.5, include_top=False,input_tensor=Input(shape=(128, 128, 3)), input_shape=input_shape)

# # remove the last fully connected layer
# base_model = Sequential(base_model.layers[:-10])
# # freeze all the weights of the model except the last 4 layers
# for layer in base_model.layers[:-4]:
#     layer.trainable = False
# #adding output layer  
# model = Flatten()(base_model.output) 
# model=Dense(2,activation='softmax')(model) 

# model = Model(inputs=base_model.input, outputs=model)
# model.summary()

from keras.optimizers import Adam
opt =Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss=loss_func, metrics = ['acc'])
print('Layers: {}, parameters: {}'.format(len(model.layers), model.count_params()))
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
file_path="_MobileNet_weights.best.hdf5"
checkpoint = ModelCheckpoint(os.path.join(out_dir,file_path),
                                monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                mode='min')
#                              monitor='val_loss', 
#                              verbose=1,
#                              save_best_only=True,
#                              mode='auto')
early = EarlyStopping(monitor="val_acc",
                      mode="max",
                      patience=4)
callbacks_list = [checkpoint] 
%env JOBLIB_TEMP_FOLDER=/tmp
history = model.fit_generator(train_generator, 
                    steps_per_epoch = STEP_SIZE_TRAIN, 
                    validation_data = valid_generator,
                    validation_steps =STEP_SIZE_VALID ,
                    epochs=epochs,
                   callbacks = callbacks_list)
import math
score = model.evaluate_generator(generator=valid_generator,
steps=STEP_SIZE_VALID)
print("Accuracy:",score[1])
print("Loss    :",score[0])
from IPython.display import FileLink
FileLink(file_path)
from keras.models import load_model
out_dir = "./"
model.load_weights(os.path.join(out_dir,file_path)) # load the best model
training_loss_values = history.history['loss']
validation_loss_values = history.history['val_loss']

epochs = range(1, len(training_loss_values)+1)

plt.plot(epochs, training_loss_values, label='Training Loss')
plt.plot(validation_loss_values, label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
training_acc_values = history.history['acc']
validation_acc_values = history.history['val_acc']

epochs = range(1, len(training_loss_values)+1)

plt.plot(epochs, training_acc_values, label='Training acc')
plt.plot(validation_acc_values, label = 'Validation acc')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()

plt.show()
#add test if available
from sklearn.metrics import classification_report, confusion_matrix

valid_generator.reset()
pred=model.predict_generator(valid_generator,steps=STEP_SIZE_VALID,verbose=1)
print(pred)
y_true = valid_generator.classes
y_pred =list(np.argmax(pred ,axis=-1).flatten()) 

print("kk",np.shape(y_true))
print("gg",np.shape(y_pred))
print(train_generator.class_indices)
print(y_true)

print(y_pred)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true,y_pred ))
# plt.matshow(confusion_matrix(y_true,y_pred ))
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_true,y_pred )
fig, ax1 = plt.subplots(1,1)
ax1.plot(fpr, tpr, 'r.', label = 'MobileNet (AUC:%2.2f)' % roc_auc_score(y_pred, y_true))
ax1.plot(fpr, fpr, 'b-', label = 'Random Guessing')
ax1.legend()
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
# clean up the virtual directories
import shutil
for c_dir in glob('v_*'):
    shutil.rmtree(c_dir)


