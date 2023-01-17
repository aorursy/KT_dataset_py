!pip install --upgrade pip
!pip install -q efficientnet

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
import matplotlib.pyplot as plt
import csv
import os,math,re
import tensorflow as tf
from tensorflow import keras
import pydicom as dcm
import pylab
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from skimage import io
from skimage import measure
from skimage.transform import resize
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

print(tf.__version__)
# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
#loading and reading data set in pandas dataframe
os.chdir('/kaggle/input/rsna-pneumonia-detection-challenge/')

#os.chdir('/content/competitions/rsna-pneumonia-detection-challenge')
data_detailed_class_info = pd.read_csv('stage_2_detailed_class_info.csv')
data_train_labels = pd.read_csv('stage_2_train_labels.csv')
data_sample_submission = pd.read_csv('stage_2_sample_submission.csv')
print("=============================================================================================================")
print("Data stage_2_detailed_class_info.csv : ")
print("=============================================================================================================")
print("# stage_2_detailed_class_info.csv -  shape of the data       : " , data_detailed_class_info.shape)
print("# stage_2_detailed_class_info.csv -  Empty data count        : " , data_detailed_class_info.isnull().sum())
print("")
print("")
print(data_detailed_class_info.head(3))
print("")
print('# Count of Unique patientId values in stage_2_detailed_class_info.csv file : ',data_detailed_class_info['patientId'].nunique() )    
print("")
print("=============================================================================================================")
print("Data stage_2_train_labels.csv : ")
print("=============================================================================================================")
print("# stage_2_train_labels.csv -  shape of the data       : " , data_train_labels.shape)
print("# stage_2_train_labels.csv -  Empty data count        : " , data_train_labels.isnull().sum())
print("")
print("")
print(data_train_labels.head(10))
print("")
print('# Count of Unique patientId values in stage_2_train_labels.csv file : ',data_train_labels['patientId'].nunique() )
print("")
print("=============================================================================================================")
print("Data stage_2_sample_submission.csv : ")
print("=============================================================================================================")
print("# stage_2_sample_submission.csv -  shape of the data       : " , data_sample_submission.shape)
print("# stage_2_sample_submission.csv -  Empty data count        : " , data_sample_submission.isnull().sum())
print("")
print("")
print(data_sample_submission.head(3))
print("")
print('# Count of Unique patientId values in stage_2_sample_submission.csv file : ',data_sample_submission['patientId'].nunique() )
df_detailed_class_info=data_detailed_class_info
df_data_train_labels=data_train_labels
print(df_detailed_class_info.groupby('class').size().reset_index(name='count'))
dfTemp=df_detailed_class_info.groupby('class').size()
ax=dfTemp.plot(kind='bar',color=list('yrg'),figsize=(10, 10), fontsize=8)
ax.set_xlabel("class", fontsize=20)
ax.set_ylabel("count", fontsize=20)
ax.set_title('Pneumonia Class Count')

for p in ax.patches:
  ax.annotate(np.round(p.get_height(),decimals=2),(p.get_x()+p.get_width()/2., p.get_height()),ha='center',va='center',xytext=(0, 10),textcoords='offset points')
plt.show()

#Merging the data frame of 'stage_2_detailed_class_info.csv' and 'stage_2_train_labels.csv'
data_combined=df_data_train_labels.merge(df_detailed_class_info,left_on='patientId', right_on='patientId', how='inner')
print("Merged data :",data_combined.shape)
dfNew=data_combined.groupby(['Target','patientId']).size().reset_index(name='count')
ax=dfNew.groupby('count').size().plot(kind='bar',color=list('yrgb'),figsize=(10, 10),fontsize=8)
ax.set_xlabel("count of Pneumonia spotted per patient", fontsize=20)
ax.set_ylabel("count of patients", fontsize=20)
ax.set_title('Pneumonia locations per Image')
for p in ax.patches:
  ax.annotate(np.round(p.get_height(),decimals=2),(p.get_x()+p.get_width()/2., p.get_height()),ha='center',va='center',xytext=(0, 10),textcoords='offset points')
plt.show()
print(dfNew.groupby('count').size())
print("Minimum width of Pneumonia : ",data_combined['width'].min())
print("Minimum height of Pneumonia : ",data_combined['height'].min())
dfOps=data_combined.groupby('Target').size().reset_index(name='count')
dfOps['Target']=dfOps['Target'].replace(0,'Absence').replace(1,'Presence')
print("Summary of abscence and presence of pneumonia: \n " ,dfOps.head())
print("")
print("Example of Patient with presence of pneumonia with multiple areas : \n", df_data_train_labels.iloc[4:6])
print("Example of Patient where pneumonia absent : \n", df_data_train_labels.iloc[0:1])
plt.figure()
plt.title('Pneumonia width lengths')
plt.hist(data_combined[data_combined['Target'] > 0 ]['width'], bins=np.linspace(0,1000,50))
plt.show()
plt.figure()
plt.title('Pneumonia height lengths')
plt.hist(data_combined[data_combined['Target'] > 0 ]['height'], bins=np.linspace(0,1000,50))
plt.show()
#onlyfiles = next(os.walk('/content/competitions/rsna-pneumonia-detection-challenge/stage_2_train_images/'))[2] #dir is your directory path as string
onlyfiles = next(os.walk('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/'))[2]

print("Number of dcm files present in directory: ",len(onlyfiles))
#os.chdir('/content/competitions/rsna-pneumonia-detection-challenge/stage_2_train_images/')
os.chdir('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/')
#trying to view the dcm file for a particular patientId
patientId = data_combined['patientId'][5]
dcm_file = '%s.dcm' % patientId
dcm_data = dcm.read_file(dcm_file)
print("Format of .dcm files for a particular random patientId:")
print("===================================================================")
print(dcm_data)
im = dcm_data.pixel_array
print(type(im))
print(im.dtype)
print(im.shape)
pylab.imshow(im, cmap=pylab.cm.gist_gray)
#common functions to draw pneumonia location.
def getPneumoniaLocation(dataFrame):
    #create a list of location column
    l_get_loc= lambda rec: [rec['y'],rec['x'],rec['height'],rec['width']]
    #create python dictionary to stored from the dataframe patients label and location
    dictPatient={}
    for index, row in dataFrame.iterrows():
        l_patientid = row['patientId']
        #add patients in dictionary 
        if l_patientid not in dictPatient:
            dictPatient[l_patientid] = {
                #'dicom': '/content/competitions/rsna-pneumonia-detection-challenge/stage_2_train_images/%s.dcm' % l_patientid,
                'dicom': '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/%s.dcm' % l_patientid,
                'label': row['Target'],
                'location': []}
         
        #if patient is having pneumonia then add the details of location in dictionary
        if dictPatient[l_patientid]['label'] == 1:
            dictPatient[l_patientid]['location'].append(l_get_loc(row))
    
    return dictPatient


def displayPneumoniaImage(dataPatient):
    #open the file present in the input data key value 'dicom'
    workingFile = dcm.read_file(dataPatient['dicom'])
    im = workingFile.pixel_array
    #convert from single channel to 3 channel
    im = np.stack([im] * 3, axis=2)
    for rectangle in dataPatient['location']:
        rgb = np.floor(np.random.rand(3) * 256).astype('int')
        #converts coordinates to integer
        rectangle = [int(b) for b in rectangle]
        y1, x1, height, width = rectangle
        y2 = y1 + height
        x2 = x1 + width
        im[y1:y1 + 10, x1:x2] = rgb
        im[y2:y2 + 10, x1:x2] = rgb
        im[y1:y2, x1:x1 + 10] = rgb
        im[y1:y2, x2:x2 + 10] = rgb

    pylab.figure(figsize=(10,10))
    pylab.imshow(im, cmap=pylab.cm.gist_gray)
    pylab.axis('off')

#common function to shuffle file names and split  file names to training and validation set
def shuffleFileNames(folder,setName):
    filenames = os.listdir(folder)
    random.shuffle(filenames)
#Split data into train and validation samples, assumption is to split 90% train and 10% validation
    n_valid_samples = 2700
    train_filenames = filenames[n_valid_samples:]
    valid_filenames = filenames[:n_valid_samples]
    if setName == "Training":
        return train_filenames
    if setName == "Validation":
        return valid_filenames
    
    return train_filenames 

print("Random pateintId's picked to show the bounding box images : \n" ,data_train_labels.loc[data_train_labels['patientId'].isin(['0100515c-5204-4f31-98e0-f35e4b00004a','00704310-78a8-4b38-8475-49f4573b2dbb','00436515-870c-4b36-a041-de91049b9ab4','01adfd2f-7bc7-4cef-ab68-a0992752b620'])])
pneumoniaLocation=getPneumoniaLocation(data_combined)
displayPneumoniaImage(pneumoniaLocation['00436515-870c-4b36-a041-de91049b9ab4'])
displayPneumoniaImage(pneumoniaLocation['00704310-78a8-4b38-8475-49f4573b2dbb'])
displayPneumoniaImage(pneumoniaLocation['0100515c-5204-4f31-98e0-f35e4b00004a'])
displayPneumoniaImage(pneumoniaLocation['01adfd2f-7bc7-4cef-ab68-a0992752b620'])
patientid= data_combined['patientId'][0]
displayPneumoniaImage(pneumoniaLocation[patientid])
def getDictLocationFileName(dataFrame):
    pneumoniaLocation={}
    for index,row in dataFrame.iterrows():
        if row['Target'] == 1:
            filename = row[0]
            location = row[1:5]
            location = [int(float(i)) for i in location]
            if filename in pneumoniaLocation:
                pneumoniaLocation[filename].append(location)
            else:
                pneumoniaLocation[filename] = [location]
    
    return pneumoniaLocation
            
def iou_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
    return 1 - score

def iou_bce_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)

def cosine_annealing(x):
    lr = 0.001
    epochs = 25
    return lr*(np.cos(np.pi*x/epochs)+1.)/2

def mean_iou(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))
#To handle large data set avoid memory consumption during loading of large files 
#by using Keras we will create data generator class ,with multiprocessing batches will be loading and also images will be trained.

class dataGenerator(keras.utils.Sequence):
    def __init__(self, folder, filenames, pneumonia_locations=None, batch_size=32, image_size=256, shuffle=True, augment=False, predict=False):
        self.folder = folder
        self.filenames = filenames
        self.pneumonia_locations = pneumonia_locations
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.predict = predict
        self.on_epoch_end()
        
    def __load__(self, filename):
        # load dicom file as numpy array
        img = dcm.dcmread(os.path.join(self.folder, filename)).pixel_array
        # create empty mask
        msk = np.zeros(img.shape)
        # get filename without extension
        filename = filename.split('.')[0]
        # if image contains pneumonia
        if filename in self.pneumonia_locations:
            # loop through pneumonia
            for location in self.pneumonia_locations[filename]:
                # add 1's at the location of the pneumonia
                x, y, w, h = location
                msk[y:y+h, x:x+w] = 1
        # resize both image and mask
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        msk = resize(msk, (self.image_size, self.image_size), mode='reflect') > 0.5
        # if augment then horizontal flip half the time
        if self.augment and random.random() > 0.5:
            img = np.fliplr(img)
            msk = np.fliplr(msk)
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        msk = np.expand_dims(msk, -1)
        return img, msk
    
    def __loadpredict__(self, filename):
        # load dicom file as numpy array
        img = dcm.dcmread(os.path.join(self.folder, filename)).pixel_array
        # resize image
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        return img      
    
    #function which returns batch of images and filenames while training and while predicted returns masks and images
    def __getitem__(self, index):
        filenames= self.filenames[index*self.batch_size:(index+1)*self.batch_size]
        if self.predict:
            # load files
            imgs = [self.__loadpredict__(filename) for filename in filenames]
            # create numpy batch
            imgs = np.array(imgs)
            return imgs, filenames
        else:
            # load files
            items = [self.__load__(filename) for filename in filenames]
            # unzip images and masks
            imgs, msks = zip(*items)
            # create numpy batch
            imgs = np.array(imgs)
            msks = np.array(msks)
            return imgs, msks

    def on_epoch_end(self):
        #triggered at start and end of each epoch,while shuffling of file names we will get a new order of epochs        if self.shuffle:
            if self.shuffle:
                random.shuffle(self.filenames)
    
    def __len__(self):
        #'Denotes the number of batches per epoch'
        if self.predict:
            return int(np.ceil(len(self.filenames) / self.batch_size))
        else:
            return int(len(self.filenames) / self.batch_size)
        
    
#folder='/content/competitions/rsna-pneumonia-detection-challenge/stage_2_train_images/'
folder='/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/'
X_train=shuffleFileNames(folder,'Training')
print(len(X_train))
X_Valid=shuffleFileNames(folder,'Validation')
print(len(X_Valid))
pneumonia_locations=getDictLocationFileName(data_combined)
train_gen = dataGenerator(folder, X_train, pneumonia_locations=pneumonia_locations, batch_size=32, image_size=256, shuffle=True, augment=True, predict=False)
valid_gen = dataGenerator(folder, X_Valid, pneumonia_locations=pneumonia_locations, batch_size=32, image_size=256, shuffle=False, predict=False)

def create_downsample(channels, inputs):
    x = keras.layers.BatchNormalization(momentum=0.99)(inputs)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 1, padding='same', use_bias=False)(x)
    x = keras.layers.MaxPool2D(2)(x)
    return x

def create_resblock(channels, inputs):
    x = keras.layers.BatchNormalization(momentum=0.99)(inputs)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(momentum=0.99)(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    return keras.layers.add([x, inputs])

def create_network(input_size, channels, n_blocks=2, depth=4):
    # input
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(inputs)
    # residual blocks
    for d in range(depth):
        channels = channels * 2
        x = create_downsample(channels, x)
        for b in range(n_blocks):
            x = create_resblock(channels, x)
    # output
    x = keras.layers.BatchNormalization(momentum=0.99)(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    outputs = keras.layers.UpSampling2D(2**depth)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
train_gen = dataGenerator(folder, X_train, pneumonia_locations=pneumonia_locations, batch_size=32, image_size=256, shuffle=True, augment=True, predict=False)
valid_gen = dataGenerator(folder, X_Valid, pneumonia_locations=pneumonia_locations, batch_size=32, image_size=256, shuffle=False, predict=False)
modelResNet = create_network(input_size=256, channels=32, n_blocks=2, depth=4)
opt = keras.optimizers.Adam(learning_rate=0.01)
modelResNet.compile(optimizer=opt,loss=iou_bce_loss,metrics=['accuracy', mean_iou])
modelHist=modelResNet.fit(train_gen, validation_data=valid_gen, epochs=5)

plt.figure(figsize=(12,4))
plt.subplot(131)
plt.plot(modelHist.epoch, modelHist.history["loss"], label="Train loss")
plt.plot(modelHist.epoch, modelHist.history["val_loss"], label="Valid loss")
plt.legend()
plt.subplot(132)
plt.plot(modelHist.epoch, modelHist.history["accuracy"], label="Train accuracy")
plt.plot(modelHist.epoch, modelHist.history["val_accuracy"], label="Valid accuracy")
plt.legend()
plt.subplot(133)
plt.plot(modelHist.epoch, modelHist.history["mean_iou"], label="Train iou")
plt.plot(modelHist.epoch, modelHist.history["val_mean_iou"], label="Valid iou")
plt.legend()
plt.show()
modelHist.history
import matplotlib.patches as patches
for imgs, msks in valid_gen:
    # predict batch of images
    preds = modelResNet.predict(imgs)
    # create figure
    f, axarr = plt.subplots(4, 8, figsize=(20,15))
    axarr = axarr.ravel()
    axidx = 0
    # loop through batch
    for img, msk, pred in zip(imgs, msks, preds):
        # plot image
        axarr[axidx].imshow(img[:, :, 0])
        # threshold true mask
        comp = msk[:, :, 0] > 0.5
        # apply connected components
        comp = measure.label(comp)
        # apply bounding boxes
        predictionString = ''
        for region in measure.regionprops(comp):
            # retrieve x, y, height and width
            y, x, y2, x2 = region.bbox
            height = y2 - y
            width = x2 - x
            axarr[axidx].add_patch(patches.Rectangle((x,y),width,height,linewidth=2,edgecolor='b',facecolor='none'))
        # threshold predicted mask
        comp = pred[:, :, 0] > 0.5
        # apply connected components
        comp = measure.label(comp)
        # apply bounding boxes
        predictionString = ''
        for region in measure.regionprops(comp):
            # retrieve x, y, height and width
            y, x, y2, x2 = region.bbox
            height = y2 - y
            width = x2 - x
            axarr[axidx].add_patch(patches.Rectangle((x,y),width,height,linewidth=2,edgecolor='r',facecolor='none'))
        axidx += 1
    plt.show()
    # only plot one batch
    break
test_folder='/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_test_images/'
test_filenames = os.listdir(test_folder)
print('n test samples:', len(test_filenames))

# create test generator with predict flag set to True
test_gen = dataGenerator(test_folder, test_filenames, None, batch_size=25, image_size=256, shuffle=False, predict=True)

# create submission dictionary
submission_dict = {}
# loop through testset
for imgs, filenames in test_gen:
    # predict batch of images
    preds = modelResNet.predict(imgs)
    # loop through batch
    for pred, filename in zip(preds, filenames):
        # resize predicted mask
        pred = resize(pred, (1024, 1024), mode='reflect')
        # threshold predicted mask
        comp = pred[:, :, 0] > 0.5
        # apply connected components
        comp = measure.label(comp)
        # apply bounding boxes
        predictionString = ''
        for region in measure.regionprops(comp):
            # retrieve x, y, height and width
            y, x, y2, x2 = region.bbox
            height = y2 - y
            width = x2 - x
            # proxy for confidence score
            conf = np.mean(pred[y:y+height, x:x+width])
            # add to predictionString
            predictionString += str(conf) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + ' '
        # add filename and predictionString to dictionary
        filename = filename.split('.')[0]
        submission_dict[filename] = predictionString
    # stop if we've got them all
    #if len(submission_dict) >= len(test_filenames):
    if len(submission_dict) >= 100:
        break
print(submission_dict.items())
print(submission_dict.keys())
print(plt.imshow(dcm.dcmread('/content/competitions/rsna-pneumonia-detection-challenge/stage_2_test_images/21ea7be5-b0a4-4d96-b56d-8288dd0292cd.dcm').pixel_array, cmap=plt.cm.bone) )

EarlyStopping=tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.05, patience=2, verbose=0, mode='auto',baseline=None, restore_best_weights=True)
filepath="/kaggle/working/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath,monitor='val_loss',mode='min',save_best_only=True,verbose=1)
train_gen = dataGenerator(folder, X_train, pneumonia_locations=pneumonia_locations, batch_size=100, image_size=256, shuffle=True, augment=True, predict=False)
valid_gen = dataGenerator(folder, X_Valid, pneumonia_locations=pneumonia_locations, batch_size=100, image_size=256, shuffle=False, predict=False)
model = create_network(input_size=256, channels=32, n_blocks=2, depth=4)
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,loss=iou_bce_loss,metrics=['accuracy', mean_iou])
learning_rate = tf.keras.callbacks.LearningRateScheduler(cosine_annealing)
modelHistIncreaseBatchSize=model.fit(train_gen, validation_data=valid_gen, callbacks=[learning_rate,EarlyStopping,ModelCheckpoint],epochs=6)

modelHistIncreaseBatchSize.history
plt.figure(figsize=(12,4))
plt.subplot(131)
plt.plot(modelHistIncreaseBatchSize.epoch, modelHistIncreaseBatchSize.history["loss"], label="Train loss")
plt.plot(modelHistIncreaseBatchSize.epoch, modelHistIncreaseBatchSize.history["val_loss"], label="Valid loss")
plt.legend()
plt.subplot(132)
plt.plot(modelHistIncreaseBatchSize.epoch, modelHistIncreaseBatchSize.history["accuracy"], label="Train accuracy")
plt.plot(modelHistIncreaseBatchSize.epoch, modelHistIncreaseBatchSize.history["val_accuracy"], label="Valid accuracy")
plt.legend()
plt.subplot(133)
plt.plot(modelHistIncreaseBatchSize.epoch, modelHistIncreaseBatchSize.history["mean_iou"], label="Train iou")
plt.plot(modelHistIncreaseBatchSize.epoch, modelHistIncreaseBatchSize.history["val_mean_iou"], label="Valid iou")
plt.legend()
plt.show()
%cd /kaggle/input/pneumoniadetection-v1
!ls -lrt weights-improvement*
modelInputSize = create_network(input_size=128, channels=32, n_blocks=2, depth=4)
opt = keras.optimizers.Adam(learning_rate=0.01)
modelInputSize.compile(optimizer=opt,loss=iou_bce_loss,metrics=['accuracy', mean_iou])
modelInputSize.load_weights('/kaggle/input/pneumoniadetection-v1/weights-improvement-02-0.63.hdf5')
learning_rate = tf.keras.callbacks.LearningRateScheduler(cosine_annealing)
EarlyStopping=tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.05, patience=2, verbose=0, mode='auto',baseline=None, restore_best_weights=True)
modelHistDecreaseInputSize=modelInputSize.fit(train_gen, validation_data=valid_gen, callbacks=[learning_rate,EarlyStopping],epochs=6)
print(modelInputSize.score())
print(modelResNet.score())
print(model.score())
plt.figure(figsize=(12,4))
plt.subplot(131)
plt.plot(modelHistIncreaseBatchSize.epoch, modelHistIncreaseBatchSize.history["loss"], label="Train loss")
plt.plot(modelHistIncreaseBatchSize.epoch, modelHistIncreaseBatchSize.history["val_loss"], label="Valid loss")
plt.legend()
plt.subplot(132)
plt.plot(modelHistIncreaseBatchSize.epoch, modelHistIncreaseBatchSize.history["accuracy"], label="Train accuracy")
plt.plot(modelHistIncreaseBatchSize.epoch, modelHistIncreaseBatchSize.history["val_accuracy"], label="Valid accuracy")
plt.legend()
plt.subplot(133)
plt.plot(modelHistIncreaseBatchSize.epoch, modelHistIncreaseBatchSize.history["mean_iou"], label="Train iou")
plt.plot(modelHistIncreaseBatchSize.epoch, modelHistIncreaseBatchSize.history["val_mean_iou"], label="Valid iou")
plt.legend()
plt.show()
def map_iou(boxes_true, boxes_pred, scores, thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    Mean average precision at differnet intersection over union (IoU) threshold
    
    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output: 
        map: mean average precision of the image
    """
    
    # According to the introduction, images with no ground truth bboxes will not be 
    # included in the map score unless there is a false positive detection (?)
        
    # return None if both are empty, don't count the image in final evaluation (?)
    
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None
    
    assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]
    
    map_total = 0
    
    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1 # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1 # bt has no match, count as FN
                
        fp = len(boxes_pred) - len(matched_bt) # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        map_total += m
    
    return map_total / len(thresholds)
test_folder='/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_test_images/'
test_filenames = os.listdir(test_folder)
print('n test samples:', len(test_filenames))

# create test generator with predict flag set to True
test_gen = dataGenerator(test_folder, test_filenames, None, batch_size=25, image_size=256, shuffle=False, predict=True)
prob_thresholds = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
nthresh = len(prob_thresholds)
count = 0
ns = nthresh*[0]
nfps = nthresh*[0]
ntps = nthresh*[0]
overall_maps = nthresh*[0.]
for imgs, filenames in test_gen:
    # predict batch of images
    preds = model.predict(imgs)
    # loop through batch
    for pred, filename in zip(preds, filenames):
        count = count + 1
        maxpred = np.max(pred)
        # resize predicted mask
        pred = resize(pred, (1024, 1024), mode='reflect')
        # threshold predicted mask
        boxes_preds = []
        scoress = []
        for thresh in prob_thresholds:
            comp = pred[:, :, 0] > thresh
            # apply connected components
            comp = measure.label(comp)
            # apply bounding boxes
            boxes_pred = np.empty((0,4),int)
            scores = np.empty((0))
            for region in measure.regionprops(comp):
                # retrieve x, y, height and width
                y, x, y2, x2 = region.bbox
                boxes_pred = np.append(boxes_pred, [[x, y, x2-x, y2-y]], axis=0)
                # proxy for confidence score
                conf = np.mean(pred[y:y2, x:x2])
                scores = np.append( scores, conf )
            boxes_preds = boxes_preds + [boxes_pred]
            scoress = scoress + [scores]
        boxes_true = np.empty((0,4),int)
        fn = filename.split('.')[0]
        # if image contains pneumonia
        if fn in pneumonia_locations:
            # loop through pneumonia
            for location in pneumonia_locations[fn]:
                x, y, w, h = location
                boxes_true = np.append(boxes_true, [[x, y, w, h]], axis=0)
        for i in range(nthresh):
            if ( boxes_true.shape[0]==0 and boxes_preds[i].shape[0]>0 ):
                ns[i] = ns[i] + 1
                nfps[i] = nfps[i] + 1
            elif ( boxes_true.shape[0]>0 ):
                ns[i] = ns[i] + 1
                contrib = map_iou( boxes_true, boxes_preds[i], scoress[i] ) 
                overall_maps[i] = overall_maps[i] + contrib
                if ( boxes_preds[i].shape[0]>0 ):
                    ntps[i] = ntps[i] + 1

    # stop if we've got them all
    if count >= len(test_filenames):
        break
print(plt.imshow(dcm.dcmread('/content/competitions/rsna-pneumonia-detection-challenge/stage_2_test_images/1b8d026f-3989-4cd6-a4c7-4ed8fb8887d0.dcm').pixel_array, cmap=plt.cm.bone) )
print(plt.imshow(dcm.dcmread('/content/competitions/rsna-pneumonia-detection-challenge/stage_2_test_images/1e8359a0-6313-4908-9971-5682d02db185.dcm').pixel_array, cmap=plt.cm.bone) )