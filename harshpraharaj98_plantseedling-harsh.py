import os
import fnmatch
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import preprocess_input
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
np.random.seed(21)



import tensorflow as tf
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,BatchNormalization,Activation,Dropout,GlobalAveragePooling2D
from keras.models import Sequential,Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam


a = '/kaggle/input/plant-seedlings-classification'

print(os.listdir(os.path.join(a,'train')))
def get_train_data(root):
    
    """Performs required pre processing on the input images and fetches data

    Args:
      root: Directory in which we are working

    Returns: 
      train_img: A numpy array consisting of train images
      train_y : A OHE numpy array of train labels
    """
    train_dir = (os.path.join(root,'train'))
    train_label = []
    train_img = []
    label2num = {'Loose Silky-bent':0, 'Charlock':1, 'Sugar beet':2, 'Small-flowered Cranesbill':3,
                 'Common Chickweed':4, 'Common wheat':5, 'Maize':6, 'Cleavers':7, 'Scentless Mayweed':8,
                 'Fat Hen':9, 'Black-grass':10, 'Shepherds Purse':11}
    for i in os.listdir(train_dir):
        label_number = label2num[i]
        new_path = os.path.join(train_dir,i)
        for j in fnmatch.filter(os.listdir(new_path), '*.png'):
            temp_img = image.load_img(os.path.join(new_path,j), target_size=(128,128))
            train_label.append(label_number)
            temp_img = image.img_to_array(temp_img)
            train_img.append(temp_img)
        print(i)
    train_img = np.array(train_img)

    train_y=pd.get_dummies(train_label)
    train_y = np.array(train_y)
    train_img=preprocess_input(train_img)
    
    return train_img,train_y


train_img,train_y = get_train_data(a)
print('Training data shape: ', train_img.shape)
print('Training labels shape: ', train_y.shape)
from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid=train_test_split(train_img,train_y,test_size=0.1, random_state=42)
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

datagen.fit(X_train)

def vgg16_model(num_classes=None):
    
    """ Adding custom model to the VGG-16

    Args:
      num_classes: Number of layers in the final layer(Number of classes)

    Returns:
      model: Returns the custom model added to VGG
    """

    model = VGG16(weights='imagenet', include_top=False,input_shape=(128,128,3))
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()

    model.outputs = [model.layers[-1].output]

    #model.layers[-2].outbound_node= []
    x=Conv2D(256, kernel_size=(2,2),strides=2)(model.output)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)    
    x=Conv2D(128, kernel_size=(2,2),strides=1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x=Flatten()(x)
    x=Dense(num_classes, activation='softmax')(x)

    model=Model(model.input,x)

    for layer in model.layers[:15]:

        layer.trainable = False


    return model
from keras.applications.vgg16 import VGG16
from keras import backend as K
num_classes=12
model = vgg16_model(num_classes)
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
from keras.callbacks import ModelCheckpoint
epochs = 10
batch_size = 32
model_checkpoint = ModelCheckpoint('vgg_weights.h5', monitor='val_accuracy', save_best_only=True,mode='max',verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=7, min_lr=0.000001)
#early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=7, verbose=0, mode='min', restore_best_weights=True)

model.fit(datagen.flow(X_train,Y_train),
          batch_size=128,
          epochs=20,
          verbose=1, shuffle=True, validation_data=(X_valid,Y_valid), callbacks=[model_checkpoint,reduce_lr])
plt.style.use('seaborn')

# Accuracy History
def accuracy_curves(m):
    """ Plots the Train and validation Accuracy Curves

    Args:
      m: Model for which the curves are plotted

    Returns:
      
    """
    
    plt.figure(1, figsize=(16, 10))
    plt.plot(m.history.history['accuracy'])
    plt.plot(m.history.history['val_accuracy'])
    plt.title('Train and Validation Accuracy', fontsize = 16)
    plt.ylabel('Accuracy', fontsize = 14)
    plt.xlabel('Epoch', fontsize = 14)
    plt.legend(['Train', 'Test'], fontsize = 14)
    plt.show()
    
# Loss History

def loss_curves(m):
    """ Plots the Train and validation Loss Curves

    Args:
      m: Model for which the curves are plotted

    Returns:
      
    """
    plt.figure(2, figsize=(16, 10))
    plt.plot(m.history.history['loss'])
    plt.plot(m.history.history['val_loss'])
    plt.title('Train and Validation Loss', fontsize = 16)
    plt.ylabel('Loss', fontsize = 14)
    plt.xlabel('Epoch', fontsize = 14)
    plt.legend(['Train', 'Test'], fontsize = 14)
    plt.show()

    
from sklearn.metrics import classification_report

def class_report(m):
    """ Creates  Classification Report 

    Args:
      m: Model for which the report is generated

    Returns:
      
    """
    target_names = os.listdir(os.path.join(a,'train'))
    print(classification_report(Y_valid.argmax(axis=1), m.predict(X_valid).argmax(axis=1), target_names=target_names))
    
    
from sklearn.metrics import confusion_matrix

def conf_matrix(m):
    """ Creates  Confusion Matrix 

    Args:
      m: Model for which the confusion matrix is generated

    Returns:
      
    """
    target_names = os.listdir(os.path.join(a,'train'))
    cf_matrix = confusion_matrix(Y_valid.argmax(axis=1), m.predict(X_valid).argmax(axis=1))
    plt.figure(figsize=(20,20))
    sns.heatmap(cf_matrix/1000, annot=True, xticklabels=target_names, yticklabels=target_names, cmap='Blues')
accuracy_curves(model)
loss_curves(model)
class_report(model)
conf_matrix(model)
def resnet_model(num_classes=None):
    
    """ Adding custom model to the ResNet50

    Args:
      num_classes: Number of layers in the final layer(Number of classes)

    Returns:
      model: Returns the custom model added to ResNet
    """

    model = ResNet50(weights='imagenet', include_top=False,input_shape=(128,128,3))
    model.layers.pop()
    #model.layers.pop()
    #model.layers.pop()

    model.outputs = [model.layers[-1].output]

    #model.layers[-2].outbound_node= []
    x=Conv2D(256, kernel_size=(2,2),strides=2)(model.output)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)    
    x=Conv2D(128, kernel_size=(2,2),strides=1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x=Flatten()(x)
    x=Dense(num_classes, activation='softmax')(x)

    model=Model(model.input,x)

    for layer in model.layers[:15]:

        layer.trainable = False


    return model
from keras.applications.resnet50 import ResNet50
from keras import backend as K
num_classes=12
model_res = resnet_model(num_classes)
model_res.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
model_res.summary()
epochs = 10
batch_size = 32
model_checkpoint = ModelCheckpoint('resnet_weights.h5', monitor='val_accuracy', save_best_only=True,mode='max',verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=7, min_lr=0.000001)
#early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=7, verbose=0, mode='min', restore_best_weights=True)

model_res.fit(datagen.flow(X_train,Y_train),
          batch_size=128,
          epochs=20,
          verbose=1, shuffle=True, validation_data=(X_valid,Y_valid), callbacks=[model_checkpoint,reduce_lr])
accuracy_curves(model_res)
loss_curves(model_res)
class_report(model_res)
conf_matrix(model_res)
def incep_model(num_classes=None):
    
    """ Adding custom model to the InceptionV3

    Args:
      num_classes: Number of layers in the final layer(Number of classes)

    Returns:
      model: Returns the custom model added to Inception
    """

    model = InceptionV3(weights='imagenet', include_top=False,input_shape=(128,128,3))
    #model.layers.pop()
    #model.layers.pop()
    #model.layers.pop()

#    for layer in model.layers:
 #     layer.trainable = False
    '''
    model.outputs = model.output
    
    x = GlobalAveragePooling2D()(model.outputs)
    #x=Conv2D(256, kernel_size=(2,2),strides=2)(model.output)
    #x = BatchNormalization()(x)    
    #x=Conv2D(128, kernel_size=(2,2),strides=1)(x)
    #x = Activation('relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(128,activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    #x=Flatten()(x)
    x=Dense(num_classes, activation='softmax')(x)

    model=Model(model.input,x)'''

    
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)

    predictions = Dense(12, activation='softmax')(x)

    model = Model(model.input, predictions)




    return model
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K
num_classes=12
model_incep = incep_model(num_classes)
model_incep.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
model_incep.summary()
epochs = 10
batch_size = 32
model_checkpoint = ModelCheckpoint('incep_weights.h5', monitor='val_accuracy', save_best_only=True,mode='max',verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=7, min_lr=0.000001)
#early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=7, verbose=0, mode='min', restore_best_weights=True)

model_incep.fit(datagen.flow(X_train,Y_train),
          batch_size=128,
          epochs=20,
          verbose=1, shuffle=True, validation_data=(X_valid,Y_valid), callbacks=[model_checkpoint])
accuracy_curves(model_incep)
loss_curves(model_incep)
class_report(model_incep)
conf_matrix(model_incep)
!mkdir -p saved_models

model.save('m_vgg.h5')
model_res.save('m_res.h5')
model_incep.save('m_incep.h5')
new_model = tf.keras.models.load_model('m_res.h5')
new_model.summary()
def predict_on_test(model):
    """ Using the Model with the highest Validation accuracy to predict on test set

    Args:
      model: Model with highest validation accuracy(ResNet50)

    Returns:
      df: A Dataframe in a format required by Kaggle 
    """
    prob=[]
    num=[]
    test_img=[]
    test_path = os.path.join('/kaggle/input/plant-seedlings-classification','test')
    test_all = fnmatch.filter(os.listdir(test_path), '*.png')

    test_img=[]
    for i in range(len(test_all)):
        path=test_path+'/'+test_all[i]
        temp_img=image.load_img(path,target_size=(128,128))
        temp_img=image.img_to_array(temp_img)
        test_img.append(temp_img) 
    test_img=np.array(test_img)    
    test_img= tf.keras.applications.vgg16.preprocess_input(test_img)


    test_labels=[]
    pred=model.predict(test_img)
    num2label =  {0:'Loose Silky-bent', 1:'Charlock',2: 'Sugar beet',3: 'Small-flowered Cranesbill',
                  4:'Common Chickweed',5: 'Common wheat',6: 'Maize', 7:'Cleavers', 8:'Scentless Mayweed',
                 9: 'Fat Hen', 10:'Black-grass', 11:'Shepherds Purse'}
    for i in range(len(test_all)):
        max_score =0
        lab=-1
        for j in range(12):
            if pred[i][j]>max_score:
                max_score=pred[i][j]
                lab=j
        test_labels.append(num2label[lab])


    d = {'file': test_all, 'species': test_labels}
    df = pd.DataFrame(data=d)
    return df
df = predict_on_test(new_model)
print(df.head(5))
df.to_csv("/kaggle/working/submit.csv",index=False) 
