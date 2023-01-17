# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install -U git+https://github.com/qubvel/efficientnet
!pip install imgaug
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os 
import tqdm
import gc

import matplotlib.pyplot as plt
import numpy as np

from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras.layers import *
from keras.callbacks import *
import tensorflow as tf

from keras.regularizers import l1

from tensorflow.keras.utils import to_categorical


from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects

import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.nasnet import  preprocess_input

from keras import optimizers

#from cutmix_keras import CutMixImageDataGenerator
from keras.callbacks import LearningRateScheduler


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
import itertools

from keras.callbacks import EarlyStopping
df=pd.read_csv("/kaggle/input/celeba-dataset/list_attr_celeba.csv")
df
df1=df[["image_id","Black_Hair",'Blond_Hair',"Brown_Hair","Gray_Hair"]]
df1.loc[df1['Black_Hair']==-1,"Black_Hair"]= np.nan
df1.loc[df1['Blond_Hair']==-1,'Blond_Hair']= np.nan
df1.loc[df1["Brown_Hair"]==-1,"Brown_Hair"]= np.nan
df1.loc[df1["Gray_Hair"]==-1,"Gray_Hair"]= np.nan
black=pd.DataFrame({"Black_Hair":df1['Black_Hair'].to_list()},index=df1['image_id']).dropna()
brown=pd.DataFrame({"Black_Hair":df1['Brown_Hair'].to_list()},index=df1['image_id']).dropna()
gray=pd.DataFrame({"Black_Hair":df1['Gray_Hair'].to_list()},index=df1['image_id']).dropna()
blond=pd.DataFrame({"Black_Hair":df1['Blond_Hair'].to_list()},index=df1['image_id']).dropna()
black_hair_value=0
brown_hair_value=2
gray_hair_value=3
blond_hair_value=1

black_label_images=black.index.to_list()
blond_label_images=blond.index.to_list()
brown_label_images=brown.index.to_list()
gray_label_images=gray.index.to_list()
source='/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/*.jpg'
names=sorted(glob.glob(source))
total=len(black)+len(brown)+len(gray)+len(blond)
labels=[]
for j,i in tqdm.tqdm(enumerate((names))):   
    #print(i.split("\\")[1])
    if i.split("/")[6] in black_label_images:
        labels.append(black_hair_value)
    elif i.split("/")[6] in blond_label_images:
        labels.append(blond_hair_value)
    elif i.split("/")[6] in brown_label_images:
        labels.append(brown_hair_value)
    elif i.split("/")[6] in gray_label_images:
           labels.append(gray_hair_value)
    else:
            pass
import efficientnet.keras as eff
from efficientnet.keras import preprocess_input ,center_crop_and_resize
model1 = eff.EfficientNetB5(weights='noisy-student',input_shape=(218,178,3),include_top=False)
model1.trainable=True
def custom_gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
get_custom_objects().update({'custom_gelu': Activation(custom_gelu)})
for i in range(len(model1.layers)):
    model1.layers[i].activation="custom_gelu"
newOutputs = model1.output
x = GlobalAveragePooling2D()(newOutputs)
x= Dense(800,activation="custom_gelu")(x)
x= Dense(500,activation="custom_gelu")(x)
output = Dense(4, activation='softmax', name='predictions',use_bias=True)(x)
newModel1 = Model(model1.input, output)
training_label=np.array(labels)
values=sorted(blond.append(gray.append(black.append(brown))).index.to_list())
image_list=[]
for i in values:
    image_list.append("/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/"+i)
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, image_path, labels,index,shuffle=False,augment=False,batch_size=16,n_classes=4,dims=(218, 178, 3)):
        'Initialization'
        self.image_path=image_path
        self.labels=labels
        self.index=index
        self.shuffle=shuffle
        self.augment=augment
        self.batch_size=batch_size
        self.dims=dims
        self.n_classes=n_classes
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_path) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        print(index)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        #assert for no empty list
        assert indexes!=[],"Indexes cannot be empty caused with in this index ranges {0} and with {1}".format(indexes,index)
        # Find list of IDs
        image_array = np.array([cv2.imread(self.image_path[k]) for k in indexes])
        #assert shape of the array
        assert image_array.shape==(len(image_array),*self.dims)
        #labels
        labels=[self.labels[k] for k in indexes]
        #assert for labels 
        assert len(labels)!=0,"Length cannot be zero"
        #convert to categorical
        labels=keras.utils.to_categorical(labels, num_classes=self.n_classes)
        return image_array,labels
    
    def on_epoch_end(self):
        'Updates indexes before each epoch'
        self.indexes = self.index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
len(training_label)
training_label[7301*16:7302*16]
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
		                                            patience=10,
		                                            verbose=1,
		                                            factor=0.5,
		                                            min_lr=0.0000001)

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule,verbose=1)

lr_sched = step_decay_schedule(initial_lr=3e-3, decay_factor=0.25, step_size=10)
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.09, random_state=0)
for train_index, test_index in sss.split(np.zeros(len(training_label)), training_label):
    print("TRAIN:", train_index, "TEST:", test_index)
    print("Training length",len(train_index),"Testing length",len(test_index))
train_data=DataGenerator(image_path=image_list,
                         labels=training_label,
                         index=train_index,
                         shuffle=True,
                         augment=False)
# val_data=DataGenerator(images_paths=image_list,
#                          labels=training_label,
#                          ids=test_index,
#                          shuffle=True,
#                          augment=False)
newModel1.compile(optimizers.Nadam(), loss="categorical_crossentropy", metrics=["acc"])
with tf.device('/CPU:0'):
    EPOCHS = 10
    h=newModel1.fit_generator(generator=train_data,
                              epochs=EPOCHS,
                              steps_per_epoch=len(train_data),
                              #validation_data=train_data,
                              #validation_steps =len(val_data),
                              callbacks=[EarlyStopping(patience=2, restore_best_weights=True),
                                   lr_sched,])
keras.utils.plot_model(newModel1, to_file="model.png", show_shapes=True)
# for i in range(10):
#     print("batch change\n")
#     newModel1.fit(train_data[i][0],train_data[i][1],epochs=4)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title='Normalized confusion matrix'
    else:
        title='Confusion matrix'
    fig = plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
## multiclass or binary report
## If binary (sigmoid output), set binary parameter to True
def full_multiclass_report(model,
                           x,
                           y_true,
                           classes,
                           batch_size=32,
                           binary=False):

    # 1. Transform one-hot encoded y_true into their class number
    if not binary:
        y_true = np.argmax(y_true,axis=1)
    
    # 2. Predict classes and stores in y_pred
    y_pred = model.predict(x, batch_size=batch_size)
    y_pred= np.argmax(y_pred,axis=1)
    
    # 3. Print accuracy score
    print("Accuracy : "+ str(accuracy_score(y_true,y_pred)))
    
    print("")
    
    # 4. Print classification report
    print("Classification Report")
    print(classification_report(y_true,y_pred,digits=5))    
    
    # 5. Plot confusion matrix
    cnf_matrix = confusion_matrix(y_true,y_pred)
    return(cnf_matrix)


matrix=full_multiclass_report(newModel1,
                       x_test,
                       y_test1,
                      range(3))
plot_confusion_matrix(matrix,classes=range(3))