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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
data_dir = '../input/intel-image-classification'
os.listdir(data_dir)
test_dir = data_dir + '/seg_test/seg_test'
train_dir = data_dir + '/seg_train/seg_train'
pred_dir = data_dir + '/seg_pred/seg_pred'
plt.imshow(imread(train_dir+'/buildings/'+os.listdir(train_dir+'/buildings')[0])) # an example of a building
plt.imshow(imread(train_dir+'/forest/'+os.listdir(train_dir+'/forest')[0])) # an example of a forest
plt.imshow(imread(train_dir+'/glacier/'+os.listdir(train_dir+'/glacier')[0])) # an example of a glacier
plt.imshow(imread(train_dir+'/mountain/'+os.listdir(train_dir+'/mountain')[0])) # an example of a mountain
plt.imshow(imread(train_dir+'/sea/'+os.listdir(train_dir+'/sea')[0])) # an example of a sea
plt.imshow(imread(train_dir+'/street/'+os.listdir(train_dir+'/street')[0])) # an example of street
# this is done so if there are any image of variable shapes so we will reshape all of them to an average shape
dim1 = []
dim2 = []

for image_file in os.listdir(train_dir+'/buildings'):
    img = imread(train_dir+'/buildings/'+image_file)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)
height = int(np.average(d1))
height
width = int(np.average(d2))
width
img_shape = (height,width,3)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# performing data augmentation and scaling on train set
image_gen_train = ImageDataGenerator(rescale=1/255,
                                    horizontal_flip=True,
                                    zoom_range=0.5,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.1,
                                    rotation_range=10,
                                    fill_mode='nearest')
# performing scaling on test set
image_gen_test = ImageDataGenerator(rescale=1/255)
# returns an iterator of tuples of (x,y)
# here x is like our x_train and y like y_train
train_data_gen = image_gen_train.flow_from_directory(directory=train_dir,
                                                    class_mode='categorical',
                                                    batch_size=128,
                                                    color_mode='rgb',
                                                    shuffle=True,
                                                    target_size=img_shape[:2])
# returns an iterator of tuples of (x,y)
# here x is like our x_train and y like y_train
test_data_gen = image_gen_test.flow_from_directory(directory=test_dir,
                                                  class_mode='categorical',
                                                  color_mode='rgb',
                                                  batch_size=128,
                                                  target_size=img_shape[:2],
                                                  shuffle=False)
train_data_gen.class_indices
test_data_gen.class_indices
from tensorflow.keras.applications import MobileNetV2
# instantiating a base model which we will not be trained and we'll import this model along with trained weights and biases
# (actually this model was trained on imagenet dataset) we'll not include top of layers of that model because they are less generic
# instead we'll add on more layers later so that this model could make predictions on this dataset.
base_model = MobileNetV2(include_top=False,
                        weights='imagenet',
                        input_shape=img_shape)
base_model.trainable = False # freezing the base model layers to avoid it's retraining.
base_model.summary()
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
global_layer = GlobalAveragePooling2D() # this layer provides us a vetor of features from the just previous volume of base model.
pred_layer = Dense(6) # this layer makes raw predictions i.e, it returns numbers as logits.
model = Sequential([base_model,global_layer,pred_layer]) # our modified model.
model.summary() # see we have some trainable parameters these are due to the layers which we added on later.
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
model.compile(optimizer=Adam(),
             loss=CategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=2)
history = model.fit(train_data_gen,
         validation_data=test_data_gen,
         epochs=15,
         callbacks=[early_stop])
# trend of losses
loss_metrics = pd.DataFrame(model.history.history)
loss_metrics
loss_metrics[['loss','val_loss']].plot(title='LOSS VS EPOCH COUNT')
loss_metrics[['accuracy','val_accuracy']].plot(title='ACCURACY VS EPOCH COUNT')
test_data_gen.classes
predictions = model.predict_classes(test_data_gen)
predictions
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(test_data_gen.classes,predictions))
print(confusion_matrix(test_data_gen.classes,predictions))
def predict_label(class_number):
    if class_number==0:
        return 'building'
    elif class_number==1:
        return 'forest'
    elif class_number==2:
        return 'glacier'
    elif class_number==3:
        return 'mountain'
    elif class_number==4:
        return 'sea'
    else:
        return 'street'
from tensorflow.keras.preprocessing import image
def predict_name(directory_to_img):
    pred_image = image.load_img(directory_to_img,target_size=img_shape)
    pred_image_array = image.img_to_array(pred_image)
    pred_image_array = pred_image_array/255
    pred_image_array = pred_image_array.reshape(1,150,150,3)
    prediction = model.predict_classes(pred_image_array)[0]
    plt.imshow(imread(directory_to_img))
    return predict_label(prediction)
predict_name(data_dir+'/seg_pred/seg_pred/'+os.listdir(data_dir+'/seg_pred/seg_pred/')[0])
predict_name(data_dir+'/seg_pred/seg_pred/'+os.listdir(data_dir+'/seg_pred/seg_pred/')[1])
predict_name(data_dir+'/seg_pred/seg_pred/'+os.listdir(data_dir+'/seg_pred/seg_pred/')[9])
predict_name(data_dir+'/seg_pred/seg_pred/'+os.listdir(data_dir+'/seg_pred/seg_pred/')[3])
predict_name(data_dir+'/seg_pred/seg_pred/'+os.listdir(data_dir+'/seg_pred/seg_pred/')[4])
predict_name(data_dir+'/seg_pred/seg_pred/'+os.listdir(data_dir+'/seg_pred/seg_pred/')[10])
predict_name(data_dir+'/seg_pred/seg_pred/'+os.listdir(data_dir+'/seg_pred/seg_pred/')[67])
predict_name(data_dir+'/seg_pred/seg_pred/'+os.listdir(data_dir+'/seg_pred/seg_pred/')[33])
predict_name(data_dir+'/seg_pred/seg_pred/'+os.listdir(data_dir+'/seg_pred/seg_pred/')[37])
predict_name(data_dir+'/seg_pred/seg_pred/'+os.listdir(data_dir+'/seg_pred/seg_pred/')[72])
# that's it for this notebook soon updating with even better classifier.