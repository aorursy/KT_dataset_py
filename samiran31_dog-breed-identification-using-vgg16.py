%matplotlib inline
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from os import listdir, makedirs
from os.path import join, exists, expanduser
from tqdm import tqdm
from sklearn.metrics import log_loss, accuracy_score
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications import xception
from keras.applications import inception_v3
from keras.applications.vgg16 import preprocess_input, decode_predictions
from sklearn.linear_model import LogisticRegression

from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split
from keras import  backend as K
from keras.preprocessing.image import ImageDataGenerator,load_img
start = dt.datetime.now()
label=pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
label
label_df=pd.DataFrame(label['breed'].value_counts()).reset_index()
label_df.columns=['breed_name','count']
label_df=label_df.head(16)
label_df
label_df.sort_values(by="count",ascending=False)
label = label[label['breed'].isin(label_df['breed_name'])]
    
label['id_ext']=label['id'].apply(lambda x:x+'.jpg')
label=label.reset_index()
label=label.drop(['index','id'],axis=1)
label.head()
label_onehot=pd.get_dummies(label,columns=['breed'],prefix=None)
label_onehot
label_onehot.columns
label_onehot.columns = label_onehot.columns.str.replace(r'breed_', '')
#label_onehot
label_onehot=label_onehot.rename(columns={'id_ext':'id'})
label_onehot
import random
sample=random.choice(label_onehot['id'])
sample
image=load_img("/kaggle/input/dog-breed-identification/train/"+sample)
image
train_df, validate_df = train_test_split(label_onehot, test_size=0.1)
train_df = train_df.reset_index()
validate_df = validate_df.reset_index()

# validate_df = validate_df.sample(n=100).reset_index() # use for fast testing code purpose
# train_df = train_df.sample(n=1800).reset_index() # use for fast testing code purpose

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
train_df.shape,validate_df.shape
train_df
from keras.models import Sequential
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation,GlobalMaxPooling2D
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications import VGG16
from keras.models import Model

image_size = 224
input_shape = (image_size, image_size, 3)

epochs = 4
batch_size = 16

pre_trained_model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")
    
for layer in pre_trained_model.layers[:15]:
    layer.trainable = False

for layer in pre_trained_model.layers[15:]:
    layer.trainable = True
    
last_layer = pre_trained_model.get_layer('block5_pool')
last_output = last_layer.output
    
# Flatten the output layer to 1 dimension
x = GlobalMaxPooling2D()(last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x = Dense(512, activation='relu')(x)
# Add a dropout rate of 0.5
x = Dropout(0.5)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(16, activation='softmax')(x)

model = Model(pre_trained_model.input, x)

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.summary()
train_df
train_df.columns
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "/kaggle/input/dog-breed-identification/train", 
    x_col='id',
    y_col=['afghan_hound', 'airedale', 'basenji', 'beagle',
       'bernese_mountain_dog', 'cairn', 'entlebucher', 'great_pyrenees',
       'japanese_spaniel', 'leonberg', 'maltese_dog', 'pomeranian', 'samoyed',
       'scottish_deerhound', 'shih-tzu', 'tibetan_terrier'],
    class_mode='raw',
    target_size=(image_size, image_size),
    batch_size=batch_size
)
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "/kaggle/input/dog-breed-identification/train", 
    x_col='id',
    y_col=['afghan_hound', 'airedale', 'basenji', 'beagle',
       'bernese_mountain_dog', 'cairn', 'entlebucher', 'great_pyrenees',
       'japanese_spaniel', 'leonberg', 'maltese_dog', 'pomeranian', 'samoyed',
       'scottish_deerhound', 'shih-tzu', 'tibetan_terrier'],
    class_mode='raw',
    target_size=(image_size, image_size),
    batch_size=batch_size
)
example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "/kaggle/input/dog-breed-identification/train/", 
    x_col='id',
    y_col=['afghan_hound', 'airedale', 'basenji', 'beagle',
       'bernese_mountain_dog', 'cairn', 'entlebucher', 'great_pyrenees',
       'japanese_spaniel', 'leonberg', 'maltese_dog', 'pomeranian', 'samoyed',
       'scottish_deerhound', 'shih-tzu', 'tibetan_terrier'],
    class_mode='raw',
)
plt.figure(figsize=(12, 12))
for i in range(0, 9):
    plt.subplot(3, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()
# fine-tune the model
history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size)
loss, accuracy = model.evaluate_generator(validation_generator, total_validate//batch_size, workers=12)
print("Test: accuracy = %f  ;  loss = %f " % (accuracy, loss))
validate_df
def get_dog(row):
    for i in validate_df.columns[2:]:
        if row[i]==1:
            return i        
validate_df['breed']=validate_df.apply(get_dog,axis=1)
validate_df
validate_df=validate_df[['id','breed']]
validate_df.shape
sample_test = validate_df.sample(n=9).reset_index()
#print(sample_test)
plt.figure(figsize=(12, 12))
for index, row in sample_test.iterrows():
    filename = row['id']
    print(filename)
    category = row['breed']
    img = load_img("/kaggle/input/dog-breed-identification/train/"+filename, target_size=(256, 256))
    
    plt.subplot(3, 3,index+1)
    plt.imshow(img)
    plt.xlabel('(' + "{}".format(category) + ')')
plt.tight_layout()
plt.show()

end = dt.datetime.now()
print('Total time {} s.'.format((end - start).seconds))
