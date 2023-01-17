import os
import zipfile
import numpy as np 
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
np.random.seed(9)
tf.random.set_seed(9)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

weights_file = '/kaggle/input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(weights_file)
#pre_trained_model.summary()
for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
with zipfile.ZipFile("../input/dogs-vs-cats/train.zip","r") as z:
    z.extractall("../kaggle/working/train_unzip")
    
print(f"We have total {len(os.listdir('../kaggle/working/train_unzip/train'))} images in our training data.")
filenames = os.listdir('../kaggle/working/train_unzip/train')
labels = [str(fname)[:3] for fname in filenames]
train_df = pd.DataFrame({'filename': filenames, 'label': labels})
train_df.head()
print((train_df['label']).value_counts())
train_set_df, dev_set_df = train_test_split(train_df[['filename', 'label']], test_size=0.3, random_state = 42, shuffle=True, stratify=train_df['label'])
print(train_set_df.shape, dev_set_df.shape)
print('Training Set image counts:')
print(train_set_df['label'].value_counts())
print('Validation Set image counts:')
print(dev_set_df['label'].value_counts())
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen  = ImageDataGenerator( rescale = 1.0/255 )
train_generator = train_datagen.flow_from_dataframe(
    train_set_df, 
    directory="../kaggle/working/train_unzip/train/", 
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    class_mode='binary',
    batch_size=32,
    validate_filenames=False 
)

validation_generator = validation_datagen.flow_from_dataframe(
    dev_set_df, 
    directory="../kaggle/working/train_unzip/train/", 
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    class_mode='binary',
    batch_size=32,
    validate_filenames=False 
)
x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense  (1, activation='sigmoid')(x)           

model = Model( pre_trained_model.input, x ) 

model.summary()
model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])
history = model.fit(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 100,
            epochs = 20,
            validation_steps = 50)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs   = range(len(acc))

plt.plot(epochs, acc, label="Training accuracy")
plt.plot(epochs, val_acc, label="Validation accuracy")
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, label="Training loss")
plt.plot(epochs, val_loss, label="Validation loss")
plt.title('Training and validation loss')
plt.legend()
plt.show()
loss, accuracy = model.evaluate_generator(validation_generator)
print("Test: accuracy = %f  ;  loss = %f " % (accuracy, loss))
dev_true = dev_set_df['label'].map({'dog': 1, "cat": 0})
dev_predictions =  model.predict_generator(validation_generator)
dev_set_df['pred'] = np.where(dev_predictions>0.5, 1, 0)
dev_pred = dev_set_df['pred']
dev_set_df.head()
dev_set_predictions_plot = dev_set_df['pred'].value_counts().plot.bar(title='Predicted number of Dog vs Cat Images in dev set')
confusion_mtx = confusion_matrix(dev_true, dev_pred) 

f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Blues",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
with zipfile.ZipFile("../input/dogs-vs-cats/test1.zip","r") as z:
    z.extractall("../kaggle/working/test1_unzip")
    
print(f"We have total {len(os.listdir('../kaggle/working/test1_unzip/test1'))} images in our test1.zip")
test_filenames = os.listdir('../kaggle/working/test1_unzip/test1')
test_df = pd.DataFrame({'filename': test_filenames})
test_df.head()
test_generator = validation_datagen.flow_from_dataframe(
    test_df, 
    directory="../kaggle/working/test1_unzip/test1/", 
    x_col='filename',
    y_col=None,
    target_size=(150, 150),
    class_mode=None,
    batch_size=32,
    validate_filenames=False 
)
predictions = model.predict_generator(test_generator, steps=np.ceil(len(test_filenames)/32))
test_df['id'] = test_df['filename'].str.split('.').str[0]
test_df['label'] = np.where(predictions>0.5, 1, 0)
result_df = test_df[['id','label']]
result_df.head()
test_set_predictions_plot = dev_set_df['pred'].value_counts().plot.bar(title='Predicted number of Dog vs Cat Images in test set')
result_df.to_csv("cats_vs_dogs.csv",index=False)
sample_test = test_df.sample(n=9)

plt.figure(figsize=(12, 12))
labels ={0:'cat', 1:'dog'}
for i, row in sample_test.reset_index(drop=True).iterrows():
    filename = row['filename']
    category = labels[row['label']]
    img = load_img("../kaggle/working/test1_unzip/test1/"+filename, target_size=(150, 150))
    plt.subplot(3, 3, i+1)
    plt.imshow(img)   
    plt.xlabel('(' + "{}".format(category) + ')')
 
    
plt.tight_layout()
plt.show()