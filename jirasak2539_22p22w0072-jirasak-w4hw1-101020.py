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
import numpy as np
import tensorflow as tf
import pandas as pd
import keras as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# locate the necessary file location
img_folder = '../input/thai-mnist-classification/train'
csv_file = '../input/thai-mnist-classification/mnist.train.map.csv'
df = pd.read_csv(csv_file,dtype = 'str')
# use the transfer-learning with pre-trained weights
input_t = K.Input(shape = (224,224,3))
conv_base = K.applications.VGG19(include_top=False,
                                     weights='imagenet',
                                     input_tensor = input_t)
conv_base.summary()
# define the function to extract the feature from image using the pre-trained model before passsing 
# the extracted data to our model

TARGET_SIZE = (224,224)
BATCH_SIZE = 8
def extract_features(subset,validation_split):
    
    if subset == 'training':
        datagen = ImageDataGenerator(rescale=1./255,validation_split= validation_split)
        generator = datagen.flow_from_dataframe(dataframe=df,
                                                    directory=img_folder,
                                                    x_col = 'id', y_col = 'category',
                                                    subset = 'training',
                                                    shuffle = True, seed = 0,
                                                    class_mode='categorical',
                                                    target_size = TARGET_SIZE,
                                                    batch_size = BATCH_SIZE
                                                    )
    elif subset == 'validation':
        datagen = ImageDataGenerator(rescale=1./255,validation_split= validation_split)
        generator = datagen.flow_from_dataframe(dataframe=df,
                                                    directory=img_folder,
                                                    x_col = 'id', y_col = 'category',
                                                    subset = 'validation',
                                                    shuffle = True, seed = 0,
                                                    class_mode='categorical',
                                                    target_size = TARGET_SIZE,
                                                    batch_size = BATCH_SIZE
                                                    )
    sample_counts =  generator.samples   
    features = np.zeros([sample_counts,7,7,512])
    labels = np.zeros([sample_counts,10])    
        
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = features_batch
        labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = labels_batch
        i = i+1
        print(i)
        if i * BATCH_SIZE >= sample_counts:
            break
    return features,labels, generator
# extract the feature from our data, set the validation-split to be 0.15
train_features, train_labels, train_gen = extract_features('training',0.15)
val_features, val_labels, val_gen = extract_features('validation',0.15)
# define the sequential model which we will pass the extracted feature in to it

from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.regularizers import l2

model = Sequential()
model.add(GlobalAveragePooling2D(input_shape=(7,7,512)))
model.add(Dropout(0.05))
model.add(Dense(10))
model.add(Dense(10, activation='softmax'))
model.summary()
# set the checkpoint to save the weight when the validation loss is new lower than the current lowest one
checkpoint = K.callbacks.ModelCheckpoint('model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', 
                                         verbose=1, monitor='val_loss',
                                         save_best_only=True, 
                                         mode='auto')  

# Compile model
from keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['acc'])

# Train model
history = model.fit(train_features, train_labels,
                    epochs=300,
                    batch_size=32, 
                    callbacks=[checkpoint],
                    validation_data=(val_features, val_labels))

# visualize the loss and the accuracy of the model
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'g', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
# visualize the loss and the accuracy of the model
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'g', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
from tensorflow.keras.models import load_model
import os
# load the best weight from the trained model
model = load_model('../input/best-weight/model-239-0.945895-0.966061.h5')
# define the function to output the prediction and the generator from the input folder
def predict_class(folder):
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(directory=folder,
                                                shuffle = False,
                                                class_mode='categorical',
                                                target_size=(224, 224),
                                                batch_size = 1
                                                )
    predicted_class_list = []
    sample_counts =  generator.samples
    for i in range(sample_counts):
        print(i)
        features = conv_base.predict(generator[i][0])
        try:
            prediction = model.predict(features)
        except:
            prediction = model.predict(features.reshape(1,7*7*512))
        classes = list(range(0,10))
        predicted_class = classes[np.argmax(np.array(prediction[0]))]

        predicted_class_list.append(predicted_class)
    return predicted_class_list,generator

# took very long time due to explicit for loop over ~12000 images
predicted_class_list,generator = predict_class('../input/thai-mnist-classification')
filenames = generator.filenames
filenames
# visualize some picture and its prediction

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.figure(figsize = (10,10))
for i in range(15):
    plt.subplot(3,5,i+1)
    img_path = np.random.choice(filenames)
    index = filenames.index(img_path)
    predicted_label  =  predicted_class_list[index]
    img = mpimg.imread(os.path.join('../input/thai-mnist-classification',img_path))
    plt.imshow(img)
    plt.title(predicted_label)
    plt.axis('off')
plt.tight_layout()
# remove the directory prefix from the test_generator.filenames
filenames = pd.Series(generator.filenames).apply(lambda x: x.split('/')[1])
filenames = list(filenames)
# build dictionary to match filenames with the predicted class
predict_dict = dict(zip(filenames,predicted_class_list))
# replace the image string with the prediction number 
test_rule_csv = pd.read_csv('../input/thai-mnist-classification/test.rules.csv')
predicted_test_rule_csv = test_rule_csv.replace(predict_dict)
predicted_test_rule_csv
# replace the image string with the prediction number 
train_rule_csv = pd.read_csv('../input/thai-mnist-classification/train.rules.csv')
predicted_train_rule_csv = train_rule_csv.replace(predict_dict)
# save the predicted test rule to csv file
predicted_test_rule_csv.to_csv('./predicted_test_rule.csv',index = False)
# save the predicted train rule to csv file
predicted_train_rule_csv.to_csv('./predicted_train_rule.csv',index = False)
# define the prediction rules
def solve(f1,f2,f3):
    
    def solve3(f2,f3):
        if f2 == f3+1:
            return 0
        if f2 <= f3:
            return f3 + solve3(f2,f3-1)
        if f2 >= f3:
            return solve3(f2,f3+1) + f3+1
    
    
    if pd.isna(f1):
        return f2+f3
    if f1 == 0:
        return f2*f3
    if f1 == 1:
        return abs(f2-f3)
    if f1 == 2:
        return (f2+f3) * abs(f2-f3)
    if f1 == 3:
        return solve3(f2,f3)
    if f1 == 4:
        return 50+f2-f3
    if f1 == 5:
        return min(f2,f3)
    if f1 == 6:
        return max(f2,f3)
    if f1 == 7:
        return (f2*f3*11)%99
    if f1 == 8:
        return (((f2**2+1)*f2) + f3*(f3+1))%99
    if f1 == 9:
        return 50 + f2
        
# load the predicted file
csv_file = pd.read_csv('../input/predicted-rule/predicted_test_rule (1).csv')
csv_file
csv_file['predict'] = csv_file.apply(lambda x: solve(x['feature1'],x['feature2'],x['feature3']),axis = 1)
csv_file
submission = pd.read_csv("../input/thai-mnist-classification/submit.csv")
submission
submission['predict'] = csv_file['predict']
submission.to_csv('submission.csv',index = False)
