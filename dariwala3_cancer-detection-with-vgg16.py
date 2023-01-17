import pandas as pd
import os

# Check folder
print(os.listdir("../input"))

# Save train labels to dataframe
df = pd.read_csv("../input/histopathologic-cancer-detection/train_labels.csv")

# Remove error image
df = df[df['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']

# Remove error black image
df = df[df['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']

# Show first rows
df.head()
# True positive diagnosis
df_true_positive = df[df["label"] == 1]
print("True positive diagnosis: " + str(len(df_true_positive)))

# True negative diagnosis
df_true_negative = df[df["label"] == 0]
print("True negative diagnosis: " + str(len(df_true_negative)))
# Undersampling of dominant class to reach a 1/1 balance

from sklearn.utils import shuffle

df_true_negative = shuffle(df_true_negative)
df_true_negative = df_true_negative[0:88800]

df_true_positive = shuffle(df_true_positive)
df_true_positive = df_true_positive[0:88800]

# concat the dataframes
df = pd.concat([df_true_negative, df_true_positive], axis=0)

# shuffle
df = shuffle(df)
df['label'].value_counts()
# Split data set  to train and validation sets
from sklearn.model_selection import train_test_split

# Use stratify= df['label'] to get balance ratio 1/1 in train and validation sets
df_train, df_val = train_test_split(df, test_size=0.2, stratify= df['label'])

# Check balancing
print("True positive in train data: " +  str(len(df_train[df_train["label"] == 1])))
print("True negative in train data: " +  str(len(df_train[df_train["label"] == 0])))
print("True positive in validation data: " +  str(len(df_val[df_val["label"] == 1])))
print("True negative in validation data: " +  str(len(df_val[df_val["label"] == 0])))
# Delete directory
import shutil
shutil.rmtree('main', ignore_errors=True)

# Create directory
os.mkdir('main')

# Create subfolder for train and val images
os.mkdir(os.path.join('main', 'train'))
os.mkdir(os.path.join('main', 'val'))

# Create subfolders for true positive and true negative in train
os.mkdir(os.path.join('main','train','true_positive'))
os.mkdir(os.path.join('main','train','true_negative'))      
         
# Create subfolders for true positive and true negative in val
os.mkdir(os.path.join('main','val','true_positive'))
os.mkdir(os.path.join('main','val','true_negative'))
# Prepare image name classes for the directory structure
# Save all train true positive names to list and add .tif
train_true_positive = df_train[df_train["label"] == 1]['id'].tolist()
train_true_positive = [name + ".tif" for name in train_true_positive]

# Save all train true negativeto names list and add .tif
train_true_negative = df_train[df_train["label"] == 0]['id'].tolist()
train_true_negative = [name + ".tif" for name in train_true_negative]

# Save all val true positive "id" to list and add .tif
val_true_positive = df_val[df_val["label"] == 1]['id'].tolist()
val_true_positive = [name + ".tif" for name in val_true_positive]

# Save all val true negative "id" to list
val_true_negative = df_val[df_val["label"] == 0]['id'].tolist()
val_true_negative = [name + ".tif" for name in val_true_negative]
from PIL import Image
from PIL import ImageDraw
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

image_numbers = 3 
fig = plt.figure()
fig, ax = plt.subplots(2,image_numbers, figsize=(18,8))

for i in range(0,image_numbers):
    
    # Random images with true positive diagnosis
    pos_train_names = shuffle(val_true_positive)
    image = Image.open(os.path.join("../input/histopathologic-cancer-detection/train",pos_train_names[i]))
    print(image.size)
    draw = ImageDraw.Draw(image)
    draw.rectangle([(32,32),(64,64)], outline="yellow")
    ax[0,i].imshow(image)
    ax[0,i].set_title("True Positive",fontsize=14)
    
    # Random images with true negative diagnosis
    neg_train_names = shuffle(train_true_negative)
    image = Image.open(os.path.join("../input/histopathologic-cancer-detection/train",neg_train_names[i]))
    draw = ImageDraw.Draw(image)
    draw.rectangle([(32,32),(64,64)], outline="yellow")
    ax[1,i].imshow(image)
    ax[1,i].set_title("True Negative",fontsize=14)
# Move images to directory structure
import shutil
import os
from tqdm import tqdm

def transfer(source,destination,files):
    for image in tqdm(files):
        # source path to image
        src = os.path.join(source,image)
        dst = os.path.join(destination,image)
        # copy the image from the source to the destination
        shutil.copyfile(src,dst)
        
# transfer
transfer('../input/histopathologic-cancer-detection/train','main/train/true_positive',train_true_positive)
transfer('../input/histopathologic-cancer-detection/train','main/train/true_negative',train_true_negative)
transfer('../input/histopathologic-cancer-detection/train','main/val/true_positive',val_true_positive)
transfer('../input/histopathologic-cancer-detection/train','main/val/true_negative',val_true_negative)
# Import VGG16 model, with weights pre-trained on ImageNet.
from keras.applications.vgg16 import VGG16, preprocess_input

# VGG model without the last classifier layers (include_top = False)
vgg16_model = VGG16(include_top = False,
                    input_shape = (96,96,3),
                    #weights='../input/VGG16weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
                    weights = '../input/vgg16weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    
# Freeze the layers 
for layer in vgg16_model.layers[:-12]:
    layer.trainable = False
    
# Check the trainable status of the individual layers
for layer in vgg16_model.layers:
    print(layer, layer.trainable)
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout

model = Sequential()
model.add(vgg16_model)
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))
# Load previous weights to save the time
#model.load_weights('../input/pretrained3/model.h5')
# Generate batches of tensor image data with real-time data augmentation. 
import numpy as np
num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 32
val_batch_size = 32

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

print(train_steps)
print(val_steps)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Augmentation 
train_datagen = ImageDataGenerator(
                rescale=1./255,
                vertical_flip=True,
                horizontal_flip=True,
                rotation_range=90,
                shear_range=0.05)

# Augmentation 
# train_datagen = ImageDataGenerator(
#                 rescale=1./255,
#                 vertical_flip=True, # set True or False
#                 horizontal_flip=True, # set True or False
#                 rotation_range=180, # 0-180 range, degrees of rotatation
#                 shear_range=0.01,# 0-1 range for shearing
#                 width_shift_range=0.2, # 0-1 range, horizontal translation
#                 height_shift_range=0.2) # 0-1 range vertical translation

#train_datagen = ImageDataGenerator(rescale=1./255)

# Augmentation configuration for validatiopn and testing: only rescaling!
test_datagen = ImageDataGenerator(rescale=1./255)

# Generator that will read pictures found in subfolers of 'main/train', and indefinitely generate batches of augmented image data
train_generator = train_datagen.flow_from_directory('main/train',
                                            target_size=(96,96),
                                            batch_size=train_batch_size,
                                            class_mode='categorical')

val_generator = test_datagen.flow_from_directory('main/val',
                                            target_size=(96,96),
                                            batch_size=val_batch_size,
                                            class_mode='categorical')

# !!! batch_size=1 & shuffle=False !!!!
test_generator = test_datagen.flow_from_directory('main/val',
                                            target_size=(96,96),
                                            batch_size=1,
                                            class_mode='categorical',
                                            shuffle=False)

# Get the labels that are associated with each index
print(val_generator.class_indices)
from keras.optimizers import Adam, SGD
from keras import optimizers

#model.compile(Adam(lr=0.00001), loss='binary_crossentropy', metrics=['acc'])
model.compile(loss='binary_crossentropy',optimizer=optimizers.SGD(lr=0.00001, momentum=0.95),metrics=['accuracy'])

# Due to Disk limits the saving of best weights isn't possible during the training process                                            
history = model.fit_generator(
                    train_generator, 
                    steps_per_epoch  = train_steps, 
                    validation_data  = val_generator,
                    validation_steps = val_steps,
                    epochs           = 20, 
                    verbose          = 1)
# Plot validation and accuracies over epochs
import matplotlib.pyplot as plt

train_acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(len(train_acc))

plt.plot(epochs,train_acc,'b',label='Training accuracy')
plt.plot(epochs,val_acc,'r',label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.show()
# Due to the disk limits I couldn't save the best model during the training process
print("Validation Accuracy: " + str(history.history['val_acc'][-1:]))
# Prediction on validation data sets
val_predict = model.predict_generator(test_generator, steps=len(df_val), verbose=1)
from sklearn.metrics import confusion_matrix
print("Confusion matrix is: ")
print(confusion_matrix(test_generator.classes, val_predict.argmax(axis=1)))
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(test_generator.classes, val_predict.argmax(axis=1))   
# Compute ROC area
print("ROC area is: " + str(auc(fpr, tpr)))

plt.figure()
plt.plot(fpr, tpr, color='darkred', label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.show()
from tqdm import tqdm

# Delete directory structure
shutil.rmtree('main')

# Create test folder
os.mkdir('test')
    
# Create test_images folder inside test folder
os.mkdir(os.path.join('test','test_images'))

# Save test image identification names
test_images = os.listdir('../input/histopathologic-cancer-detection/test')

# Move images to test folder
for test_image in tqdm(test_images):   
    # source 
    src = os.path.join('../input/histopathologic-cancer-detection/test',test_image)
    # destination 
    dst = os.path.join('test/test_images',test_image)
    # copy the image
    shutil.copyfile(src, dst)
    
# !!! batch_size=1 & shuffle=False !!!!
test_generator = test_datagen.flow_from_directory('test',
                                            target_size=(96,96),
                                            batch_size=1,
                                            class_mode='categorical',
                                            shuffle=False)

# model.load_weights('best_model.h5')
# Predict 
test_predict = model.predict_generator(test_generator, steps=57458, verbose=1)
# Extract test names from test_generator
test_names = [name.split("/")[1].split(".")[0] for name in test_generator.filenames]

# Create a dataframe
df_pred_test = pd.DataFrame(test_names, columns=["id"])

# !!! label == probability of true positive (has cancer) !!!
df_pred_test["label"] = pd.DataFrame(test_predict[:,1])

df_pred_test.head()
# Save sample submissions as dataframe
df_samples = pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv')

# Merge df_samples and df_pred_test using unique key combination through "id"
submission = pd.merge(df_samples.drop("label",axis=1), df_pred_test, on = 'id')
submission.head()
shutil.rmtree('test')

# Save submission file
submission.to_csv("submission.csv",index=False) 

# Due to the disk limits I can only save the last model
model.save('model.h5')
