import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle           
import matplotlib.pyplot as plt             
import cv2                                 
import tensorflow as tf                
from tqdm import tqdm
class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}
nb_classes = len(class_names)
IMAGE_SIZE = (150, 150)
print(class_names_label['sea'])
def load():
    datasets=['../input/intel-image-classification/seg_train/seg_train','../input/intel-image-classification/seg_test/seg_test']
    output=[]
    for dataset in datasets:
        images=[]
        labels=[]
        for folder in os.listdir(dataset):
            for file in tqdm(os.listdir(os.path.join(dataset,folder))):
                labels.append(class_names_label[folder])
                im_path=os.path.join(os.path.join(dataset,folder),file)
                image=cv2.imread(im_path)
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image=cv2.resize(image,IMAGE_SIZE) 
                images.append(image)
        images=np.array(images,dtype='float32')
        labels=np.array(labels,dtype='int32')
        output.append((images,labels))
    return output
    
(train_images,train_labels),(test_images,test_labels)=load()

train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
print(train_images.shape)
print(train_labels.shape[0])
print(test_images.shape)
print(test_labels.shape[0])
_, train_counts = np.unique(train_labels, return_counts=True)
plt.pie(train_counts,
        labels=class_names,
        autopct='%1.1f%%')
plt.axis('equal')
plt.title('Proportion of each class')
plt.show()
train_images = train_images / 255.0 
test_images = test_images / 255.0

def display_examples(class_names, images, labels):
     
    fig = plt.figure(figsize=(15,15))
    fig.suptitle("Some images from the dataset", fontsize=20)
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i+50], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i+50]])
    plt.show()
display_examples(class_names, train_images, train_labels)

model=tf.keras.Sequential([tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
                          tf.keras.layers.MaxPooling2D(2,2),
                          tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
                          tf.keras.layers.MaxPooling2D(2,2),
                          tf.keras.layers.Flatten(),
                          tf.keras.layers.Dense(128, activation='relu'),
                          tf.keras.layers.Dense(6, activation='softmax')])
model.summary()
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import RMSprop
base_model =VGG16(input_shape=(150,150,3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
base_model.summary()    

last_layer = base_model.get_layer('block4_pool')
print('last layer output shape: ', last_layer.output_shape)
last_output =last_layer.output 
x = layers.Flatten()(last_output)
x = layers.Dense(72,activation='relu')(x)
x = layers.Dense(6,activation='softmax')(x)   
model = Model(base_model.input,x) 
model.summary()
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import RMSprop
base_model =VGG16(input_shape=(150,150,3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
base_model.summary()    

last_layer = base_model.get_layer('block3_pool')
print('last layer output shape: ', last_layer.output_shape)
last_output =last_layer.output 
x=layers.Conv2D(32,(3,3),activation='relu')(last_output)
x=layers.Conv2D(64,(3,3),activation='relu')(x)
x=layers.MaxPooling2D(2,2)(x)
x=layers.Flatten()(x)
x=layers.Dense(72,activation='relu')(x)
x=layers.Dropout(0.2)(x)
x = layers.Dense(6,activation='softmax')(x)   
model = Model(base_model.input,x) 
model.summary()
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import RMSprop

pretrained_model=ResNet50( input_shape=(150,150,3),
                                  include_top=False,
                                  weights='imagenet'
                                   )

pretrained_model.trainable = False


for layer in pretrained_model.layers:
    if "BatchNormalization" in layer.__class__.__name__:
        layer.trainable = True

last_layer = pretrained_model.get_layer('conv5_block3_out')
print('last layer of resnet : output shape: ', last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(102, activation='relu')(x)
x = layers.Dropout(0.3)(x)                  
x = layers.Dense(6, activation='softmax')(x)

model_resnet = Model(pretrained_model.input, x) 

model_resnet.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'sparse_categorical_crossentropy', 
              metrics = ['acc'])

history = model_resnet.fit(train_images, train_labels, batch_size=128, epochs=20, validation_split = 0.2)
def plot_accuracy_loss(history):
   
    fig = plt.figure(figsize=(10,5))

    plt.subplot(221)
    plt.plot(history.history['acc'],'bo--', label = "acc")
    plt.plot(history.history['val_acc'], 'ro--', label = "val_acc")
    plt.title("train_acc vs val_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    plt.subplot(222)
    plt.plot(history.history['loss'],'bo--', label = "loss")
    plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")

    plt.legend()
    plt.show()
plot_accuracy_loss(history)    
test_loss = model_resnet.evaluate(test_images, test_labels)
predictions = model_resnet.predict(test_images) 
print(predictions.shape)
pred_labels = np.argmax(predictions, axis = 1)
def print_mislabeled_images(class_names, test_images, test_labels, pred_labels):
   
    BOO = (test_labels == pred_labels)
    mislabeled_indices = np.where(BOO == 0)
    mislabeled_images = test_images[mislabeled_indices]
    mislabeled_labels = pred_labels[mislabeled_indices]

    title = "Some examples of mislabeled images by the classifier:"
    display_examples(class_names,  mislabeled_images, mislabeled_labels)
print_mislabeled_images(class_names, test_images, test_labels, pred_labels)
CM = confusion_matrix(test_labels, pred_labels)
ax = plt.axes()
sn.heatmap(CM, annot=True, 
           annot_kws={"size": 10}, 
           xticklabels=class_names, 
           yticklabels=class_names, ax = ax)
ax.set_title('Confusion matrix')
plt.show()