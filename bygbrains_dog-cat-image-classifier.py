import torchvision
import  torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms,models,datasets
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch import optim
train_data_dir = '/kaggle/input/cat-and-dog/training_set/training_set'

transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

dataset = torchvision.datasets.ImageFolder(train_data_dir, transform= transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=400 ,shuffle=True)
test_data_dir = '/kaggle/input/cat-and-dog/test_set/test_set'

transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

dataset = torchvision.datasets.ImageFolder(train_data_dir, transform= transform)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=400 ,shuffle=True)
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(20,150))
    plt.imshow(inp)

inputs, classes = next(iter(train_loader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs, scale_each= True)

imshow(out)
model = models.densenet121(pretrained = True)
for params in model.parameters():
    params.requires_grad = False
from collections import OrderedDict

classifier = nn.Sequential(OrderedDict([
    ('fc1',nn.Linear(1024,500)),
    ('relu',nn.ReLU()),
    ('fc2',nn.Linear(500,2)),
    ('Output',nn.LogSoftmax(dim=1))
]))

model.classifier = classifier
model = model.cuda()
test_dir="/kaggle/input/cat-and-dog/test_set"
train_dir="/kaggle/input/cat-and-dog/training_set"

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

batch_size=64

training_set = train_datagen.flow_from_directory(train_dir,
target_size = (100, 100),
batch_size = batch_size,
color_mode='rgb',
class_mode = 'binary',
shuffle=True)

test_set = test_datagen.flow_from_directory(test_dir,
target_size = (100, 100),
batch_size = batch_size,
color_mode='rgb',
class_mode = 'binary')
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
from keras.optimizers import adam
import numpy as np
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (100, 100, 3)))
classifier.add(Activation("relu"))
classifier.add(MaxPooling2D(pool_size = (3, 3)))
classifier.add(Conv2D(64, (3, 3), input_shape = (100, 100, 3)))
classifier.add(Activation("relu"))
classifier.add(MaxPooling2D(pool_size = (3, 3)))

classifier.add(Flatten())

classifier.add(Dense(64))
classifier.add(Activation("relu")) 
classifier.add(Dense(128))
classifier.add(Activation("relu")) 
classifier.add(Dense(activation = 'sigmoid', units=1))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit_generator(training_set,
                        steps_per_epoch=np.ceil(training_set.samples / batch_size),
                        epochs=20,
                        validation_steps=np.ceil(test_set.samples / batch_size),
                         validation_data=test_set
                        )

import joblib
# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(model, filename)
import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing import image
# test_image = image.load_img("/kaggle/input/dogs-vs-cats/train/dogs/dog.1.jpg", target_size = (200, 200)) 
# plt.imshow(test_image)
# plt.grid(None) 
# plt.show()
import cv2
sample_path='/kaggle/input/dogs-vs-cats/train/dogs/dog.1.jpg'
test_image=cv2.imread(sample_path)
pimg = test_image(test_image).unsqueeze(0).to(device)
plt.imshow(pimg)
print(pimg.shape)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img = Image.open(sample_path)
nimg = np.array(img)
plt.imshow(nimg)
pimg = transform(img).unsqueeze(0).to(device)
pimg.shape
res_list= ["It's a cat !","It's a dog !"]
prediction = model(pimg)
_, tpredict = torch.max(prediction.data, 1)
print(res_list[classes[tpredict[0].item()]])
model = joblib.load('finalized_model.sav')
est_loss = classifier.evaluate(test_images, test_labels)
model = joblib.load('finalized_model.sav')

predictions = classifier(test_image)     # Vector of probabilities
pred_labels = np.argmax(predictions, axis = 1)
loaded_model = joblib.load('finalized_model.sav')
res_list= ["It's a cat !","It's a dog !"]
test_image = image.img_to_array(test_image)
test_image_final = np.expand_dims(test_image, axis = 0)

print(res_list[int(loaded_model.predict(test_image))])
import tensorflow as tf  
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(6, activation=tf.nn.softmax)
])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(training_set, test_set, batch_size=128, epochs=20, validation_split = 0.2)
# testing the model

def testing_image(image_directory):
    model = joblib.load('finalized_model.sav')
    test_image = image.load_img(image_directory, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    print(result)
    if result[0][0]  == 1:
        prediction = 'Dog'
    else:
        prediction = 'Cat'
    return prediction
print(testing_image('/kaggle/input/dogs-vs-cats/train/dogs/dog.1.jpg'))