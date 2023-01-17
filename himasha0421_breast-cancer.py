!pip install imutils
#set the data set folder
import os
#original input data set path
ORIGINAL_INPUT_DATASET = "../input/breast-histopathology-images"
#master folder to contain the train test valid data 
BASE_PATH = "/kaggle/working/datasets"
#derive the training testing and validation directories
TRAIN_PATH = os.path.sep.join([BASE_PATH , 'training'])
VAL_PATH = os.path.sep.join([BASE_PATH , 'validation'])
TEST_PATH = os.path.sep.join([BASE_PATH , "testing"])
#Train test split
TRAIN_SPLIT = 0.8
#validation split
VAL_SPLIT = 0.1
!ls /kaggle/working/
from imutils import paths
import shutil
import random
import os

#grab the paths to the input images in the base folder
imgPaths = list(paths.list_images(ORIGINAL_INPUT_DATASET))
#define a random seed to shuffle
random.seed(42)
random.shuffle(imgPaths)
print(len(imgPaths))
from PIL import Image
%matplotlib inline
import matplotlib.pyplot as plt
#do some visualization
test_img_path = imgPaths[5]
class_label = test_img_path.split(os.path.sep)[-2]
print("Class Label " ,class_label)
image = Image.open(test_img_path)
plt.figure(figsize=(5,5))
plt.imshow(image)
plt.show()
#take 80% of the data from the folder
imgPaths = imgPaths[:int(len(imgPaths)*0.7)]
#define the split index for train  test
split_idx = int(len(imgPaths)*TRAIN_SPLIT)
trainPaths= imgPaths[:split_idx]
testPaths = imgPaths[split_idx:]

#define the train val split
val_split = int(len(trainPaths)*VAL_SPLIT)
valPaths = trainPaths[:val_split]
trainPaths = trainPaths[val_split:]

#define the datasets
datasets=[
    ('training',trainPaths , TRAIN_PATH),
    ('validation' , valPaths , VAL_PATH),
    ('testing' , testPaths , TEST_PATH)
]

for (dtype , imgPaths , baseoutput) in datasets:
    print("Building the Dataset for ",dtype)
    #if the base output is not exists then create a folder
    if not os.path.exists(baseoutput):
        print("Create a directory for the dataset ", baseoutput)
        os.makedirs(baseoutput)
    
    for imgPath in imgPaths :
        #define the class and the file name
        filename = imgPath.split(os.path.sep)[-1]
        label = filename[-5:-4]
        #build the path to the label directory for 1 and 0
        labelPath = os.path.sep.join([baseoutput,label])
        
        if not os.path.exists(labelPath):
            print("Create the directory for the label {}".format(labelPath))
            os.makedirs(labelPath)        
        p = os.path.sep.join([labelPath,filename])
     
        shutil.copy2(imgPath , p)
!ls /kaggle/working
from torchvision import datasets
import torch
from torchvision import transforms
def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight 

transform_train = transforms.Compose([
    transforms.Pad(64, padding_mode='reflect'),
    transforms.RandomRotation((0,10)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor() ,
    transforms.Normalize([0.5,0.5,0.5] , [0.5,0.5,0.5])
])
transform_test = transforms.Compose([
    transforms.Pad(64, padding_mode='reflect'),
    transforms.ToTensor() ,
    transforms.Normalize([0.5,0.5,0.5] , [0.5,0.5,0.5])
])
transform_valid = transforms.Compose([
    transforms.Pad(64, padding_mode='reflect'),
    transforms.ToTensor() ,
    transforms.Normalize([0.5,0.5,0.5] , [0.5,0.5,0.5])
])


dataset_train = datasets.ImageFolder('/kaggle/working/datasets/training' ,transform=transform_train)                                                                         
dataset_test = datasets.ImageFolder("/kaggle/working/datasets/testing",transform=transform_test)   
dataset_valid = datasets.ImageFolder("/kaggle/working/datasets/validation",transform=transform_valid)

# For unbalanced dataset we create a weighted sampler                       
weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))                                                                
weights = torch.DoubleTensor(weights)                                       
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
                                                                                
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=64 , sampler = sampler, num_workers=0, pin_memory=True)   
test_loader = torch.utils.data.DataLoader(dataset_test , batch_size=32 , pin_memory=True)
valid_loader = torch.utils.data.DataLoader(dataset_valid , batch_size=32 , pin_memory=True)
import torch.nn as nn
import torch.nn.functional as F
#define the CNN model
class Cancer_Net(nn.Module):
    def __init__(self , input_shape , output_shape , seed):
        super(Cancer_Net , self).__init__()
    
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape , out_channels=32 , kernel_size=3 , stride=1 , padding=1),
            nn.ReLU() ,
            nn.BatchNorm2d(32) ,
            #nn.Conv2d(32 , 32 , kernel_size=3 ,stride=1 , padding=1) ,            
            #nn.ReLU() , 
            #nn.BatchNorm2d(32) , 
            nn.MaxPool2d(kernel_size=(2,2) , stride=(2,2)) , 
            #nn.Dropout(p=0.2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32 , out_channels=64 , kernel_size=3 , stride=1 , padding=1),
            nn.ReLU() ,
            nn.BatchNorm2d(64) ,
            #nn.Conv2d(64 , 64 , kernel_size=3 ,stride=1 , padding=1) ,            
            #nn.ReLU() , 
            #nn.BatchNorm2d(64) , 
            nn.MaxPool2d(kernel_size=(2,2) , stride=(2,2)) , 
            #nn.Dropout(p=0.2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64 , out_channels=128 , kernel_size=3 , stride=1 , padding=1),
            nn.ReLU() ,
            nn.BatchNorm2d(128) ,
            #nn.Conv2d(128 , 128 , kernel_size=3 ,stride=1 , padding=1) ,            
            #nn.ReLU() , 
            #nn.BatchNorm2d(128) , 
            #nn.Conv2d(128 , 128 , kernel_size=3 ,stride=1 , padding=1) ,            
            #nn.ReLU() , 
            #nn.BatchNorm2d(128) , 
            nn.MaxPool2d(kernel_size=(2,2) , stride=(2,2)) , 
            #nn.Dropout(p=0.2)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=256 , out_channels=512 , kernel_size=3 , stride=1 , padding=1),
            nn.ReLU(inplace=True) ,
            nn.BatchNorm2d(512) ,
            #nn.Conv2d(512 , 512 , kernel_size=3 ,stride=1 , padding=1) ,            
            #nn.ReLU(inplace=True) , 
            #nn.BatchNorm2d(512) , 
            nn.MaxPool2d(kernel_size=(2,2) , stride=(2,2)) , 
            #nn.Dropout(p=0.5)
        )
        self.size = 512*4*4
        self.fc_block = nn.Sequential(
            nn.Linear(self.size ,128) ,
            nn.ReLU() , 
            nn.Dropout(p=0.5) ,
            nn.Linear(128,1) , 
            nn.Sigmoid()
        )
        
    def forward(self, x):
        
        #forward pass on the conv layers
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.view(-1,self.size)
        x = self.fc_block(x)
        #since the binary classificcation
        return x

cancer_net = Cancer_Net(3 , 1 , 0)       
import torch
device='cuda:0' if torch.cuda.is_available() else 'cpu'
cancer_net.to(device)
import torch.optim as optim
end_lr = 1e-6
start_lr = 0.001
#define the loss function and the optimizer use the binary cross entropy loss
criterion = nn.BCELoss()
#lets use the gpu and move data to gpu
#criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.SGD(cancer_net.parameters(), start_lr)
def get_lr_search_scheduler(optimizer, min_lr, max_lr, max_iterations):
    # max_iterations should be the number of steps within num_epochs_*epoch_iterations
    # this way the learning rate increases linearily within the period num_epochs*epoch_iterations 
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, 
                                               base_lr=min_lr,
                                               max_lr=max_lr,
                                               step_size_up=max_iterations,
                                               step_size_down=max_iterations,
                                               mode="triangular")
    
    return scheduler
lr_find_epochs=1
scheduler = get_lr_search_scheduler(optimizer, start_lr, end_lr, lr_find_epochs*len(train_loader))
from collections import deque
import numpy as np
EPOCHS = 10
total_score = []
mean_loss = deque(maxlen=500)
for epoch in range(EPOCHS):
    
    epoch_loss =0
    epoch_val_loss = 0
    epoch_accuracy = 0
    for idx , (data , label) in enumerate(train_loader) :
        data , label = data.to(device) , label.view(-1,1).to(device).float()
        output = cancer_net(data)
        loss = criterion(output , label)
        epoch_loss += loss.to('cpu').detach().numpy()
        mean_loss.append(loss.to('cpu').detach().item())
        #reset the optimizer
        optimizer.zero_grad()
        #backprop the loss
        loss.backward()
        #optimize the model
        optimizer.step()
        if((idx+1)%500 == 0 ):
            print("Epoch : {} Mean Train loss : {} ".format(epoch , np.mean(mean_loss)))
        total_score.append(loss.to('cpu').detach().numpy())
    cancer_net.eval()
    with torch.no_grad():
        for data , label in valid_loader :
            data , label = data.to(device) , label.to(device)
            output = cancer_net(data)
            top_prob , top_k = torch.topk(output , 1)
            eqauls = (label == top_k.view(label.shape))
            accuracy = torch.mean(eqauls.type(torch.FloatTensor))
            epoch_accuracy += accuracy.to('cpu').detach().numpy()
    cancer_net.train()
    scheduler.step()  
    
    print("Epoch : {} Total loss : {} Total Accuracy : {}".format(epoch ,epoch_loss/len(train_loader) , epoch_accuracy/len(valid_loader)))
    
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SeparableConv2D , Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
class CancerNet :
    @staticmethod
    def build(width , height , depth , classes):
        model= Sequential()
        inputShape = (height , width , depth)
        channel_dim = -1
        
        if K.image_data_format()=="channels_first":
            inputShape = (depth , height , width)
            channel_dim =1
            
        #using the seprable convs getting higher computation but the kernels not getting much higher accuracy    
        model.add(Conv2D(32 , (3,3) , padding='same' , input_shape=inputShape , 
                         kernel_regularizer = regularizers.l1_l2(l1=1e-2 , l2=1e-3)))
        
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64 , (3,3) , padding='same' , 
                                 kernel_regularizer = regularizers.l1_l2(l1=1e-3)))
        
        
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(Conv2D(128 , (3,3) , padding='same' , 
                                 kernel_regularizer = regularizers.l1_l2(l1=1e-3)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(256 , (3,3) , padding='same' , 
                                 kernel_regularizer = regularizers.l1_l2(l1=1e-3)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(Conv2D(512 , (3,3) , padding='same' , 
                                 kernel_regularizer = regularizers.l1_l2(l1=1e-3)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(Conv2D(1024 , (3,3) , padding='same' , 
                                 kernel_regularizer = regularizers.l1_l2(l1=1e-3)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25)) 
        
        #add the dense layers
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        
        return model
#implement the training scipt
import matplotlib
matplotlib.use("Agg")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.utils  import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import keras
NUM_EPOCHS = 40
INIT_LR = 1e-2
BS = 32

trainPaths = list(paths.list_images(TRAIN_PATH))
totalTrain = len(trainPaths)
totalVal = len(list(paths.list_images(VAL_PATH)))
totalTest = len(list(paths.list_images(TEST_PATH)))

trainLabels = [int(p.split(os.path.sep)[-2])  for p in trainPaths]
trainLabels = to_categorical(trainLabels)
classTotals = trainLabels.sum(axis=0)
classWeight = dict()

#for all the classes calculate the weight
for i in range(0,len(classTotals)):
    classWeight[i] = classTotals.max()/classTotals[i]
!pip install keras --upgrade
import tensorflow as tf
with tf.device("gpu:0"):
   print("tf.keras code in this scope will run on GPU")
# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=20,
	zoom_range=0.05,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.05,
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode="nearest")
# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)
#define the training generators
trainGen = trainAug.flow_from_directory(
    TRAIN_PATH ,
    class_mode = 'categorical',
    target_size=(48,48) ,
    color_mode = 'rgb',
    shuffle=True ,
    batch_size=BS
)

# initialize the validation generator
valGen = valAug.flow_from_directory(
	VAL_PATH,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)
# initialize the testing generator
testGen = valAug.flow_from_directory(
	TEST_PATH,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)
model = CancerNet.build(width=48 , height=48 , depth=3 ,classes=2)
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
opt = keras.optimizers.SGD(learning_rate=lr_schedule)
model.compile(loss='binary_crossentropy', optimizer=opt , metrics=['accuracy'])

H = model.fit(
    x = trainGen , 
    steps_per_epoch = totalTrain // BS ,
    validation_data = valGen ,
    validation_steps = totalVal//BS ,
    class_weight = classWeight ,
    epochs = 10 )


print("Model ecavluation ")
testGen.reset()
predIdx = model.predict(x = testGen , steps=(totaltest//BS)+1)

predIdx = np.argmax(predIdx , axis=1)
print(classification_report(testGen.classes , predIdx , target_names=testGen.class_indices.keys()))
cm = confusion_matrix(testGen.classes , predIdx)
print(cm)
