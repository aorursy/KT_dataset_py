#Download Dependencies
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import zipfile
import glob
import os
import torch as th
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from kaggle_datasets import KaggleDatasets

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.resnet50 import preprocess_input
import zipfile
import glob

zip_file = glob.glob('/kaggle/input/dogs-vs-cats/*.zip')  #return any files with .zip extension
print(zip_file)

#extract file into a temp folder
def extract_zip(file):
    with zipfile.ZipFile(file,"r") as zip_ref:
        zip_ref.extractall("temp")
        
#extract both train and test1 zip
for files in zip_file:
    extract_zip(files)
#instantiate the constants
batch_size = 16
img_size = 224
epochs = 5
print(len(os.listdir('/kaggle/working/temp/train')), "training data")
print(len(os.listdir('/kaggle/working/temp/test1')), "test data")
os.listdir("temp")
def gen_label(directory):
    label = []
    for file in os.listdir(directory):
        if (file.split('.')[0] == 'dog'):
            label.append(str(1))
        elif (file.split('.')[0] == 'cat'):
            label.append(str(0))
    return label
    #print(len(label),"files in", directory)
    
def get_path(directory):
    path = []
    for files in os.listdir(directory):
        path.append(files)
    return path

train_y = gen_label('temp/train')
train_x = get_path('temp/train')
test_x = get_path('temp/test1')
df = pd.DataFrame({'filename': train_x,
                  'category': train_y})
print(df.head())

sns.countplot(x='category',data=df).set_title("Data Distribution")
# Change working directory
os.chdir('temp/train')

img = load_img(df['filename'].iloc[0]) 
  
# Displaying the image 
plt.figure(figsize=(8,8))
plt.imshow(img)
train_df, valid_df = train_test_split(df, test_size=0.25)
print(train_df.shape)
print(valid_df.shape)
def generate_train_batch(model):
    
    if model == 'resnet':      #use of resnet requires its specific preprocessing_function for better accuracy for augmentation
        print('resnet data')
        train_datagen = ImageDataGenerator(
                    rotation_range=10,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    preprocessing_function = preprocess_input)

    else:
        train_datagen = ImageDataGenerator(    #standard augmentation
                    rotation_range=10,
                    rescale=1./255,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    width_shift_range=0.1,
                    height_shift_range=0.1)

    if model == 'vgg':   #VGG16 will only generate mini-batches of x_features; y_col=None as feature extractor
        print('vgg data')
        train_gen = train_datagen.flow_from_dataframe(
            train_df[['filename']],
            x_col='filename',
            y_col=None,
            target_size=(img_size, img_size),
            batch_size = batch_size,
            class_mode=None,
            shuffle=False)
        
    else:
        train_gen = train_datagen.flow_from_dataframe(
                    train_df,
                    x_col='filename',
                    y_col='category',
                    target_size=(img_size, img_size),
                    batch_size = batch_size,
                    class_mode='binary')

    return train_gen


def generate_valid_batch(model):
    if model == 'resnet':
        print('resnet validation set')
        valid_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
    else:
        valid_datagen = ImageDataGenerator(rescale=1./255)
        
    valid_gen = valid_datagen.flow_from_dataframe(
            valid_df,
            x_col='filename',
            y_col='category',
            target_size=(img_size, img_size),
            batch_size = batch_size,
            class_mode='binary')
    
    return valid_gen

train_gen = generate_train_batch('others')
valid_gen = generate_valid_batch('others')
visual_datagen = ImageDataGenerator(    #standard augmentation
                    rotation_range=10,
                    rescale=1./255,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    width_shift_range=0.1,
                    height_shift_range=0.1)

visualise_df = train_df.sample(n=1).reset_index(drop=True)
visualisation_generator = visual_datagen.flow_from_dataframe(
    visualise_df,  
    x_col='filename',
    y_col='category'
)
plt.figure(figsize=(8, 8))
for i in range(0, 9):
    plt.subplot(3, 3, i+1)
    for X_batch, Y_batch in visualisation_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()
from keras.applications import VGG16, resnet
from keras.layers import *
from keras.models import Model,Sequential
from keras import optimizers
from keras import regularizers
from keras import backend as K
K.clear_session()
#model instantiation
modelcnn=Sequential()
modelcnn.add(Conv2D(16, (3,3), activation="relu", input_shape=(img_size, img_size, 3)))
modelcnn.add(Conv2D(16, (3,3), activation="relu",))
modelcnn.add(MaxPooling2D((3,3)))

modelcnn.add(Conv2D(32, (3,3), activation="relu"))
modelcnn.add(Conv2D(32, (3,3), activation="relu"))
modelcnn.add(MaxPooling2D(2,2))

modelcnn.add(Conv2D(64, (3,3), activation="relu"))
modelcnn.add(Conv2D(64, (3,3), activation="relu"))
modelcnn.add(MaxPooling2D(2,2))
modelcnn.add(Dropout(0.3))

modelcnn.add(Conv2D(32, (3,3), activation="relu"))
modelcnn.add(MaxPooling2D((2,2)))

modelcnn.add(Flatten())
modelcnn.add(Dense(512, activation="relu"))
modelcnn.add(Dropout(0.5))
modelcnn.add(Dense(1, activation="sigmoid"))

modelcnn.compile(loss="binary_crossentropy", 
         optimizer=optimizers.RMSprop(lr=1e-4),
         metrics=["accuracy"])
modelcnn.summary()
modelcnn.fit_generator(train_gen,
                    epochs=epochs,
                    validation_data=valid_gen)
loss, accuracy = modelcnn.evaluate_generator(valid_gen, valid_gen.samples//batch_size, workers=12)
print("Validation: accuracy = %f  ;  loss = %f " % (accuracy, loss))
vgg = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

for layers in vgg.layers:
    layers.trainable=False

print(vgg.output)
feature_list = []
for path in train_df['filename'].to_numpy():
    x = load_img(path,target_size=(img_size,img_size))
    img_array = img_to_array(x)
    img_array = np.expand_dims(img_array, axis=0)
    features = vgg.predict(img_array)
    feature_list.append(features)
    
feat_lst = np.reshape(feature_list,(-1,7*7*512))
del feature_list
print(feat_lst.shape)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

y = train_df['category'].to_numpy()  #convert df to numpy array with shape(18750,)

X_train, X_test, y_train, y_test = train_test_split(feat_lst, y, test_size=0.2, random_state=2020)

glm = LogisticRegression(C=0.1)
glm.fit(X_train,y_train)
print("Accuracy on validation set using Logistic Regression: ",glm.score(X_test,y_test))
np.random.seed(2020)

res = resnet.ResNet50(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

res_train_gen = generate_train_batch('resnet')
res_valid_gen = generate_valid_batch('resnet')


for layer in res.layers[:171]:
    layer.trainable=False
    

flat = Flatten()(res.output)   #Flatten the output layer from our Resnet model
dense = Dense(1024,activation='relu')(flat)
drop = Dropout(0.5)(dense)
classifier = Dense(1, activation='sigmoid')(drop)


res_model = Model(res.input, classifier)
optimizer=optimizers.Adam(1e-5)


res_model.compile(optimizer= optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

res_model.fit_generator(res_train_gen,
                    epochs=epochs,
                    validation_data=res_valid_gen,
                    validation_steps=res_train_gen.samples//batch_size,
                    steps_per_epoch = res_valid_gen.samples//batch_size)
loss, accuracy = res_model.evaluate_generator(res_valid_gen, res_valid_gen.samples//batch_size, workers=12)
print("Validation: accuracy = %f  ;  loss = %f " % (accuracy, loss))
# define function for evaluating model performance on test images

testdf = pd.DataFrame({'filename': test_x})
test_sample = testdf.sample(n=12, random_state=2020)

def test_img(model,name):
    result_lst = []
    for path in test_sample['filename'].to_numpy():
        full_path = '../test1/'+path
        x = load_img(full_path, target_size=(224,224))
        img_array = img_to_array(x)
        img_array = np.expand_dims(img_array, axis=0)
        if name == 'vgg':
            features = model.predict(img_array)
            features = np.reshape(features,(-1,7*7*512))
            result = glm.predict(features)
        else:
            result =  model.predict(img_array)
        
        result = 'dog' if float(result) >0.5 else 'cat'
        
        result_lst.append(result)
    return result_lst
# get test predictions from all models
custom_cnn_result = test_img(modelcnn, 'cnn')
trflearn_result = test_img(vgg,'vgg')
finetune_result = test_img(res_model,'resnet')
# plotting images with prediction
pred_results  = list(zip(custom_cnn_result,trflearn_result,finetune_result))
test_array = test_sample['filename'].to_numpy()

plt.figure(figsize=(15, 15))
for i in range(0, 12):
    plt.subplot(4, 3, i+1)
    cust,tf,ft = pred_results[i]
    img = test_array[i]
    path = '../test1/' + img
    image = load_img(path, target_size=(256,256))
    plt.text(135, 200, 'Custom CNN: {}'.format(cust), color='lightgreen',fontsize= 11, bbox=dict(facecolor='black', alpha=0.9))
    plt.text(135, 225, 'Transfer Learn: {}'.format(tf), color='lightgreen',fontsize= 11, bbox=dict(facecolor='black', alpha=0.9))
    plt.text(135, 250, 'Fine Tune: {}'.format(ft), color='lightgreen',fontsize= 11, bbox=dict(facecolor='black', alpha=0.9))
    plt.imshow(image)

plt.tight_layout()
plt.show()
# test_generator = ImageDataGenerator(rescale=1./255)
# test_gen = test_generator.flow_from_dataframe(
#     testdf,
#     '../test1',
#     x_col='filename',
#     y_col=None,
#     class_mode=None,
#     batch_size=batch_size,
#     target_size=(img_size, img_size),
#     shuffle=False
# )
# predict = model.predict_generator(test_gen, steps=test_gen.samples/batch_size)
# threshold = 0.5
# testdf['category'] = np.where(predict > threshold, 1,0)

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os
class CatDogData(Dataset):
    def __init__(self, direct, df, transform):
        self.dir = direct
        self.df = df
        self.transform = transform
        
    def __getitem__(self, idx):
        img, label = self.df.iloc[idx]
        path = os.path.join(self.dir, img)
        img_array = Image.open(path)
        
        if self.transform:
            img_array = self.transform(img_array)
        label = torch.as_tensor(int(label))
        
        return img_array, label
    
    def __len__(self):
        return self.df.shape[0]
from torchvision import transforms

train_trf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

valid_trf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
])
train_data = CatDogData('/kaggle/working/temp/train', train_df, train_trf)
valid_data = CatDogData('/kaggle/working/temp/train', valid_df, valid_trf)

train_loader = DataLoader(train_data, batch_size = 32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
from torchvision.models import resnet50
import torch.nn as nn
model = resnet50(pretrained=True)
!pip install torchsummary
c = 0
for child in model.children():
    if c < 9:
        for param in child.parameters():
            param.requires_grad = False
    c+=1
output_layer = nn.Linear(1000, 2)
model = nn.Sequential(model, output_layer)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0
import tqdm
class Fitter:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
        param_list = [p for p in model.parameters() if p.requires_grad]
    
        self.optimizer = torch.optim.Adam(param_list, lr = 1e-5)
        
        self.criterion = nn.CrossEntropyLoss()
        
    def fit(self, model, train_loader, valid_loader):
        bst_loss = 10
        print(f'Model running with {self.device}')
        for epoch in range(5):
            self.train_one_epoch(model, train_loader)
            val_loss = self.validate(model, valid_loader)
            
            print("Previous loss: {} Current loss: {}".format(bst_loss, val_loss))
            if val_loss < bst_loss:
                model.eval()
                torch.save(model, f'{epoch}.pth')
                bst_loss = val_loss
            
    def train_one_epoch(self, model, train_loader):
        itr = 1
        loss_hist = Averager()
        self.model.train()
        
        for images, targets in train_loader:
            images = images.to(self.device).float()
            targets = targets.to(self.device)

            output = self.model(images)
            loss = self.criterion(output, targets)
            loss_hist.send(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                
            if itr % 100 == 0:
                print(f"Iteration #{itr} loss: {loss.item()}")


            itr += 1
                
        print(f"Training Epoch loss: {loss_hist.value}")
        loss_hist.reset()
        
        
    def validate(self, model, valid_loader):
        acc = 0
        val_loss = 0
        loss_hist = Averager()
        self.model.eval()
        
        for images, targets in valid_loader:
            with torch.no_grad():
                images = images.to(self.device).float()
                targets = targets.to(self.device)

                output = self.model(images)
                loss = self.criterion(output, targets)
                loss_hist.send(loss.item())

                #acc
                log = torch.sigmoid(output)
                pred = torch.argmax(log, 1)
                acc += (targets.cpu() == pred.cpu()).sum().item()
                

        val_loss = loss_hist.value    
        print(f"Validation Epoch loss: {loss_hist.value}")
        loss_hist.reset()
        print("Validation Accuracy: {}".format(acc/len(valid_data)))
        
        return val_loss
run_model = Fitter(model, device)
run_model.fit(model, train_loader, valid_loader)

