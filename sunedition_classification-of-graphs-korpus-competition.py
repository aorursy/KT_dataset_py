# get dataset from google disk
!conda install -y gdown
!gdown https://drive.google.com/uc?id=1MyLFn80PnqOO53rU1jP9zpdVai_oHMzY

# install skorch wrapper for pytorch
!pip install -U skorch
# import libraries for file processing
import os
import zipfile

# unzip file to /kaggle/working folder
with zipfile.ZipFile('./cv_train.zip', 'r') as zip_ref:
    zip_ref.extractall('./cv_train')  

# delete unnecessary file
for dump in os.listdir('./cv_train/cv_train'):
    if dump.split('.')[::-1][0] == 'ini':
        os.remove('./cv_train/cv_train/' + dump)
        print(f'{dump} removed')
# libraries for image processing
import cv2
import numpy as np
import pandas as pd

# pytorch libraries
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

# skorch wrapper classes and functions
from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit
from skorch.callbacks import LRScheduler, Checkpoint 
from skorch.callbacks import Freezer, EarlyStopping

# sklearn models for further stacking
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

# albumentations for image augmentation
import albumentations
from albumentations import pytorch

# for multiprocessing
import multiprocessing as mp

# plot graphs for data exploration
import matplotlib.pyplot as plt
# Here we seed our environmental variables and pytorch variables
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    seed_everything(42)
# Read train dataset from file
train_dir = './cv_train/cv_train'        
train_files = os.listdir(train_dir)

# Read test dataset from file
test_dir = '/kaggle/input/korpus-ml-2/test/test/test'
test_files = os.listdir(test_dir)

print(len(train_files), len(test_files))
# class for dataset loading, labeling and augmentation
class Graphs(Dataset):
    def __init__(self, dir_path, file_list, transform=None, mode='train'):
        self.dir_path = dir_path
        self.file_list = file_list
        self.transform = transform
        self.mode = mode
        self.label_dict = {'just_image' : 0, 'bar_chart' : 1, 'diagram' : 2, 
                           'flow_chart' : 3, 'graph' : 4, 'growth_chart' : 5,
                           'pie_chart' : 6, 'table' : 7}

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        image_name = os.path.join(self.dir_path, self.file_list[idx])
        
        if image_name.split('.')[::-1][0] == "gif":
            gif = cv2.VideoCapture(image_name)
            _, image = gif.read()
        else:
            image = cv2.imread(image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for name, label in self.label_dict.items():
            if name in image_name:
                self.label = label
                break

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        if self.mode == 'train':
            return image, self.label
        else:
            return image, image_name
# declaration of constant variables
batch_size = 128
num_workers = mp.cpu_count()
img_size = 224
n_classes = 8
# function that prepares dataset for further training
def prepare_datasets(train_dir, train_files, test_dir, test_files):
    # augmentation parameters for train
    data_transforms_train = albumentations.Compose([
        albumentations.Resize(img_size, img_size),
        albumentations.HorizontalFlip(),
        albumentations.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.10,
                                        rotate_limit=15),
        albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        pytorch.ToTensor()
    ]) 
    # augmentation parameters for test
    data_transforms_test = albumentations.Compose([
        albumentations.Resize(img_size, img_size),
        albumentations.HorizontalFlip(),
        albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        pytorch.ToTensor()
        ])

    trainset = Graphs(train_dir, train_files, transform=data_transforms_train)
    testset = Graphs(test_dir, test_files, transform=data_transforms_test,
                     mode='test')
    
    print(f'Train dataset length: {len(trainset)}')
    print(f'Testset dataset length: {len(testset)}')
    
    return trainset, testset
# get prepared datasets from files stored in directory
train_set, test_set = prepare_datasets(train_dir, train_files, test_dir, test_files)

# create dataloaders for loading data in batches=128
trainloader = DataLoader(train_set, batch_size=batch_size,
                         pin_memory=True, num_workers=num_workers, shuffle=True)
testloader = DataLoader(test_set, batch_size=batch_size,
                        pin_memory=True, num_workers=num_workers)
# plot bar chart to observe amount of images by classes in dataset
class_names = ['just_image', 'bar_chart', 'diagram', 'flow_chart', 'graph',
               'growth_chart', 'pie_chart', 'table']

train_images = np.array([X.size() for X, y in iter(train_set)])
train_labels = np.array([y for X, y in iter(train_set)])

_, train_counts = np.unique(train_labels, return_counts=True)
pd.DataFrame({'train': train_counts}, index=class_names).plot.bar()
plt.show()
# plot pie chart to observe proportion of classes in dataset
plt.pie(train_counts, explode=(0, 0, 0, 0, 0, 0, 0, 0) , 
        labels=class_names, autopct='%1.1f%%')
plt.axis('equal')
plt.title('Proportion of each observed category')
plt.show()
# print images with labels
samples, labels = next(iter(trainloader))

fig = plt.figure(figsize=(16, 16))
fig.suptitle("Some examples of images of the dataset", fontsize=16)
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(np.transpose(samples[i], (1, 2, 0)), cmap=plt.cm.binary)
    plt.xlabel(class_names[labels[i]])
plt.show()
# class which uses ResNet50 pretrained model
# + added custom classifier in the last layer
class ResNet50(nn.Module):
    def __init__(self, output_features, num_units=512, drop=0.3337,
                 num_units1=256, drop1=0.1):
        super().__init__()
        model = torchvision.models.resnet50(pretrained=True)
        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
                                nn.Linear(n_inputs, num_units),
                                nn.ReLU(),
                                nn.Dropout(p=drop),
                                nn.Linear(num_units, num_units1),
                                nn.ReLU(),
                                nn.Dropout(p=drop1), 
                                nn.Linear(num_units1, output_features))
        self.model = model
        
    def forward(self, x):
        return self.model(x)

# class which uses DenseNet169 pretrained model
# + added custom classifier in the last layer
class DenseNet169(nn.Module):
    def __init__(self, output_features, num_units=512, drop=0.3337,
                 num_units1=256, drop1=0.1):
        super().__init__()
        model = torchvision.models.densenet169(pretrained=True)
        n_inputs = model.classifier.in_features
        model.classifier = nn.Sequential(
                                nn.Linear(n_inputs, num_units),
                                nn.ReLU(),
                                nn.Dropout(p=drop),
                                nn.Linear(num_units, num_units1),
                                nn.ReLU(),
                                nn.Dropout(p=drop1), 
                                nn.Linear(num_units1, output_features))
        self.model = model
        
    def forward(self, x):
        return self.model(x)

# class which uses VGG16 pretrained model
class VGG16(nn.Module):
    def __init__(self, output_features):
        super().__init__()
        model = torchvision.models.vgg16(pretrained=True)
        n_inputs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(n_inputs, output_features)
        self.model = model
        
    def forward(self, x):
        return self.model(x)
    
class EfficientNet_b3(nn.Module):
    def __init__(self, output_features):
        super().__init__()
        model = EfficientNet.from_pretrained('efficientnet-b3')
        n_inputs = model._fc.in_features
        model._fc = nn.Linear(n_inputs, output_features)
        self.model = model
        
    def forward(self, x):
        return self.model(x)
# callback functions for models

# ResNet50
# callback for Reduce on Plateau scheduler 
lr_scheduler_resnet = LRScheduler(policy='ReduceLROnPlateau',
                                  factor=0.5, patience=1)
# callback for saving the best on validation accuracy model
checkpoint_resnet = Checkpoint(f_params='best_model_resnet50.pkl',
                               monitor='valid_acc_best')
# callback for freezing all layer of the model except the last layer
freezer_resnet = Freezer(lambda x: not x.startswith('model.fc'))
# callback for early stopping
early_stopping_resnet = EarlyStopping(patience=10)

# DenseNet169
# callback for Reduce on Plateau scheduler 
lr_scheduler_densenet = LRScheduler(policy='ReduceLROnPlateau',
                                    factor=0.5, patience=1)
# callback for saving the best on validation accuracy model
checkpoint_densenet = Checkpoint(f_params='best_model_densenet169.pkl',
                                 monitor='valid_acc_best')
# callback for freezing all layer of the model except the last layer
freezer_densenet = Freezer(lambda x: not x.startswith('model.classifier'))
# callback for early stopping
early_stopping_densenet = EarlyStopping(patience=10)

# VGG16
# callback for Reduce on Plateau scheduler 
lr_scheduler_vgg = LRScheduler(policy='ReduceLROnPlateau',
                               factor=0.5, patience=1)
# callback for saving the best on validation accuracy model
checkpoint_vgg = Checkpoint(f_params='best_model_vgg16.pkl',
                            monitor='valid_acc_best')
# callback for freezing all layer of the model except the last layer
freezer_vgg = Freezer(lambda x: not x.startswith('model.classifier'))
# callback for early stopping
early_stopping_vgg = EarlyStopping(patience=10)
# NeuralNetClassifier for based on ResNet50 with custom parameters
resnet = NeuralNetClassifier(
    # pretrained ResNet50 + custom classifier 
    module=ResNet50,          
    # fine tuning model's inner parameters
    module__output_features=n_classes,
    module__num_units=512,
    module__drop=0.5,
    module__num_units1=512,
    module__drop1=0.5,
    # criterion
    criterion=nn.CrossEntropyLoss,
    # batch_size = 128
    batch_size=batch_size,
    # number of epochs to train
    max_epochs=100,
    # optimizer Adam used
    optimizer=torch.optim.Adam,
    optimizer__lr = 0.001,
    optimizer__weight_decay=1e-6,
    # shuffle dataset while loading
    iterator_train__shuffle=True,
    # load in parallel
    iterator_train__num_workers=num_workers,
    # stratified kfold split of loaded dataset
    train_split=CVSplit(cv=5, stratified=True, random_state=42),
    # callbacks declared earlier
    callbacks=[lr_scheduler_resnet, checkpoint_resnet, 
               freezer_resnet, early_stopping_resnet],
    # use GPU or CPU
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)
# NeuralNetClassifier for based on DenseNet169 with custom parameters
densenet = NeuralNetClassifier(
    # pretrained DenseNet169 + custom classifier 
    module=DenseNet169, 
    # fine tuning model's inner parameters
    module__output_features=n_classes,
    module__num_units=512,
    module__drop=0.5,
    module__num_units1=512,
    module__drop1=0.5,
    # criterion
    criterion=nn.CrossEntropyLoss,
    # batch_size = 128
    batch_size=batch_size,
    # number of epochs to train
    max_epochs=100,
    # optimizer Adam used
    optimizer=torch.optim.Adam,
    optimizer__lr = 0.001,
    optimizer__weight_decay=1e-6,
    # shuffle dataset while loading
    iterator_train__shuffle=True,
    # load in parallel
    iterator_train__num_workers=num_workers,
    # stratified kfold split of loaded dataset
    train_split=CVSplit(cv=5, stratified=True, random_state=42),
    # callbacks declared earlier
    callbacks=[lr_scheduler_densenet, checkpoint_densenet, 
               freezer_densenet, early_stopping_densenet],
    # use GPU or CPU
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)
# NeuralNetClassifier for based on VGG16 with custom parameters
vgg = NeuralNetClassifier(
    # pretrained VGG16
    module=VGG16,
    # fine tuning model's inner parameters
    module__output_features=n_classes, 
    # criterion
    criterion=nn.CrossEntropyLoss,
    # batch_size = 128
    batch_size=batch_size,
    # number of epochs to train
    max_epochs=100,
    # optimizer Adam used
    optimizer=torch.optim.Adam,
    optimizer__lr = 0.001,
    optimizer__weight_decay=1e-6,
    # shuffle dataset while loading
    iterator_train__shuffle=True,
    # load in parallel
    iterator_train__num_workers=num_workers, 
    # stratified kfold split of loaded dataset
    train_split=CVSplit(cv=5, stratified=True, random_state=42),
    # callbacks declared earlier
    callbacks=[lr_scheduler_vgg, checkpoint_vgg,
               freezer_vgg, early_stopping_vgg],
    # use GPU or CPU
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)
# Load y_train labels for training in Skorch 
# (labels in trainset are ignored while training)
y_train = np.array([y for X, y in iter(train_set)])
# fit prepared model with custom parameters

# resnet.fit(train_set, y=y_train)
# densenet.fit(train_set, y=y_train)
# vgg16.fit(train_set, y=y_train)
# ResNet50
resnet.initialize()
resnet.load_params(f_params='/kaggle/input/best-models/best_model_resnet50.pkl')

# DenseNet169
densenet.initialize()
densenet.load_params(f_params='/kaggle/input/best-models/best_model_densenet169.pkl')

# VGG16
vgg.initialize()
vgg.load_params(f_params='/kaggle/input/best-models/best_model_vgg16.pkl')

# ensemble
models = [resnet, densenet, vgg]
# create validation dataset (30%) by splitting train dataset (70%)
valid_size = int(len(train_files) * 0.3)
trainset, validset = random_split(train_set, 
                                  (len(train_files)-valid_size, valid_size))

# create dataloader for loading data in batches=128
validloader = DataLoader(validset, batch_size=batch_size, pin_memory=True)
# predict on validation set ResNet50
pred_resnet = np.array([])
for batch_idx, (X_test, labels) in enumerate(validloader):
    pred_resnet = np.append(pred_resnet, resnet.predict(X_test).tolist())
print('ResNet50 prediction done!')
    
# predict on validation set DenseNet169
pred_densenet = np.array([])
for batch_idx, (X_test, labels) in enumerate(validloader):
    pred_densenet = np.append(pred_densenet, densenet.predict(X_test).tolist())
print('DenseNet169 prediction done!')

# predict on validation set VGG16
pred_vgg = np.array([])
for batch_idx, (X_test, labels) in enumerate(validloader):
    pred_vgg = np.append(pred_vgg, vgg.predict(X_test).tolist())
print('VGG16 prediction done!')
# extract labels from validation set
y_valid = np.array([y for X, y in iter(validset)])
pd.DataFrame(np.column_stack((pred_resnet, pred_densenet, pred_vgg, y_valid))[25:50],
            columns=['ResNet50', 'DenseNet169', 'VGG16', 'True Label'])
# Decision Tree Classifier
clf1 = DecisionTreeClassifier(random_state=4)
clf1.fit(np.column_stack((pred_resnet, pred_densenet, pred_vgg)), y_valid)

# Support Vector Machine Classifier
clf2 = SVC(random_state=4)
clf2.fit(np.column_stack((pred_resnet, pred_densenet, pred_vgg)), y_valid)

# Random Forest Classifier
clf3 = RandomForestClassifier(random_state=4)
clf3.fit(np.column_stack((pred_resnet, pred_densenet, pred_vgg)), y_valid)

# Gradient Boosting Classifier
clf4 = GradientBoostingClassifier(learning_rate=0.05, max_depth=1, random_state=4)
clf4.fit(np.column_stack((pred_resnet, pred_densenet, pred_vgg)), y_valid)
clfs = ['Decision Tree', 'SVC', 'Random Forest', 'Gradient Boosting']
scores = []
for clf in [clf1, clf2, clf3, clf4]:
    scores.append(
        cross_val_score(clf, np.column_stack((pred_resnet, pred_densenet, pred_vgg)),
                        y_valid, scoring='accuracy', cv=5))

pd.DataFrame(scores, index=clfs, columns=[i for i in range(1, 6)])
# Voting Classifier
eclf = VotingClassifier(estimators=[('dt', clf1), ('svc', clf2),
                                    ('rf', clf3), ('gbc', clf4)], voting='hard')
# output results of cross validation score
for clf, label in zip([clf1, clf2, clf3, clf4, eclf], 
                      ['Desicion Trees', 'SVC', 'Random Forest',
                       'Gradient Boosting', 'Ensemble']):
    scores = cross_val_score(clf, np.column_stack((pred_resnet, pred_densenet, pred_vgg)),
                             y_valid, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
# fit Voting Classifier
eclf.fit(np.column_stack((pred_resnet, pred_densenet, pred_vgg)), y_valid)
# list to store names of images
test_names_list = []

# predict on test set ResNet50
test_pred_resnet = np.array([])
for batch_idx, (X_test, names) in enumerate(testloader):
    test_pred_resnet = np.append(test_pred_resnet, resnet.predict(X_test).tolist())
    test_names_list += [name.split('/')[::-1][0] for name in names]

# predict on test set DenseNet169
test_pred_densenet = np.array([])
for batch_idx, (X_test, names) in enumerate(testloader):
    test_pred_densenet = np.append(test_pred_densenet, densenet.predict(X_test).tolist())

# predict on test set VGG16
test_pred_vgg = np.array([])
for batch_idx, (X_test, names) in enumerate(testloader):
    test_pred_vgg = np.append(test_pred_vgg, vgg.predict(X_test).tolist())
# predict Voting Classifier with 4 estimators
final = eclf.predict(np.column_stack((test_pred_resnet, test_pred_densenet, test_pred_vgg)))
# print images with labels
samples, names = next(iter(testloader))

fig = plt.figure(figsize=(16, 16))
fig.suptitle("Some examples of test images with predicted results", fontsize=16)
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(np.transpose(samples[i], (1, 2, 0)), cmap=plt.cm.binary)
    plt.xlabel(class_names[final[i]])
plt.show()
# create submission file
submission = pd.DataFrame({'image_name': test_names_list, 'label': final})
submission.to_csv('submission_final.csv', index=False)
print('Submission file is created!')