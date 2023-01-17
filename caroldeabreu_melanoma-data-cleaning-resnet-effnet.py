import seaborn as sns
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import pydicom
from skimage.transform import resize
import os
import numpy as np
import pandas as pd
import random
import time
train_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
test_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
train_df.head()
directory = '/kaggle/input/siim-isic-melanoma-classification'

# === JPEG ===
# Create the paths
path_train = directory + '/jpeg/train/' + train_df['image_name'] + '.jpg'
path_test = directory + '/jpeg/test/' + test_df['image_name'] + '.jpg'

# Append to the original dataframes
train_df['path_jpeg'] = path_train
test_df['path_jpeg'] = path_test
# # === DICOM ===
# # Create the paths
# path_train = directory + '/train/' + train_df['image_name'] + '.dcm'
# path_test = directory + '/test/' + test_df['image_name'] + '.dcm'

# # Append to the original dataframes
# train_df['path_dicom'] = path_train
# test_df['path_dicom'] = path_test
def show_images(data, n = 5, rows=1, cols=5, title='Default'):
    plt.figure(figsize=(16,4))

    for k, path in enumerate(data['path_dicom'][:n]):
        image = pydicom.read_file(path)
        image = image.pixel_array
        
        image = resize(image, (200, 200), anti_aliasing=True)

        plt.suptitle(title, fontsize = 16)
        plt.subplot(rows, cols, k+1)
        plt.imshow(image)
        plt.axis('off')
shapes_train = []

for k, path in enumerate(train_df['path_jpeg']):
    image = Image.open(path)
    shapes_train.append(image.size)
    
    if k >= 100: break
        
shapes_train = pd.DataFrame(data = shapes_train, columns = ['H', 'W'], dtype='object')
shapes_train['Size'] = '[' + shapes_train['H'].astype(str) + ', ' + shapes_train['W'].astype(str) + ']'
# Show Benign Samples
show_images(train_df[train_df['target'] == 1], n=10, rows=2, cols=5, title='Benign Sample')
# Libraries

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
!pip install -q efficientnet_pytorch   

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Data Augmentation for Image Preprocessing
from albumentations import (ToFloat, Normalize, VerticalFlip, HorizontalFlip, Compose, Resize,
                            RandomBrightness, RandomContrast, HueSaturationValue, Blur, GaussNoise,
                            Rotate, RandomResizedCrop, Cutout)
from albumentations.pytorch import ToTensorV2, ToTensor

from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet34, resnet50

import warnings
warnings.filterwarnings("ignore")
X = train_df.drop(['diagnosis', 'benign_malignant'], axis=1)
y = train_df['target']

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())
])

cat_cols = ['sex', 'anatom_site_general_challenge']

preprocessor = ColumnTransformer(
    transformers=[
        ('num_transform', numerical_transformer, ['age_approx']),
        ('cat_transform', categorical_transformer, cat_cols)
    ], remainder='passthrough')

def preprocessor():
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    cat_cols = ['sex', 'anatom_site_general_challenge']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_transform', numerical_transformer, ['age_approx']),
            ('cat_transform', categorical_transformer, cat_cols)
        ], remainder='passthrough')

    return preprocessor

def clean_dataframe(dataframe, preprocessor, is_train=True, is_valid=False):
    
    cols = ['age_approx', 'sex', 'anatom_site_general_challenge', 'path_jpeg']
        
    if is_train:
        df = pd.DataFrame(preprocessor.fit_transform(dataframe[cols]))
    else:
        df = pd.DataFrame(preprocessor.transform(dataframe[cols]))
        
    df.columns = [*df.columns[:-1], 'path_jpeg']
    if is_train or is_valid:
        df['target'] = dataframe.reset_index()['target']
    
    return df
train_X, val_X, train_y, val_y = train_test_split(train_df, train_df['target'], random_state=1)
transformer = preprocessor()
clean_train_X = clean_dataframe(train_X, transformer, is_train=True, is_valid=False)
clean_val_X = clean_dataframe(val_X, transformer, is_train=False, is_valid=True)
clean_test = clean_dataframe(test_df, transformer, is_train=False, is_valid=False)
class MelanomaDataset(Dataset):
    
    def __init__(self, dataframe, vertical_flip, horizontal_flip,
                 is_train=True, is_valid=False, is_test=False):
        self.dataframe, self.is_train, self.is_valid = dataframe, is_train, is_valid
        self.vertical_flip, self.horizontal_flip = vertical_flip, horizontal_flip
        
        # Data Augmentation (custom for each dataset type)
        if is_train or is_test:
            self.transform = Compose([RandomResizedCrop(height=224, width=224, scale=(0.7, 1.0)),
                                      HorizontalFlip(p = self.horizontal_flip),
                                      VerticalFlip(p = self.vertical_flip),
                                      Cutout(),
                                      HueSaturationValue(),
                                      Normalize(),
                                      ToTensor()])
        else:
            self.transform = Compose([Normalize(),
                                      ToTensor()])
            
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        # Select path and read image
        image_path = self.dataframe['path_jpeg'][index]
        image = cv2.imread(image_path)
        # For this image also import .csv information (sex, age, anatomy)
        csv_data = np.array(self.dataframe.drop(['path_jpeg', 'target'], axis=1).iloc[index].values, 
                            dtype=np.float32)
        
        # Apply transforms
        image = self.transform(image=image)
        # Extract image from dictionary
        image = image['image']
        
        # If train/valid: image + class | If test: only image
        if self.is_train or self.is_valid:
            return (image, csv_data), self.dataframe['target'][index]
        else:
            return (image, csv_data)
class ResNet50Network(nn.Module):
    def __init__(self, output_size, no_columns):
        super().__init__()
        self.no_columns, self.output_size = no_columns, output_size
        
        # Define Feature part (IMAGE)
        self.features = resnet50(pretrained=True) # 1000 neurons out
        # (CSV data)
        self.csv = nn.Sequential(nn.Linear(self.no_columns, 500),
                                 nn.BatchNorm1d(500),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2))
        
        # Define Classification part
        self.classification = nn.Linear(1000 + 500, output_size)
        
    def forward(self, image, csv_data, prints=True):
        
        if prints: print('Input Image shape:', image.shape, '\n'+
                         'Input csv_data shape:', csv_data.shape)
        
        # Image CNN
        image = self.features(image)
        if prints: print('Features Image shape:', image.shape)
        
        # CSV FNN
        csv_data = self.csv(csv_data)
        if prints: print('CSV Data:', csv_data.shape)
            
        # Concatenate layers from image with layers from csv_data
        image_csv_data = torch.cat((image, csv_data), dim=1)
        
        # CLASSIF
        out = self.classification(image_csv_data)
        if prints: print('Out shape:', out.shape)
        
        return out
# ----- STATICS -----
vertical_flip = 0.5
horizontal_flip = 0.5
# ------------------
# Data object and Loader

transformer = preprocessor()
train_X_clean = clean_dataframe(train_X, transformer)

example_data = MelanomaDataset(train_X_clean, vertical_flip=0.5, horizontal_flip=0.5, 
                               is_train=True, is_valid=False, is_test=False)
example_loader = torch.utils.data.DataLoader(example_data, batch_size = 3, shuffle=True)

no_columns = len(example_data.dataframe.columns)-2
model_example = ResNet50Network(output_size=1, no_columns=no_columns)

# Get a sample
for (image, csv_data), labels in example_loader:
    image_example, csv_data_example = image, csv_data
    labels_example = torch.tensor(labels, dtype=torch.float32)
    break
print('Data shape:', image_example.shape, '| \n' , csv_data_example)
print('Label:', labels_example, '\n')

# Outputs
out = model_example(image_example, csv_data_example, prints=True)

# Criterion example
criterion_example = nn.BCEWithLogitsLoss()
# Unsqueeze(1) from shape=[3] to shape=[3, 1]
loss = criterion_example(out, labels_example.unsqueeze(1))   
print('Loss:', loss.item())
class EfficientNetwork(nn.Module):
    def __init__(self, output_size, no_columns, b4=False, b2=False):
        super().__init__()
        self.b4, self.b2, self.no_columns = b4, b2, no_columns
        
        # Define Feature part (IMAGE)
        if b4:
            self.features = EfficientNet.from_pretrained('efficientnet-b4')
        elif b2:
            self.features = EfficientNet.from_pretrained('efficientnet-b2')
        else:
            self.features = EfficientNet.from_pretrained('efficientnet-b7')
        
        # (CSV)
        self.csv = nn.Sequential(nn.Linear(self.no_columns, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),
                                 
                                 nn.Linear(250, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2))
        
        # Define Classification part
        if b4:
            self.classification = nn.Sequential(nn.Linear(1792 + 250, output_size))
        elif b2:
            self.classification = nn.Sequential(nn.Linear(1408 + 250, output_size))
        else:
            self.classification = nn.Sequential(nn.Linear(2560 + 250, output_size))
        
        
    def forward(self, image, csv_data, prints=False):    
        
        if prints: print('Input Image shape:', image.shape, '\n'+
                         'Input csv_data shape:', csv_data.shape)
        
        # IMAGE CNN
        image = self.features.extract_features(image)
        if prints: print('Features Image shape:', image.shape)
            
        if self.b4:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1792)
        elif self.b2:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1408)
        else:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 2560)
        if prints: print('Image Reshaped shape:', image.shape)
            
        # CSV FNN
        csv_data = self.csv(csv_data)
        if prints: print('CSV Data:', csv_data.shape)
            
        # Concatenate
        image_csv_data = torch.cat((image, csv_data), dim=1)
        
        # CLASSIF
        out = self.classification(image_csv_data)
        if prints: print('Out shape:', out.shape)
        
        return out
# Data object and Loader
transformer = preprocessor()
train_X_clean = clean_dataframe(train_X, transformer)

example_data = MelanomaDataset(train_X_clean, vertical_flip=0.5, horizontal_flip=0.5, 
                               is_train=True, is_valid=False, is_test=False)
example_loader = torch.utils.data.DataLoader(example_data, batch_size = 3, shuffle=True)
no_columns = len(example_data.dataframe.columns)-2

# Create an example model - Effnet
model_example = EfficientNetwork(output_size=1, no_columns=no_columns,
                                 b4=False, b2=False)
# Get a sample
for (image, csv_data), labels in example_loader:
    image_example, csv_data_example = image, csv_data
    labels_example = torch.tensor(labels, dtype=torch.float32)
    break
print('Data shape:', image_example.shape, '| \n' , csv_data_example)
print('Label:', labels_example, '\n')

# Outputs
out = model_example(image_example, csv_data_example, prints=True)

# Criterion example
criterion_example = nn.BCEWithLogitsLoss()
# Unsqueeze(1) from shape=[3] to shape=[3, 1]
loss = criterion_example(out, labels_example.unsqueeze(1))   
print('Loss:', loss.item())
def set_seed(seed = 1234):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device available now:', device)
def train_folds(preds_submission, model, version = 'v1'):
    # Creates a .txt file that will contain the logs
    f = open(f"logs_{version}.txt", "w+")
    
    
    for fold, (train_index, valid_index) in enumerate(list(folds)[:2]):
        # Append to .txt
        with open(f"logs_{version}.txt", 'a+') as f:
            print('-'*10, 'Fold:', fold+1, '-'*10, file=f)
        print('-'*10, 'Fold:', fold+1, '-'*10)


        # --- Create Instances ---
        # Best ROC score in this fold
        best_roc = None
        # Reset patience before every fold
        patience_f = patience
        
        # Initiate the model
        model = model

        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', 
                                      patience=lr_patience, verbose=True, factor=lr_factor)
        criterion = nn.BCEWithLogitsLoss()


        # --- Read in Data ---
        print('Reading data ... \n')
        train_data = train_df.iloc[train_index].reset_index(drop=True)
        valid_data = train_df.iloc[valid_index].reset_index(drop=True)

        transformer = preprocessor()
        train_data_clean = clean_dataframe(train_data, transformer)
        valid_data_clean = clean_dataframe(valid_data, transformer, is_train=False, is_valid=True)
        
        # Create Data instances
        train = MelanomaDataset(train_data_clean, vertical_flip=vertical_flip, horizontal_flip=horizontal_flip, 
                                is_train=True, is_valid=False, is_test=False)
        valid = MelanomaDataset(valid_data_clean, vertical_flip=vertical_flip, horizontal_flip=horizontal_flip, 
                                is_train=False, is_valid=True, is_test=False)
        # Read in test data | Remember! We're using data augmentation like we use for Train data.
        
        test_clean = clean_dataframe(valid_data, transformer, is_train=False, is_valid=False)
        test = MelanomaDataset(test_clean, vertical_flip=vertical_flip, horizontal_flip=horizontal_flip,
                               is_train=False, is_valid=False, is_test=True)

        # Dataloaders
        train_loader = DataLoader(train, batch_size=batch_size1, shuffle=True, num_workers=num_workers)
        # shuffle=False! Otherwise function won't work!!!
                # how do I know? ^^
        valid_loader = DataLoader(valid, batch_size=batch_size2, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test, batch_size=batch_size2, shuffle=False, num_workers=num_workers)

        print('Starting training ...\n')
        # === EPOCHS ===
        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1} ... \n")
            start_time = time.time()
            print(f"Time: {start_time}")
            correct = 0
            train_losses = 0

            # === TRAIN ===
            # Sets the module in training mode.
            model.train()

            for (images, csv_data), labels in train_loader:
                # Save them to device
                images = torch.tensor(images, device=device, dtype=torch.float32)
                csv_data = torch.tensor(csv_data, device=device, dtype=torch.float32)
                labels = torch.tensor(labels, device=device, dtype=torch.float32)

                # Clear gradients first; very important, usually done BEFORE prediction
                optimizer.zero_grad()

                # Log Probabilities & Backpropagation
                out = model(images, csv_data)
                loss = criterion(out, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()

                # --- Save information after this batch ---
                # Save loss
                train_losses += loss.item()
                # From log probabilities to actual probabilities
                train_preds = torch.round(torch.sigmoid(out)) # 0 and 1
                # Number of correct predictions
                correct += (train_preds.cpu() == labels.cpu().unsqueeze(1)).sum().item()

            # Compute Train Accuracy
            train_acc = correct / len(train_index)

            print('Starting evaluation for epoch ... \n')
            # === EVAL ===
            # Sets the model in evaluation mode
            model.eval()

            # Create matrix to store evaluation predictions (for accuracy)
            valid_preds = torch.zeros(size = (len(valid_index), 1), device=device, dtype=torch.float32)

            # Disables gradients (we need to be sure no optimization happens)
            with torch.no_grad():
                for k, ((images, csv_data), labels) in enumerate(valid_loader):
                    images = torch.tensor(images, device=device, dtype=torch.float32)
                    csv_data = torch.tensor(csv_data, device=device, dtype=torch.float32)
                    labels = torch.tensor(labels, device=device, dtype=torch.float32)

                    out = model(images, csv_data)
                    pred = torch.sigmoid(out)
                    valid_preds[k*images.shape[0] : k*images.shape[0] + images.shape[0]] = pred

                # Compute accuracy
                valid_acc = accuracy_score(valid_data['target'].values, 
                                           torch.round(valid_preds.cpu()))
                # Compute ROC
                valid_roc = roc_auc_score(valid_data['target'].values, 
                                          valid_preds.cpu())

                # Compute time on Train + Eval
                duration = str(datetime.timedelta(seconds=time.time() - start_time))[:7]


                # PRINT INFO
                # Append to .txt file
                with open(f"logs_{version}.txt", 'a+') as f:
                    print('{} | Epoch: {}/{} | Loss: {:.4} | Train Acc: {:.3} | Valid Acc: {:.3} | ROC: {:.3}'.\
                     format(duration, epoch+1, epochs, train_losses, train_acc, valid_acc, valid_roc), file=f)
                # Print to console
                print('{} | Epoch: {}/{} | Loss: {:.4} | Train Acc: {:.3} | Valid Acc: {:.3} | ROC: {:.3}'.\
                     format(duration, epoch+1, epochs, train_losses, train_acc, valid_acc, valid_roc))


                # === SAVE MODEL ===

                # Update scheduler (for learning_rate)
                scheduler.step(valid_roc)

                # Update best_roc
                if not best_roc: # If best_roc = None
                    best_roc = valid_roc
                    torch.save(model.state_dict(),
                               f"Fold{fold+1}_Epoch{epoch+1}_ValidAcc_{valid_acc:.3f}_ROC_{valid_roc:.3f}.pth")
                    continue

                if valid_roc > best_roc:
                    best_roc = valid_roc
                    # Reset patience (because we have improvement)
                    patience_f = patience
                    torch.save(model.state_dict(),
                               f"Fold{fold+1}_Epoch{epoch+1}_ValidAcc_{valid_acc:.3f}_ROC_{valid_roc:.3f}.pth")
                else:
                    # Decrease patience (no improvement in ROC)
                    patience_f = patience_f - 1
                    if patience_f == 0:
                        with open(f"logs_{version}.txt", 'a+') as f:
                            print('Early stopping (no improvement since 3 models) | Best ROC: {}'.\
                                  format(best_roc), file=f)
                        print('Early stopping (no improvement since 3 models) | Best ROC: {}'.\
                              format(best_roc))
                        break


        # === INFERENCE ===
        # Choose model with best_roc in this fold
        best_model_path = '../working/' + [file for file in os.listdir('../working') if str(round(best_roc, 3)) in file and 'Fold'+str(fold+1) in file][0]
        # Using best model from Epoch Train
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        model = EfficientNetwork(output_size = output_size, no_columns=no_columns,
                         b4=False, b2=True).to(device)
        model.load_state_dict(torch.load(best_model_path))
        # Set the model in evaluation mode
        model.eval()


        with torch.no_grad():
            # --- EVAL ---
            # Predicting again on Validation data to get preds for OOF
            valid_preds = torch.zeros(size = (len(valid_index), 1), device=device, dtype=torch.float32)

            for k, ((images, csv_data), _) in enumerate(valid_loader):
                images = torch.tensor(images, device=device, dtype=torch.float32)
                csv_data = torch.tensor(csv_data, device=device, dtype=torch.float32)

                out = model(images, csv_data)
                pred = torch.sigmoid(out)
                valid_preds[k*images.shape[0] : k*images.shape[0] + images.shape[0]] = pred

            # Save info to OOF
            oof[valid_index] = valid_preds.cpu().numpy()


            # --- TEST ---
            # Now (Finally) prediction for our TEST data
            for i in range(TTA):
                for k, (images, csv_data) in enumerate(test_loader):
                    images = torch.tensor(images, device=device, dtype=torch.float32)
                    csv_data = torch.tensor(csv_data, device=device, dtype=torch.float32)

                    out = model(images, csv_data)
                    # Covert to probablities
                    out = torch.sigmoid(out)

                    # ADDS! the prediction to the matrix we already created
                    preds_submission[k*images.shape[0] : k*images.shape[0] + images.shape[0]] += out


            # Divide Predictions by TTA (to average the results during TTA)
            preds_submission /= TTA


        # === CLEANING ===
        # Clear memory
        del train, valid, train_loader, valid_loader, images, labels
        # Garbage collector
        gc.collect()
# ----- STATICS -----
output_size=1
epochs = 15
patience = 3
TTA = 3
num_workers = 8
learning_rate = 0.0005
weight_decay = 0.0
lr_patience = 1 
lr_factor = 0.4            

batch_size1 = 16
batch_size2 = 16

version = 'v1'
# ----- STATICS -----
train_len = len(train_df)
test_len = len(test_df)
# -------------------


# Out of Fold Predictions
oof = np.zeros(shape = (train_len, 1))

# Predictions
preds_submission = torch.zeros(size = (test_len, 1), dtype=torch.float32, device=device)

print('oof shape:', oof.shape, '\n' +
      'predictions shape:', preds_submission.shape)
k = 5

# Create Object
group_fold = GroupKFold(n_splits = k)

# Generate indices to split data into training and test set.
folds = group_fold.split(X = np.zeros(train_len), 
                         y = train_df['target'], 
                         groups = train_df['patient_id'].tolist())
# --- EffNet B2 ---
model = ResNet50Network(output_size = output_size, no_columns=9).to(device)

# ===== Uncomment and Train =====
train_folds(preds_submission = preds_submission, model = model, version = version)

# Save OOF values
save_oof = pd.DataFrame(data = oof, columns=['oof'])
save_oof.to_csv(f'oof_{version}.csv', index=False)
f = open('./logs_v1.txt', "r")
contents = f.read()
print(contents)
