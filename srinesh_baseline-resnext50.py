import os

import numpy as np

import pandas as pd 

from tqdm import tqdm

import gc



# Imaging libraries

import seaborn as sns; sns.set()

import pydicom

import matplotlib.pyplot as plt

import cv2



# Deep learning libraries

import torch.optim as optim

import torch 

import torchvision.models as models

from torch.utils.data import Dataset
BASE_PATH = '/kaggle/input/rsna-intracranial-hemorrhage-detection/'

TRAIN_DIR = 'stage_1_train_images/'

TEST_DIR = 'stage_1_test_images/'



TRAIN_CSV = 'stage_1_train.csv'

TEST_CSV = 'stage_1_sample_submission.csv'



MODEL_PATH = '/kaggle/input/baseline-resnext50/resnext50_10.pth'



TRAIN_CSV_PATH = os.path.join(BASE_PATH, TRAIN_CSV)

TEST_CSV_PATH = os.path.join(BASE_PATH, TEST_CSV)
df_train = pd.read_csv(TRAIN_CSV_PATH)

df_train[['id', 'img', 'subtype']] = df_train['ID'].str.split('_', n=3, expand=True)

df_train['img'] = 'ID_' + df_train['img'] 



df_train.drop_duplicates(inplace=True)

df_train = df_train.pivot(index='img', columns='subtype', values='Label').reset_index()

df_train['path'] = os.path.join(BASE_PATH, TRAIN_DIR) + df_train['img'] + '.dcm'



# Only include valid images (some images are excluded for training)

legit_images = pd.read_csv('/kaggle/input/true-imagescsv/legit-images.csv')

df_train = df_train.merge(legit_images, left_on='img', right_on='0').drop(['0'], axis=1)

df_train.head()
df_test = pd.read_csv(TEST_CSV_PATH)

df_test[['id','img','subtype']] = df_test['ID'].str.split('_', expand=True)

df_test['img'] = 'ID_' + df_test['img']

df_test = df_test[['img', 'Label']]

df_test['path'] = os.path.join(BASE_PATH, TEST_DIR) + df_test['img'] + '.dcm'

df_test.drop_duplicates(inplace=True)



df_test = df_test.reset_index(drop=True)
class RSNADataset(Dataset):

  def __init__(self, df, labels):

        self.data = df

        self.labels = labels



  def __len__(self):

        return len(self.data)



  def __getitem__(self, index):

        

        img_name = self.data.loc[index, 'path']   

        

        img_dcm = pydicom.read_file(img_name)

        img = RSNADataset.brain_window(img_dcm)

        img = cv2.resize(img, (200,200))

        

        img = np.stack((img,)*3, axis=-1)

        img = np.transpose(img, (2, 1, 0))

    

                

        if self.labels:        

            labels = torch.tensor(

                self.data.loc[index, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])

            return {'image': img, 'labels': labels}   

        else:

            return {'image': img}

  

  @staticmethod      

  def brain_window(img):

        window_min = 0

        window_max = 80

        _, _, intercept, slope = RSNADataset.get_windowing(img)

        img = img.pixel_array.astype('float32')

        img = img * slope + intercept

        img[img < window_min] = window_min

        img[img > window_max] = window_max

        img = (img - np.min(img)) / 1e-5+ (np.max(img) - np.min(img))

        return img

  

  @staticmethod

  def get_windowing(data):

        dicom_fields = [data[('0028','1050')].value, #window center

                        data[('0028','1051')].value, #window width

                        data[('0028','1052')].value, #intercept

                        data[('0028','1053')].value] #slope

        return [RSNADataset.get_first_of_dicom_field_as_int(x) for x in dicom_fields]

  

  @staticmethod

  def get_first_of_dicom_field_as_int(x):

    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)

    if type(x) == pydicom.multival.MultiValue:

        return int(x[0])

    else:

        return int(x)
params = {'batch_size': 64,

          'shuffle': False,

          'num_workers': 4}



train_dataset = RSNADataset(df= df_train, labels=True)

test_dataset = RSNADataset(df= df_test, labels=False)



data_train_generator = torch.utils.data.DataLoader(train_dataset, **params)

data_test_generator = torch.utils.data.DataLoader(test_dataset,**params)
# Plot train images

batch = next(iter(data_train_generator))

fig, axs = plt.subplots(1, 3, figsize=(15,5))



for i in np.arange(3):

    

    axs[i].imshow(batch['image'][i][0].numpy(), cmap=plt.cm.bone)
# Plot test images

batch = next(iter(data_test_generator))

fig, axs = plt.subplots(1, 3, figsize=(15,5))



for i in np.arange(3):

    

    axs[i].imshow(batch['image'][i][0].numpy(), cmap=plt.cm.bone)
device = torch.device("cuda:0")

model0 = models.resnext50_32x4d(pretrained=True)

model = torch.nn.Sequential(model0, torch.nn.Linear(1000, 6) ) 



model = model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()
n_epochs = 2

optimizer = optim.Adam(model.parameters(), lr=4e-5)



try:

    model.load_state_dict(torch.load(MODEL_PATH))

    torch.save(model.state_dict(), 'resnext50_0.pth') 

except Exception as e:

    print('The pre-trained model is used')
for epoch in range(1, n_epochs+1):

    

    print('Epoch {}/{}'.format(epoch, n_epochs))

    print('-' * 10)



    model.train()    

    tr_loss = 0

    

    tk0 = tqdm(data_train_generator, desc="Iteration")

    

    for step, batch in enumerate(tk0):

        

        inputs = batch["image"]

        labels = batch["labels"]



        inputs = inputs.to(device, dtype=torch.float)

        labels = labels.to(device, dtype=torch.float)



        outputs = model(inputs)

        loss = criterion(outputs, labels)

                

        loss.backward()



        tr_loss += loss.item()



        optimizer.step()

        optimizer.zero_grad()

     

    torch.save(model.state_dict(), f'resnext50_{epoch}.pth') 



    epoch_loss = tr_loss / len(data_train_generator)

    print('Training Loss: {:.4f}'.format(epoch_loss))
for param in model.parameters():

    param.requires_grad = False



model.eval()



test_pred = np.zeros((len(test_dataset) * 6, 1))



for i, batch_ in enumerate(tqdm(data_test_generator)):

    batch_ = batch_["image"]

    batch_ = batch_.to(device, dtype=torch.float)

    

    with torch.no_grad():

        

        pred = model(batch_)

        

        test_pred[(i * 64 * 6):((i + 1) * 64 * 6)] = torch.sigmoid(

            pred).detach().cpu().reshape((len(batch_) * 6, 1))  
submission =  pd.read_csv(TEST_CSV_PATH)

submission = pd.concat([submission.drop(columns=['Label']), pd.DataFrame(test_pred)], axis=1)

submission.columns = ['ID', 'Label']



submission.to_csv('submission.csv', index=False)