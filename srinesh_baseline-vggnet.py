import os

import numpy as np

import pandas as pd 

from glob import glob

from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score

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



MODEL_PATH = '/kaggle/input/baseline-vggnet/vggNet19.pth'

TRAIN_CSV_PATH = os.path.join(BASE_PATH, TRAIN_CSV)

TEST_CSV_PATH = os.path.join(BASE_PATH, TEST_CSV)
df_train = pd.read_csv(TRAIN_CSV_PATH)

df_train[['id', 'img', 'subtype']] = df_train['ID'].str.split('_', n=3, expand=True)

df_train['img'] = 'ID_' + df_train['img'] 



df_train.drop_duplicates(inplace=True)

df_train = df_train.pivot(index='img', columns='subtype', values='Label').reset_index()

df_train['path'] = os.path.join(BASE_PATH, TRAIN_DIR) + df_train['img'] + '.dcm'



# Only include valid images

legit_images = pd.read_csv('/kaggle/input/true-imagescsv/legit-images.csv')

df_train = df_train.merge(legit_images, left_on='img', right_on='0')

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

        img = pydicom.read_file(img_name).pixel_array.astype('float32')

        img = cv2.resize(img, (224,224))



        img = np.stack((img,)*3, axis=-1)

        img = np.transpose(img, (2, 1, 0))

    

                

        if self.labels:        

            labels = torch.tensor(

                self.data.loc[index, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])

            return {'image': img, 'labels': labels}   

        else:

            return {'image': img}

params = {'batch_size': 64,

          'shuffle': False,

          'num_workers': 4}



train_dataset = RSNADataset(df= df_train, labels=True)

test_dataset = RSNADataset(df= df_test, labels=False)



data_train_generator = torch.utils.data.DataLoader(train_dataset, **params)

data_test_generator = torch.utils.data.DataLoader(test_dataset,**params)
batch = next(iter(data_train_generator))

fig, axs = plt.subplots(1, 3, figsize=(15,5))



for i in np.arange(3):

    

    axs[i].imshow(batch['image'][i][0].numpy(), cmap=plt.cm.bone)
device = torch.device("cuda:0")

model0 = models.vgg19_bn()



model = torch.nn.Sequential(model0, torch.nn.Linear(1000, 6) ) 



model = model.to(device)
n_epochs = 3

criterion = torch.nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=2e-5)



def weights_init_uniform_rule(m):

    classname = m.__class__.__name__

    # for every Linear layer in a model..

    if classname.find('Linear') != -1:

        # get the number of the inputs

        n = m.in_features

        y = 1.0/np.sqrt(n)

        m.weight.data.uniform_(-y, y)

        m.bias.data.fill_(0)



# Load weights of the last trained model or create a new model with random weights 

try:

    model.load_state_dict(torch.load(MODEL_PATH))

    print(f'loaded the model at {MODEL_PATH}')

except Exception as e:

    model = model.apply(weights_init_uniform_rule)
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

            



    epoch_loss = tr_loss / len(data_train_generator)

    print('Training Loss: {:.4f}'.format(epoch_loss))
# Saving the model

torch.save(model.state_dict(), 'vggNet19-1.pth') 
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



submission.to_csv('submission-1.csv', index=False)