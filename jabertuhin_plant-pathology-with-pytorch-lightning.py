import numpy as np

import pandas as pd

import os



import matplotlib.pyplot as plt

import matplotlib.image as mpimg



%matplotlib inline
TRAIN_CSV_PATH = '../input/plant-pathology-2020-fgvc7/train.csv'

TEST_CSV_PATH = '../input/plant-pathology-2020-fgvc7/test.csv'

INPUT_IMAGES_DIR = '../input/plant-pathology-2020-fgvc7/images/'

ROOT_INPUT_DIR = '../input/plant-pathology-2020-fgvc7/'
dataset = pd.read_csv(TRAIN_CSV_PATH)

dataset.head()
dataset.info()
num_healthy_plant = dataset['healthy'].value_counts()

print("Number of healthy leafs: {}".format(num_healthy_plant.loc[1]))

print("Number of unhealthy leafs: {}".format(num_healthy_plant.loc[0]))
num_multi_diseases_plant = dataset['multiple_diseases'].value_counts()

print("Number of leafs has multiple diseases: {}".format(num_multi_diseases_plant.loc[1]))

print("Number of leafs doesn't has multiple diseases: {}".format(num_multi_diseases_plant.loc[0]))



num_multi_diseases_plant.plot.bar()
num_rust_plant = dataset['rust'].value_counts()

print("Number of leafs has rust: {}".format(num_rust_plant.loc[1]))

print("Number of leafs doesn't has rust: {}".format(num_rust_plant.loc[0]))



num_rust_plant.plot.bar()
num_scab_plant = dataset['scab'].value_counts()

print("Number of leafs has scab: {}".format(num_scab_plant.loc[1]))

print("Number of leafs doesn't has scab: {}".format(num_scab_plant.loc[0]))



num_scab_plant.plot.bar()
def show_rgb_color_channels(image_id):        

    fig = plt.figure(figsize=(15, 12))

    img = mpimg.imread(INPUT_IMAGES_DIR + image_id + '.jpg')

    image_titles = ['Original Image', 'R Channels', 'G Channels', 'B Channels']

    # showing RGB image

    ax = fig.add_subplot(1, 4, 1)

    ax.title.set_text(image_titles[0])

    plt.imshow(img)



    for i in range(3):

        ax = fig.add_subplot(1, 4, i+2)

        ax.title.set_text(image_titles[i+1])

        plt.imshow(img[:, :, i])



    plt.show()
# A healthy leaf

healthy_leaf_image_ids = dataset[dataset['healthy'] == 1]['image_id']

show_rgb_color_channels(healthy_leaf_image_ids.iloc[0])
# A leaf with multiple_diseases

multiple_diseases_leaf_image_ids = dataset[dataset['multiple_diseases'] == 1]['image_id']

show_rgb_color_channels(multiple_diseases_leaf_image_ids.iloc[0])
# A leaf with rust

rust_leaf_image_ids = dataset[dataset['rust'] == 1]['image_id']

show_rgb_color_channels(rust_leaf_image_ids.iloc[0])
# A leaf with scab

scab_leaf_image_ids = dataset[dataset['scab'] == 1]['image_id']

show_rgb_color_channels(scab_leaf_image_ids.iloc[0])
unique_shapes_of_images = set()



for image_id in dataset.image_id:

    img = mpimg.imread(INPUT_IMAGES_DIR + image_id + '.jpg')

    unique_shapes_of_images.add(img.shape)



unique_shapes_of_images
## Installing pytorch lightning

!pip install pytorch_lightning
# Ignore warnings

import warnings

warnings.filterwarnings("ignore")

from PIL import Image

from argparse import Namespace



import torch

from torch import nn

from torch.utils.data import Dataset

from torchvision import transforms

from torch.utils.data import DataLoader, random_split

from torch.nn import functional as F

import torchvision.models as models



import pytorch_lightning as pl
class PlantTrainDataset(Dataset):

    def __init__(self, df, image_root_dir):

        self.dataset = df                

        self.image_root_dir = image_root_dir  #'../input/images/'              

        self.transforms = transforms.Compose([transforms.RandomApply([transforms.RandomHorizontalFlip(),

                                                    transforms.RandomRotation(10),

                                                    transforms.RandomVerticalFlip(),

                                                    transforms.CenterCrop((224,224))

                                                    ]),

                                                transforms.Resize((224,224)),

                                                transforms.ToTensor(),

                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                                                        std=[0.229, 0.224, 0.225])                                                

                                                ])              



    def __len__(self):

        return len(self.dataset.image_id)



    def __getitem__(self, idx):        

        image_id = self.dataset.iloc[idx, 0]

        label = self.dataset.iloc[idx, 1:]

        label = torch.tensor(label, dtype=torch.long)        

        

        image = Image.open(self.image_root_dir + image_id + '.jpg')    

        image = self.transform_image(image)

        

        return image, label



    def transform_image(self, image):                

        image = self.transforms(image)

        return image
class PlantTestDataset(Dataset):

    def __init__(self, df, image_root_dir):

        self.dataset = df                

        self.transforms = transforms.Compose([transforms.Resize((224,224)),

                                                transforms.ToTensor(),

                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                                                        std=[0.229, 0.224, 0.225])                                                

                                                ])    

        self.image_root_dir = image_root_dir #'../input/images/'        



    def __len__(self):

        return len(self.dataset.image_id)



    def __getitem__(self, idx):

        image_id = self.dataset.iloc[idx, 0]        

        image = Image.open(self.image_root_dir + image_id + '.jpg')    

        image = self.transforms(image)

        

        return image_id, image
class PlantDiseaseClassifier(pl.LightningModule):



    def __init__(self, hyper_params): 

        super(PlantDiseaseClassifier, self).__init__()

        self.root_input_dir = ROOT_INPUT_DIR

        self.train_batch_size = hyper_params.train_batch_size

        self.test_batch_size = hyper_params.test_batch_size

        self.val_batch_size = hyper_params.val_batch_size

        self.learning_rate = hyper_params.learning_rate

        ### model architecture

        self.model = models.resnet18(pretrained=True)



        self.model.fc = nn.Linear(self.model.fc.in_features, 4)                

        self.soft_max = nn.Softmax(dim=1)



    def forward(self, x):

        x = self.model(x)

        x = self.soft_max(x)

        return x



    def cross_entropy_loss(self, logits, labels):        

        return F.nll_loss(logits, torch.argmax(labels, dim=1))



    def training_step(self, train_batch, batch_idx):

      x, y = train_batch

      logits = self(x) ##

      loss = self.cross_entropy_loss(logits, y)



      logs = {'train_loss': loss}

      return {'loss': loss, 'log': logs}



    def validation_step(self, val_batch, batch_idx):

      x, y = val_batch

      logits = self(x) ##

      loss = self.cross_entropy_loss(logits, y)

      return {'val_loss': loss}



    def validation_epoch_end(self, outputs):        

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss}

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}



    def prepare_data(self):                            

        train_dataset = pd.read_csv(self.root_input_dir + 'train.csv')

        test_dataset = pd.read_csv(self.root_input_dir + 'test.csv')

        

        image_dir = self.root_input_dir + 'images/'

        plant_train = PlantTrainDataset(train_dataset, image_dir)

        plant_test = PlantTestDataset(test_dataset, image_dir)

        

        self.plant_train, self.plant_val = random_split(plant_train, [1500, 321])

        self.plant_test = plant_test



    def train_dataloader(self):

        return DataLoader(self.plant_train, batch_size=self.train_batch_size)



    def val_dataloader(self):

        return DataLoader(self.plant_val, batch_size=self.val_batch_size)



    def test_dataloader(self):

        return DataLoader(self.plant_test, batch_size=self.test_batch_size)



    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer
#%load_ext tensorboard

#%tensorboard --logdir lightning_logs/
hyper_params = Namespace(**{

    'train_batch_size': 64,

    'test_batch_size': 8,

    'val_batch_size': 8,

    'learning_rate': 2e-4

})





model = PlantDiseaseClassifier(hyper_params)

trainer = pl.Trainer(max_epochs=50, gpus=1)



trainer.fit(model)
model.eval()



test_dataloader = model.test_dataloader()

predictions = []

for image_ids, images in test_dataloader:

    outputs = model(images.cuda())

    outputs = outputs.detach().cpu().numpy()

    for idx, image_id in enumerate(image_ids):

        predictions.append((image_id, outputs[idx, 0], outputs[idx, 1], outputs[idx, 2], outputs[idx, 3]))        
submission = pd.DataFrame(predictions, columns=['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab'])

submission.head()
submission.tail()
submission.to_csv('submission.csv', index=False)