# separating out the filenames of training, validation and dataset separately in 3 different list variables



import glob



train_files = glob.glob('/kaggle/input/tpu-getting-started/*/train/*.tfrec')

val_files = glob.glob('/kaggle/input/tpu-getting-started/*/val/*.tfrec')

test_files = glob.glob('/kaggle/input/tpu-getting-started/*/test/*.tfrec')
train_files
# now collecting the ids , filenames and images in bytes in three different list variables for train , validation & test files



# importing tensorfow to read .tfrec files

import tensorflow as tf



for i in train_files:

    train_image_dataset = tf.data.TFRecordDataset(i)



    # Create a dictionary describing the features.

    train_feature_description = {

        'class': tf.io.FixedLenFeature([], tf.int64),

        'id': tf.io.FixedLenFeature([], tf.string),

        'image': tf.io.FixedLenFeature([], tf.string),

    }



def _parse_image_function(example_proto):

  # Parse the input tf.Example proto using the dictionary above.

  return tf.io.parse_single_example(example_proto, train_feature_description)



train_image_dataset = train_image_dataset.map(_parse_image_function)





train_ids = [str(image_features['id'].numpy())[2:-1] for image_features in train_image_dataset] # [2:-1] is done to remove b' from 1st and 'from last in train id names



train_class = [int(image_features['class'].numpy()) for image_features in train_image_dataset]



train_images = [image_features['image'].numpy() for image_features in train_image_dataset]
for i in val_files:

    val_image_dataset = tf.data.TFRecordDataset(i)



    # Create a dictionary describing the features.

    val_feature_description = {

        'class': tf.io.FixedLenFeature([], tf.int64),

        'id': tf.io.FixedLenFeature([], tf.string),

        'image': tf.io.FixedLenFeature([], tf.string),

    }



def _parse_image_function(example_proto):

  # Parse the input tf.Example proto using the dictionary above.

  return tf.io.parse_single_example(example_proto, val_feature_description)



val_image_dataset = val_image_dataset.map(_parse_image_function)





val_ids = [str(image_features['id'].numpy())[2:-1] for image_features in val_image_dataset]



val_class = [int(image_features['class'].numpy()) for image_features in val_image_dataset]



val_images = [image_features['image'].numpy() for image_features in val_image_dataset]
for i in test_files:

    test_image_dataset = tf.data.TFRecordDataset(i)



    # Create a dictionary describing the features.

    test_feature_description = {

        'id': tf.io.FixedLenFeature([], tf.string),

        'image': tf.io.FixedLenFeature([], tf.string),

    }



def _parse_image_function(example_proto):

  # Parse the input tf.Example proto using the dictionary above.

  return tf.io.parse_single_example(example_proto, test_feature_description)



test_image_dataset = test_image_dataset.map(_parse_image_function)





test_ids = [str(image_features['id'].numpy())[2:-1] for image_features in test_image_dataset]



test_images = [image_features['image'].numpy() for image_features in test_image_dataset]
# dry run for testing



import IPython.display as display



display.display(display.Image(data=train_images[200]))
# defining dataset

from PIL import Image

import cv2

import albumentations

import torch

import numpy as np

import io

from torch.utils.data import Dataset



class FlowerDataset(Dataset):

    def __init__(self, id , classes , image , img_height , img_width, mean , std , is_valid):

        self.id = id

        self.classes = classes

        self.image = image

        self.is_valid = is_valid

        if self.is_valid == 1:

            self.aug = albumentations.Compose([

               albumentations.Resize(img_height , img_width, always_apply = True) ,

               albumentations.Normalize(mean , std , always_apply = True) 

            ])

        else:

            self.aug = albumentations.Compose([

                albumentations.Resize(img_height , img_width, always_apply = True) ,

                albumentations.Normalize(mean , std , always_apply = True),

                albumentations.ShiftScaleRotate(shift_limit = 0.0625,

                                                scale_limit = 0.1 ,

                                                rotate_limit = 5,

                                                p = 0.9)

            ]) 

        

    def __len__(self):

        return len(self.id)

    

    def __getitem__(self, index):

        id = self.id[index]

        img = np.array(Image.open(io.BytesIO(self.image[index]))) 

        img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

        img = self.aug(image = img)['image']

        img = np.transpose(img , (2,0,1)).astype(np.float32)

       

        

#         return {

#             'image' : torch.tensor(img, dtype = torch.float),

#             'class' : torch.tensor(self.classes[index], dtype = torch.long) 

#         }



#         if self.is_valid == 1:

#             return torch.tensor(img, dtype = torch.float),np.eye(104, dtype='float64')[int(self.classes[index])] # 104 is the no. of classes

#         else:

        return torch.tensor(img, dtype = torch.float),torch.tensor(self.classes[index], dtype = torch.long)
# sanity check for FlowerDataset class created



train_dataset = FlowerDataset(id = train_ids, classes = train_class, image = train_images, 

                        img_height = 224 , img_width = 224, 

                        mean = (0.485, 0.456, 0.406),

                        std = (0.229, 0.224, 0.225) , is_valid = 0)



val_dataset = FlowerDataset(id = val_ids, classes = val_class, image = val_images, 

                        img_height = 224 , img_width = 224, 

                        mean = (0.485, 0.456, 0.406),

                        std = (0.229, 0.224, 0.225) , is_valid = 1)





import matplotlib.pyplot as plt

%matplotlib inline



idx = 200

img = train_dataset[idx][0]



print(train_dataset[idx][1])



npimg = img.numpy()

plt.imshow(np.transpose(npimg, (1,2,0)))
# setting up the dataloader with cutmix data agumentation

!pip install git+https://github.com/ildoonet/cutmix
# setting up the train data loader



from cutmix.cutmix import CutMix



train_dataloader = CutMix(train_dataset, 

                          num_class=104, 

                          beta=1.0, 

                          prob=0.5, 

                          num_mix=2)
# setting up the validation data loader



from torch.utils.data import DataLoader



training_dataloader = DataLoader(train_dataset,

                        shuffle=True,

                        num_workers=4,

                        batch_size=1

                       )



val_dataloader = DataLoader(val_dataset,

                        shuffle=False,

                        num_workers=4,

                        batch_size=1

                       )
# keeping the train and validation data loaders in one dictionary

dataloaders = {

    'train': training_dataloader ,

    'val': val_dataloader

}



dataset_sizes = {

    'train': len(train_dataset) ,

    'val': len(val_dataset)

}
y = next(iter(dataloaders['train']))

y
# downloading the pretrained model - efficientnet b7  



!pip install efficientnet_pytorch



import efficientnet_pytorch



model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b7')
# changing the last layer from 1000 category classifier to 104 flower's catergory classifier 



in_features = model._fc.in_features

model._fc = torch.nn.Linear(in_features, 104)
model
!pip install pretrainedmodels
import pretrainedmodels

model_name = 'resnet18' # could be fbresnet152 or inceptionresnetv2

model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
model
in_features = model.last_linear.in_features

model.last_linear = torch.nn.Linear(in_features, 104)
model
# Printing out the model to see the changes in made in last layer

# device = xm.xla_device()

# model = model.to(device)

if torch.cuda.is_available():

    model.cuda()
# installing torchcontrib for Stochastic Weight Averaging in PyTorch 

!pip install torchcontrib
!pip install torchtoolbox

from torchtoolbox.tools import mixup_data, mixup_criterion
# setting up the optimizer , loss func. & scheduler for training



# from cutmix.utils import CutMixCrossEntropyLoss 



#for Stochastic Weight Averaging in PyTorch

from torchcontrib.optim import SWA



base_optimizer = torch.optim.Adam(model._fc.parameters(), lr=1e-4)



optimizer = SWA(base_optimizer, swa_start=5, swa_freq=5, swa_lr=0.05)



# loss_fn = CutMixCrossEntropyLoss(True)

loss_fn = torch.nn.CrossEntropyLoss()



scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# setting up the training function



if __name__ == "__main__":



    for param in model.parameters():

        param.requires_grad = False

        

    for param in model._fc.parameters():

        param.requires_grad = True



    epochs = 25



    for epoch in range(epochs):

        print('Epoch ', epoch,'/',epochs-1)

        print('-'*15)



        for phase in ['train', 'val']:

            if phase == 'train':

                model.train()  # Set model to training mode

            else:

                model.eval()   # Set model to evaluate mode



            running_loss = 0.0

            running_corrects = 0.0

            

            alpha = 0.2

            

            # Iterate over data.

            for i,(inputs,labels) in enumerate(dataloaders[phase]):

                if torch.cuda.is_available():

                    inputs = inputs.cuda()

                    labels = labels.cuda()

                    

                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha)



                # zero the parameter gradients

                optimizer.zero_grad()



                with torch.set_grad_enabled(phase == 'train'):

                    

                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss = mixup_criterion(loss_fn, outputs, labels_a, labels_b, lam)



                    # loss = loss_fn(outputs,labels)



                    # we backpropagate to set our learning parameters only in training mode

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()



                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data) # (preds == labels.data) as the usage of .data is not recommended, as it might have unwanted side effect.



            # scheduler for weight decay

            if phase == 'train':

                scheduler.step()



            epoch_loss = running_loss / float(dataset_sizes[phase])

            epoch_acc = running_corrects / float(dataset_sizes[phase])



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    optimizer.swap_swa_sgd()       