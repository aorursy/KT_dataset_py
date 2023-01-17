# Image size was made smaller for notebook use 
!ls ../input

!pip install pytorch-ignite
from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import Sampler

import torch

import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

import pathlib

import torchvision.transforms as transforms

import torch

import PIL

from sklearn.model_selection import train_test_split

import cv2

import sys

import pandas as pd

# from neptune import Context

from sklearn.metrics import f1_score

import torch.nn as nn

import torchvision.transforms as transforms

from torch import optim, save

from torch.utils.data import DataLoader

from torchvision.models import resnet50, resnet34, resnet18,resnext50_32x4d



from ignite.engine import Events

from ignite.engine import create_supervised_evaluator, create_supervised_trainer

from ignite.metrics import Recall, Precision

from ignite.metrics import Loss

import numpy as np

from PIL import Image

RANDOM_SEED = 666



LABEL_MAP = {

0: "Nucleoplasm" ,

1: "Nuclear membrane"   ,

2: "Nucleoli"   ,

3: "Nucleoli fibrillar center",   

4: "Nuclear speckles"   ,

5: "Nuclear bodies"   ,

6: "Endoplasmic reticulum"   ,

7: "Golgi apparatus"  ,

8: "Peroxisomes"   ,

9:  "Endosomes"   ,

10: "Lysosomes"   ,

11: "Intermediate filaments"  , 

12: "Actin filaments"   ,

13: "Focal adhesion sites"  ,

14: "Microtubules"   ,

15: "Microtubule ends"   ,

16: "Cytokinetic bridge"   ,

17: "Mitotic spindle"  ,

18: "Microtubule organizing center",  

19: "Centrosome",

20: "Lipid droplets"   ,

21: "Plasma membrane"  ,

22: "Cell junctions"   ,

23: "Mitochondria"   ,

24: "Aggresome"   ,

25: "Cytosol" ,

26: "Cytoplasmic bodies",

27: "Rods & rings"}











class MultiBandMultiLabelDataset(Dataset):

    BANDS_NAMES = ['_red.png','_green.png','_blue.png','_yellow.png']

    

    def __len__(self):

        return len(self.images_df)

    

    def __init__(self, images_df, 

                 base_path, 

                 image_transform, 

                 augmentator=None,

                 train_mode=True    

                ):

        if not isinstance(base_path, pathlib.Path):

            base_path = pathlib.Path(base_path)

            

        self.images_df = images_df.copy()

        self.image_transform = image_transform

        self.augmentator = augmentator

        self.images_df.Id = self.images_df.Id.apply(lambda x: base_path / x)

        self.mlb = MultiLabelBinarizer(classes=list(LABEL_MAP.keys()))

        self.train_mode = train_mode



                                      

        

    def __getitem__(self, index):

        y = None

        X = self._load_multiband_image(index)

        if self.train_mode:

            y = self._load_multilabel_target(index)

        

        # augmentator can be for instance imgaug augmentation object

        if self.augmentator is not None:

            X = self.augmentator(X)

            

        X = self.image_transform(X)

            

        return X, y 

    

    



    

    def _load_multiband_image(self, index):

        row = self.images_df.iloc[index]

        image_bands = []

        for band_name in self.BANDS_NAMES:

            p = str(row.Id.absolute()) + band_name

            pil_channel = PIL.Image.open(p)

            image_bands.append(pil_channel)

            

        # lets pretend its a RBGA image to support 4 channels

        band4image = PIL.Image.merge('RGBA', bands=image_bands)

        arr = np.array(band4image)

        arr = arr.astype('float')

        for i in range(4):

            minval = arr[...,i].min()

            maxval = arr[...,i].max()

            if minval != maxval:

                arr[...,i] -= minval

                arr[...,i] *= (255.0/(maxval-minval))

                

        new_img = Image.fromarray(arr.astype('uint8'),'RGBA')

        #return band4image

        return new_img

    

    def normalize(arr):

        """

        Linear normalization

        http://en.wikipedia.org/wiki/Normalization_%28image_processing%29

        """

        arr = arr.astype('float')

        for i in range(4):

            minval = arr[...,i].min()

            maxval = arr[...,i].max()

            if minval != maxval:

                arr[...,i] -= minval

                arr[...,i] *= (255.0/(maxval-minval))

        return arr

    

    

    def _load_multilabel_target(self, index):

        return list(map(int, self.images_df.iloc[index].Target.split(' ')))

    

        

    def collate_func(self, batch):

        labels = None

        images = [x[0] for x in batch]

        

        if self.train_mode:

            labels = [x[1] for x in batch]

            labels_one_hot  = self.mlb.fit_transform(labels)

            labels = torch.FloatTensor(labels_one_hot)

            

        

        return torch.stack(images)[:,:4,:,:], labels



def get_model(n_classes, image_channels=4):

    model = resnext50_32x4d(pretrained=True)

    for p in model.parameters():

        p.requires_grad = True

    inft = model.fc.in_features

    model.fc = nn.Linear(in_features=inft, out_features=n_classes)

    model.avgpool = nn.AdaptiveAvgPool2d(1)

    model.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3,

                               bias=False)

    

    return model  





def train(trainer, train_loader, test_loader, checkpoint_path='bestmodel_{}_{}.torch', epochs=1):

    @trainer.on(Events.ITERATION_COMPLETED)

    def log_training_loss(engine):

        iter = (engine.state.iteration - 1) % len(train_loader) + 1

#         ctx.channel_send('loss', engine.state.output)

        if iter % 25 == 0:

            print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"

                  "".format(engine.state.epoch, iter, len(train_loader), engine.state.output))



    @trainer.on(Events.EPOCH_COMPLETED)

    def log_training_results(engine):

        evaluator.run(test_loader)

        metrics = evaluator.state.metrics

        avg_nll = metrics['loss']

        print("Training Results - Epoch: {}  Avg loss: {:.2f}"

              .format(engine.state.epoch, avg_nll))

        save(model, checkpoint_path.format(engine.state.epoch, avg_nll))

    trainer.run(train_loader, max_epochs=epochs)

    

    return model 

    



# Eval

def evaluate(model, test_loader, threshold=0.2):

    all_preds = []

    true = []

    model.eval()

    for b in test_loader:

        X, y = b

        if torch.cuda.is_available():

            X, y = X.cuda(), y.cuda()

        pred = model(X)

        all_preds.append(pred.sigmoid().cpu().data.numpy())

        true.append(y.cpu().data.numpy())

        

        

    P = np.concatenate(all_preds)

    R = np.concatenate(true)

    

    f1 = f1_score(P>threshold, R, average='macro')

    print(f1)

    return f1

    



## Submission

def predict_submission(model, submission_load):

    all_preds = []

    model.eval()

    for i, b in enumerate(submission_load):

        if i % 100: print('processing batch {}/{}'.format(i, len(submission_load)))

        X, _ = b

        if torch.cuda.is_available():

            X = X.cuda()

        pred = model(X)

        all_preds.append(pred.sigmoid().cpu().data.numpy())

    return np.concatenate(all_preds)

        

         

def make_submission_file(sample_submission_df, predictions):

    submissions = []

    for row in predictions:

        subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))

        submissions.append(subrow)

    

    sample_submission_df['Predicted'] = submissions

    sample_submission_df.to_csv('submission.csv', index=None)

    

    return sample_submission_df



    

PATH_TO_IMAGES = '../input/train/'

PATH_TO_TEST_IMAGES = '../input/test/'

PATH_TO_META = '../input/train.csv'

SAMPLE_SUBMI = '../input/sample_submission.csv'

# Prepare dataframe files



SEED = 666

DEV_MODE = False

IMG_SIZE = 512

EPOCHS   = 20

if IMG_SIZE == 128: batch_size = 128; 

if IMG_SIZE == 256: batch_size = 32; 

if IMG_SIZE == 512: batch_size = 16; 

df = pd.read_csv(PATH_TO_META)

df_train, df_test  = train_test_split(df, test_size=0.2, random_state=SEED)

df_submission = pd.read_csv(SAMPLE_SUBMI)



if DEV_MODE:

    df_train = df_train[:200]

    df_test = df_test[:50]

    df_submission = df_submission[:50]



image_transform = transforms.Compose([

            transforms.Resize(IMG_SIZE),

            transforms.RandomChoice([

            transforms.RandomVerticalFlip(p=0.5),

            transforms.RandomHorizontalFlip(p=0.5),

            transforms.RandomAffine((0, 0),scale=(0.8,1.2)),

            transforms.RandomAffine((90, 90),scale=(0.8,1.2)),

            transforms.RandomAffine((180, 180),scale=(0.8,1.2)),

            transforms.RandomAffine((270, 270),scale=(0.8,1.2)),

            

    ]),

            transforms.ToTensor(),

    

        ])



 

# Prepare datasets and loaders

   

gtrain = MultiBandMultiLabelDataset(df_train, base_path=PATH_TO_IMAGES, image_transform=image_transform)

gtest = MultiBandMultiLabelDataset(df_test, base_path=PATH_TO_IMAGES, image_transform=image_transform)

gsub = MultiBandMultiLabelDataset(df_submission, base_path=PATH_TO_TEST_IMAGES, train_mode=False, image_transform=image_transform)



train_load = DataLoader(gtrain, collate_fn=gtrain.collate_func, batch_size=batch_size, num_workers=6)

test_load = DataLoader(gtest, collate_fn=gtest.collate_func, batch_size=batch_size, num_workers=6)

submission_load = DataLoader(gsub, collate_fn=gsub.collate_func, batch_size=8, num_workers=6)





# Prepare model 

with torch.cuda.device(1):

    model = get_model(28,4)

    device='cuda'

    criterion = nn.BCEWithLogitsLoss()

    

    if torch.cuda.is_available():

        criterion = criterion.cuda(1)

    evaluator = create_supervised_evaluator(model,

                                            device=device,

                                            metrics={'loss': Loss(criterion)

                                                    })

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=0.0001)#lr=0.00005

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
# train the model

with torch.cuda.device(1):

    model = train(trainer, train_load, test_load, epochs=EPOCHS)
# evaluate on testing data and calculate F1-macro

#bestmodel_10_0.22161348430456312.torch

# model = get_model(28,4)

import gc

gc.collect()    

test_load = DataLoader(gtest, collate_fn=gtest.collate_func, batch_size=4, num_workers=6)

with torch.cuda.device(1):

    res = evaluate(model, test_load, threshold=0.2)

    print(res)
with torch.cuda.device(1):

    submission_predictions =predict_submission(model, submission_load)
# prepare the submission file and 

with torch.cuda.device(1):

    THRESHOLD = 0.2

    p = submission_predictions>THRESHOLD



    submission_file = make_submission_file(sample_submission_df=df_submission,

                      predictions=p)
submission_file.head()
import jovian

jovian.commit()