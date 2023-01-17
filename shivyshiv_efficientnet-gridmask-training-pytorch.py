!pip install git+https://github.com/qubvel/segmentation_models

!pip install iterative-stratification

!pip install pretrainedmodels

!pip install --upgrade efficientnet-pytorch
import sys

import pretrainedmodels



import glob

import torch

import albumentations

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

from tqdm import tqdm

from PIL import Image

import joblib

import torch.nn as nn

from torch.nn import functional as F

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from efficientnet_pytorch import EfficientNet
DIR_INPUT='../input/bengaliai-cv19/'
# df = pd.read_csv(DIR_INPUT+"train.csv")

# print(df.head())

# df.loc[:,'kfold']=-1

# df = df.sample(frac=1).reset_index(drop=True)

# X = df.image_id.values

# y = df[['grapheme_root','vowel_diacritic','consonant_diacritic']].values

# mskf = MultilabelStratifiedKFold(n_splits=5)

# for fold, (trn_,val_) in enumerate(mskf.split(X,y)):

#   print("TRAIN: ",trn_,"VAL: ",val_)

#   df.loc[val_,"kfold"] = fold

# print(df.kfold.value_counts())

# df.to_csv(DIR_INPUT+"/train_folds.csv")
# files = glob.glob("/content/train_*.parquet")

# for f in files:

#   df = pd.read_parquet(f)

#   image_ids = df.image_id.values

#   df = df.drop("image_id", axis=1)

#   image_array = df.values

#   for j, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):

#       joblib.dump(image_array[j, :], f"/content/image_pickles/{image_id}.pkl")
import pandas as pd



import torch



from PIL import Image

from albumentations.core.transforms_interface import ImageOnlyTransform

from albumentations.augmentations import functional as F1

from PIL import Image, ImageOps, ImageEnhance



from albumentations.core.transforms_interface import DualTransform



class GridMask(DualTransform):

    """GridMask augmentation for image classification and object detection.



    Args:

        num_grid (int): number of grid in a row or column.

        fill_value (int, float, lisf of int, list of float): value for dropped pixels.

        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int

            an angle is picked from (-rotate, rotate). Default: (-90, 90)

        mode (int):

            0 - cropout a quarter of the square of each grid (left top)

            1 - reserve a quarter of the square of each grid (left top)

            2 - cropout 2 quarter of the square of each grid (left top & right bottom)



    Targets:

        image, mask



    Image types:

        uint8, float32



    Reference:

    |  https://arxiv.org/abs/2001.04086

    |  https://github.com/akuxcw/GridMask

    """



    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):

        super(GridMask, self).__init__(always_apply, p)

        if isinstance(num_grid, int):

            num_grid = (num_grid, num_grid)

        if isinstance(rotate, int):

            rotate = (-rotate, rotate)

        self.num_grid = num_grid

        self.fill_value = fill_value

        self.rotate = rotate

        self.mode = mode

        self.masks = None

        self.rand_h_max = []

        self.rand_w_max = []



    def init_masks(self, height, width):

        if self.masks is None:

            self.masks = []

            n_masks = self.num_grid[1] - self.num_grid[0] + 1

            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):

                grid_h = height / n_g

                grid_w = width / n_g

                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)

                for i in range(n_g + 1):

                    for j in range(n_g + 1):

                        this_mask[

                             int(i * grid_h) : int(i * grid_h + grid_h / 2),

                             int(j * grid_w) : int(j * grid_w + grid_w / 2)

                        ] = self.fill_value

                        if self.mode == 2:

                            this_mask[

                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),

                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)

                            ] = self.fill_value

                

                if self.mode == 1:

                    this_mask = 1 - this_mask



                self.masks.append(this_mask)

                self.rand_h_max.append(grid_h)

                self.rand_w_max.append(grid_w)



    def apply(self, image, mask, rand_h, rand_w, angle, **params):

        h, w = image.shape[:2]

        mask = F1.rotate(mask, angle) if self.rotate[1] > 0 else mask

        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask

        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)

        return image



    def get_params_dependent_on_targets(self, params):

        img = params['image']

        height, width = img.shape[:2]

        self.init_masks(height, width)



        mid = np.random.randint(len(self.masks))

        mask = self.masks[mid]

        rand_h = np.random.randint(self.rand_h_max[mid])

        rand_w = np.random.randint(self.rand_w_max[mid])

        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0



        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}



    @property

    def targets_as_params(self):

        return ['image']



    def get_transform_init_args_names(self):

        return ('num_grid', 'fill_value', 'rotate', 'mode')

from PIL import Image



class BengaliDatasetTrain:

    def __init__(self, folds, img_height, img_width, mean, std):

        df = pd.read_csv("../input/train-data/train_folds.csv")

        df = df[["image_id", "grapheme_root", "vowel_diacritic", "consonant_diacritic", "kfold"]]



        df = df[df.kfold.isin(folds)].reset_index(drop=True)

        

        self.image_ids = df.image_id.values

        self.grapheme_root = df.grapheme_root.values

        self.vowel_diacritic = df.vowel_diacritic.values

        self.consonant_diacritic = df.consonant_diacritic.values



        if len(folds) == 1:

            self.aug = albumentations.Compose([

                albumentations.Resize(img_height, img_width, always_apply=True),

                albumentations.Normalize(mean, std, always_apply=True)

            ])

        else:

             self.aug = albumentations.Compose([

                albumentations.Resize(img_height, img_width, always_apply=True),

                # albumentations.ShiftScaleRotate(shift_limit=0.0625,

                #                                scale_limit=0.1, 

                #                                rotate_limit=5,

                #                                p=0.9),

                albumentations.Normalize(mean, std, always_apply=True),

                albumentations.OneOf([

                    GridMask(num_grid=3, mode=0, rotate=15),

                    GridMask(num_grid=3, mode=2, rotate=15),

                ], p=0.75)

            ])







    def __len__(self):

        return len(self.image_ids)

    

    def __getitem__(self, item):

        image = joblib.load(f"/content/image_pickles/{self.image_ids[item]}.pkl")

        image = image.reshape(137, 236).astype(float)

        image = Image.fromarray(image).convert("RGB")

        image = self.aug(image=np.array(image))["image"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)



        return {

            "image": torch.tensor(image, dtype=torch.float),

            "grapheme_root": torch.tensor(self.grapheme_root[item], dtype=torch.long),

            "vowel_diacritic": torch.tensor(self.vowel_diacritic[item], dtype=torch.long),

            "consonant_diacritic": torch.tensor(self.consonant_diacritic[item], dtype=torch.long)

        }
class EfficientNetB1(nn.Module):

    def __init__(self, pretrained):

        super(EfficientNetB1, self).__init__()



        if pretrained is True:

            self.model = EfficientNet.from_pretrained("efficientnet-b1")

        else:

            self.model = EfficientNet.from_name('efficientnet-b1')

        

        self.l0 = nn.Linear(1280, 168)

        self.l1 = nn.Linear(1280, 11)

        self.l2 = nn.Linear(1280, 7)



    def forward(self, x):

        bs, _, _, _ = x.shape

        x = self.model.extract_features(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)

        l0 = self.l0(x)

        l1 = self.l1(x)

        l2 = self.l2(x)



        return l0, l1, l2



MODEL_DISPATCHER = {

    "efficientNetb1": EfficientNetB1

}  
import numpy as np

import torch



class EarlyStopping:

    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):

        """

        Args:

            patience (int): How long to wait after last time validation loss improved.

                            Default: 7

            verbose (bool): If True, prints a message for each validation loss improvement. 

                            Default: False

            delta (float): Minimum change in the monitored quantity to qualify as an improvement.

                            Default: 0

        """

        self.patience = patience

        self.verbose = verbose

        self.counter = 0

        self.best_score = None

        self.early_stop = False

        self.val_loss_min = np.Inf

        self.delta = delta



    def __call__(self, val_loss, model):



        score = -val_loss



        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:

            self.counter += 1

            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            self.save_checkpoint(val_loss, model)

            self.counter = 0



    def save_checkpoint(self, val_loss, model):

        '''Saves model when validation loss decrease.'''

        if self.verbose:

            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), 'checkpoint.pt')

        self.val_loss_min = val_loss


import os

import ast

import torch

import torch.nn as nn

import numpy as np

import sklearn.metrics

from tqdm import tqdm



DEVICE = "cuda"

IMG_HEIGHT=224

IMG_WIDTH=224

EPOCHS=20

TRAIN_BATCH_SIZE=64

TEST_BATCH_SIZE=64

MODEL_MEAN=[0.485, 0.456, 0.406]

MODEL_STD=[0.229, 0.224, 0.225]

BASE_MODEL="efficientNetb1"

TRAINING_FOLDS_CSV="../input/train-data/train_folds.csv"



TRAINING_FOLDS=[0,1,2,3]

VALIDATION_FOLDS=[4]







def macro_recall(pred_y, y, n_grapheme=168, n_vowel=11, n_consonant=7):

    

    pred_y = torch.split(pred_y, [n_grapheme, n_vowel, n_consonant], dim=1)

    pred_labels = [torch.argmax(py, dim=1).cpu().numpy() for py in pred_y]



    y = y.cpu().numpy()



    recall_grapheme = sklearn.metrics.recall_score(pred_labels[0], y[:, 0], average='macro')

    recall_vowel = sklearn.metrics.recall_score(pred_labels[1], y[:, 1], average='macro')

    recall_consonant = sklearn.metrics.recall_score(pred_labels[2], y[:, 2], average='macro')

    scores = [recall_grapheme, recall_vowel, recall_consonant]

    final_score = np.average(scores, weights=[2, 1, 1])

    print(f'recall: grapheme {recall_grapheme}, vowel {recall_vowel}, consonant {recall_consonant}, 'f'total {final_score}, y {y.shape}')

    

    return final_score





def loss_fn(outputs, targets):

    o1, o2, o3 = outputs

    t1, t2, t3 = targets

    l1 = nn.CrossEntropyLoss()(o1, t1)

    l2 = nn.CrossEntropyLoss()(o2, t2)

    l3 = nn.CrossEntropyLoss()(o3, t3)

    return (l1 + l2 + l3) / 3







def train(dataset, data_loader, model, optimizer):

    model.train()

    final_loss = 0

    counter = 0

    final_outputs = []

    final_targets = []



    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset)/data_loader.batch_size)):

        counter = counter + 1

        image = d["image"]

        grapheme_root = d["grapheme_root"]

        vowel_diacritic = d["vowel_diacritic"]

        consonant_diacritic = d["consonant_diacritic"]



        image = image.to(DEVICE, dtype=torch.float)

        grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)

        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)

        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)

        

        print(image.shape)



        optimizer.zero_grad()

        outputs = model(image)

        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)

        loss = loss_fn(outputs, targets)



        loss.backward()

        optimizer.step()



        final_loss += loss



        o1, o2, o3 = outputs

        t1, t2, t3 = targets

        final_outputs.append(torch.cat((o1,o2,o3), dim=1))

        final_targets.append(torch.stack((t1,t2,t3), dim=1))



        #if bi % 10 == 0:

        #    break

    final_outputs = torch.cat(final_outputs)

    final_targets = torch.cat(final_targets)



    print("=================Train=================")

    macro_recall_score = macro_recall(final_outputs, final_targets)

    

    return final_loss/counter , macro_recall_score







def evaluate(dataset, data_loader, model):

    with torch.no_grad():

        model.eval()

        final_loss = 0

        counter = 0

        final_outputs = []

        final_targets = []

        for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset)/data_loader.batch_size)):

            counter = counter + 1

            image = d["image"]

            grapheme_root = d["grapheme_root"]

            vowel_diacritic = d["vowel_diacritic"]

            consonant_diacritic = d["consonant_diacritic"]



            image = image.to(DEVICE, dtype=torch.float)

            grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)

            vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)

            consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)



            outputs = model(image)

            targets = (grapheme_root, vowel_diacritic, consonant_diacritic)

            loss = loss_fn(outputs, targets)

            final_loss += loss



            o1, o2, o3 = outputs

            t1, t2, t3 = targets

            #print(t1.shape)

            final_outputs.append(torch.cat((o1,o2,o3), dim=1))

            final_targets.append(torch.stack((t1,t2,t3), dim=1))

        

        final_outputs = torch.cat(final_outputs)

        final_targets = torch.cat(final_targets)



        print("=================Train=================")

        macro_recall_score = macro_recall(final_outputs, final_targets)



    return final_loss/counter , macro_recall_score







def main():

    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)

    model.to(DEVICE)



    train_dataset = BengaliDatasetTrain(

        folds=TRAINING_FOLDS,

        img_height = IMG_HEIGHT,

        img_width = IMG_WIDTH,

        mean = MODEL_MEAN,

        std = MODEL_STD

    )



    train_loader = torch.utils.data.DataLoader(

        dataset=train_dataset,

        batch_size= TRAIN_BATCH_SIZE,

        shuffle=True,

        num_workers=4

    )



    valid_dataset = BengaliDatasetTrain(

        folds=VALIDATION_FOLDS,

        img_height = IMG_HEIGHT,

        img_width = IMG_WIDTH,

        mean = MODEL_MEAN,

        std = MODEL_STD

    )



    valid_loader = torch.utils.data.DataLoader(

        dataset=valid_dataset,

        batch_size= TEST_BATCH_SIZE,

        shuffle=True,

        num_workers=4

    )



    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 

                                                            mode="min", 

                                                            patience=5, 

                                                            factor=0.3,verbose=True)



    early_stopping = EarlyStopping(patience=5, verbose=True)



    #if torch.cuda.device_count() > 1:

    #    model = nn.DataParallel(model)



    best_score = -1



    print("FOLD : ", VALIDATION_FOLDS[0] )

    

    for epoch in range(1, EPOCHS+1):



        train_loss, train_score = train(train_dataset,train_loader, model, optimizer)

        val_loss, val_score = evaluate(valid_dataset, valid_loader, model)



        scheduler.step(val_loss)



        



        if val_score > best_score:

            best_score = val_score

            torch.save(model.state_dict(), f"{BASE_MODEL}_fold{VALIDATION_FOLDS[0]}.pth")



        epoch_len = len(str(EPOCHS))

        print_msg = (f'[{epoch:>{epoch_len}}/{EPOCHS:>{epoch_len}}] ' +

                     f'train_loss: {train_loss:.5f} ' +

                     f'train_score: {train_score:.5f} ' +

                     f'valid_loss: {val_loss:.5f} ' +

                     f'valid_score: {val_score:.5f}'

                    )

        

        print(print_msg)



        early_stopping(val_score, model)

        if early_stopping.early_stop:

            print("Early stopping")

            break

# TRAINING_FOLDS=[0,1,2,3]

# VALIDATION_FOLDS=[4]

# main()





# TRAINING_FOLDS=[0,1,2,4]

# VALIDATION_FOLDS=[3]

# main()



# TRAINING_FOLDS=[0,1,3,4]

# VALIDATION_FOLDS=[2]

# main()





# TRAINING_FOLDS=[0,2,3,4]

# VALIDATION_FOLDS=[1]

# main()



# TRAINING_FOLDS=[1,2,3,4]

# VALIDATION_FOLDS=[0]

# main()




