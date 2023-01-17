!pip install "../input/pretrainedmodelspytorch/" > /dev/null

!pip install "../input/segmentation-models-dataset" > /dev/null

!pip install -U catalyst
import torch, catalyst



torch.__version__, catalyst.__version__
import os

import time



import numpy as np

import pandas as pd



import cv2



import segmentation_models_pytorch as smp

import albumentations as albu



from sklearn.model_selection import train_test_split



import torch



from torch.utils.data import Dataset, DataLoader



import matplotlib.pyplot as plt
ENCODER = 'se_resnext50_32x4d'

# ENCODER = 'inceptionresnetv2'

# ENCODER = 'dpn98'

ENCODER_WEIGHTS = 'imagenet'

DEVICE = "cuda"

ACTIVATION = 'sigmoid'



device = torch.device(DEVICE)

num_epochs = 10



logdir = "../logs/segmentation"
def get_img(x, folder: str='train_images'):

    """

    Return image based on image name and folder.

    """

    data_folder = f"{path}/{folder}"

    image_path = os.path.join(data_folder, x)

    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img





def rle_decode(mask_rle: str = '', shape: tuple = (256, 1600)):

    '''

    Decode rle encoded mask.

    

    :param mask_rle: run-length as string formatted (start length)

    :param shape: (height, width) of array to return 

    Returns numpy array, 1 - mask, 0 - background

    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape, order='F')





def make_mask(df, row_id, shape: tuple = (256, 1600)):

    """

    Create mask based on df, image name and shape.

    """

    encoded_masks = df.iloc[row_id]

    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)



    for idx, label in enumerate(encoded_masks.values):

        if label is not np.nan:

            mask = rle_decode(label)

            masks[:, :, idx] = mask

            

    return masks





def to_tensor(x, **kwargs):

    """

    Convert image or mask.

    """

    return x.transpose(2, 0, 1).astype('float32')





def mask2rle(img):

    '''

    Convert mask to rle.

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



        

def post_process(probability, threshold, min_size):

    """

    Post processing of each predicted mask, components with lesser number of pixels

    than `min_size` are ignored

    """

    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]

    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))

    predictions = np.zeros((350, 525), np.float32)

    num = 0

    for c in range(1, num_component):

        p = (component == c)

        if p.sum() > min_size:

            predictions[p] = 1

            num += 1

    return predictions, num





def get_training_augmentation():

    train_transform = [



        albu.HorizontalFlip(p=0.5),

        # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),

        # albu.GridDistortion(p=0.5),

        # albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),

        # albu.Resize(320, 640)

    ]

    return albu.Compose(train_transform)





def get_validation_augmentation():

    """Add paddings to make image shape divisible by 32"""

    test_transform = [

        albu.HorizontalFlip(p=0.0)

    ]

    return albu.Compose(test_transform)





def get_preprocessing(preprocessing_fn):

    """Construct preprocessing transform

    

    Args:

        preprocessing_fn (callbale): data normalization function 

            (can be specific for each pretrained neural network)

    Return:

        transform: albumentations.Compose

    

    """

    

    _transform = [

        albu.Lambda(image=preprocessing_fn),

        albu.Lambda(image=to_tensor, mask=to_tensor),

    ]

    return albu.Compose(_transform)





def dice(img1, img2):

    img1 = np.asarray(img1).astype(np.bool)

    img2 = np.asarray(img2).astype(np.bool)



    intersection = np.logical_and(img1, img2)



    return 2. * intersection.sum() / (img1.sum() + img2.sum())
class SteelDefectionDataset(Dataset):

    def __init__(self, df, datatype: str = 'train', transforms = None, preprocessing=None):

        self.df = df

        

        self.img_ids = list(df.index)

        

        if datatype != 'test':

            self.data_folder = "../input/severstal-steel-defect-detection/train_images/"

        else:

            self.data_folder = "../input/severstal-steel-defect-detection/test_images"

        

        self.transforms = transforms

        self.preprocessing = preprocessing



    def __getitem__(self, idx):

        image_name = self.img_ids[idx]

        mask = make_mask(self.df, idx)

        image_path = os.path.join(self.data_folder, image_name)

        img = cv2.imread(image_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = self.transforms(image=img, mask=mask)

        img = augmented['image']

        mask = augmented['mask']

        

        if self.preprocessing:

            preprocessed = self.preprocessing(image=img, mask=mask)

            return preprocessed

        else:

            return augmented



    def __len__(self):

        return len(self.img_ids)
df = pd.read_csv("../input/severstal-steel-defect-detection/train.csv")

# https://www.kaggle.com/amanooo/defect-detection-starter-u-net

df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))

df['ClassId'] = df['ClassId'].astype(int)

df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')

df['defects'] = df.count(axis=1)

df = df[df['defects'] > 0]



train, valid = train_test_split(df, test_size=0.2, stratify=df["defects"], random_state=69)



train = train.drop(columns=["defects"])

valid = valid.drop(columns=["defects"])



preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)



train_dataset = SteelDefectionDataset(df=train, datatype='train', 

                                      transforms = get_training_augmentation(),

                                      preprocessing=get_preprocessing(preprocessing_fn))



valid_dataset = SteelDefectionDataset(df=valid, datatype='val', 

                                      transforms = get_validation_augmentation(),

                                      preprocessing=get_preprocessing(preprocessing_fn))



train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)



loaders = {

    "train": train_loader,

    "valid": valid_loader

}
to_show_dataset = SteelDefectionDataset(df=train, datatype='train',

                                       transforms=get_training_augmentation())



for ii in range(5):

    idx = np.random.randint(len(to_show_dataset))

    sample =  to_show_dataset[idx]

    image, masks = sample['image'], sample['mask']

    f, ax = plt.subplots(1, 5, figsize=(24, 24))



    ax[0].imshow(image)

    for i in range(4):

        ax[i + 1].imshow(masks[:, :, i])

        ax[i + 1].set_title(f'Mask {i}', fontsize=4)
from torch import nn



from torch.optim.lr_scheduler import ReduceLROnPlateau



from catalyst.contrib.criterion import DiceLoss, IoULoss

from catalyst.dl import SupervisedRunner





model = torch.load("../input/fpn-se-resnet50-epoch20/fpn_se_resnext50_32x4d.pth")

# model = smp.FPN(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,  classes=4, activation=ACTIVATION)

# model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,  classes=4, activation=ACTIVATION)



criterion = {

    "dice": DiceLoss(),

    "iou": IoULoss(),

    "bce": nn.BCEWithLogitsLoss()

}



# Create optimizer

optimizer = torch.optim.Adam([

    {'params': model.decoder.parameters(), 'lr': 1e-5}, 

    

    # decrease lr for encoder in order not to permute 

    # pre-trained weights with large gradients on training start

    {'params': model.encoder.parameters(), 'lr': 1e-6},  

])





scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True)



runner = SupervisedRunner(device=device, input_key="image", input_target_key="mask")
from catalyst.dl.callbacks import DiceCallback, IouCallback, CriterionCallback, CriterionAggregatorCallback





runner.train(

    model=model,

    criterion=criterion,

    optimizer=optimizer,

    scheduler=scheduler,

    

    # our dataloaders

    loaders=loaders,

    

    callbacks=[

        # Each criterion is calculated separately.

        CriterionCallback(

            input_key="mask",

            prefix="loss_dice",

            criterion_key="dice"

        ),

        CriterionCallback(

            input_key="mask",

            prefix="loss_iou",

            criterion_key="iou"

        ),

        CriterionCallback(

            input_key="mask",

            prefix="loss_bce",

            criterion_key="bce",

            multiplier=0.8

        ),

        

        # And only then we aggregate everything into one loss.

        CriterionAggregatorCallback(

            prefix="loss",

            loss_keys=["loss_dice", "loss_iou", "loss_bce"],

            loss_aggregate_fn="sum" # or "mean"

        ),

        

        # metrics

        DiceCallback(input_key="mask"),

        IouCallback(input_key="mask"),

    ],

    

    # path to save logs

    logdir=logdir,

    

    num_epochs=num_epochs,

    

    # save our best checkpoint by IoU metric

    main_metric="iou",

    # IoU needs to be maximized.

    minimize_metric=False,  

    # prints train logs

    verbose=True

)



torch.save(model, './best_model.pth')