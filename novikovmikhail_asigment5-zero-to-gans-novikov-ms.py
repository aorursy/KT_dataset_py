import os
import time
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np
import cv2
from albumentations import HorizontalFlip, VerticalFlip, RandomBrightness,  ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise, ElasticTransform
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
!pip install segmentation_models_pytorch
import segmentation_models_pytorch as smp # https://github.com/qubvel/segmentation_models.pytorch
%matplotlib inline
# Exploring the Data
DATA_DIR = '../input/severstal-steel-defect-detection'

TRAIN_IMG_DIR = DATA_DIR + '/train_images'                    # Contains training images
TEST_IMG_DIR = DATA_DIR + '/test_images'                      # Contains test images

TRAIN_CSV = DATA_DIR + '/train.csv'                       # Contains real labels for training images
TEST_CSV = DATA_DIR + '/sample_submission.csv'            # Contains dummy labels for test image
TRAINED_WEIGHTS = './model.pth'     # Contains trained weights
SUBMISSION_FILE = './submission.csv'     # Contains trined labels for test image
PRE_TRAINED_WEIGHTS = '../input/asigment5-zero-to-gans-novikov-ms-model/model.pth'



train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
train_df.head()
len(train_df)
test_df.head()
def survey(df_in: pd.DataFrame):
    """
    Counts masks and classes and returns df with columns ('maskCount', 'ImageId')
    """
    df_maskCnt = pd.DataFrame({'maskCount' : df_in.groupby('ImageId').size()})
    df_out = pd.merge(df_in, df_maskCnt, on='ImageId')
    df_out = df_out.sort_values(by=['maskCount', 'ImageId'], ascending=False)

    df_out['ClassIds'] = pd.Series(dtype=object)

    for i, row_i in df_out.iterrows():
        ClassId_list = []
        for j, row_j in df_out.loc[df_out['ImageId'] == row_i['ImageId']].iterrows():
            ClassId_list.append(row_j['ClassId'])
        df_out.at[i,'ClassIds'] = ClassId_list
    # print(df_out.head())    
    return df_out

df = survey(train_df)
df.head(15)
def counter_func(df_in):
    """
    Returns total and list of num classId from df
    """
    length = 4
    counter = np.zeros(length, dtype=int)
    total = 0
    for i in range(length):
        try:
            index = class_id2index(df_in.index[i])
            counter[index] = df_in.iloc[i, 0]  
        except:
            continue
        
        
    total = counter.sum()
    return total, counter 
mask_count_df_pivot = pd.DataFrame({'ClassCount' : df.groupby('ImageId').size()})
mask_count_df_pivot = pd.DataFrame({'Num' : mask_count_df_pivot.groupby('ClassCount').size()})
mask_count_df_pivot.sort_values('ClassCount', ascending=True, inplace=True)
ClassId_count_df = df.set_index(["ImageId", "ClassId"]).count(level='ClassId')

total, counter = counter_func(ClassId_count_df)
print('Total strings: {0}, 1 class: {1}, 2 class: {2}, 3 class: {3}, 4 class: {4}'.format(total, *counter))
total, counter = counter_func(mask_count_df_pivot)
print('Total images: {0}, one class: {1}, two classes: {2}, three classes: {3}, four classes: {4}'.format(total, *counter))
def class_id2index(val: int):
    """ converts ClassId to index in masks"""
    return int(val-1)

def index2class_id(val: int):
    """ converts index to ClassId in masks"""
    return int(val+1)
# https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask2rle(img: np.array):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated (start length)
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = ' '.join(str(x) for x in runs)
    if rle == np.nan:
        return ''
    else:
        return rle #returns a string formated (start length)

def rle2mask(mask_rle: str, input_shape: Tuple[int, int, int]=(256,1600,1)):
    
    """    
    img
    The pixel order of the line is from top to bottom from the left vertical line.
     It must be made and handed over.
     width/height should be made [height, width, ...] to fit the rows and columns.

     example when width=4, height=3
    
    s = [1,2,3,4,5,6,7,8,9,10,11,12]
        => 1,2,3 First row on the left, second row on 4,5,6
    
    mask_rle: run-length as string formated (start length)
    shape: (height,width)!!!  of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    
    # print('>>> rle2mask rle2mask: ', mask_rle)
    height, width = input_shape[:2]

    """
    RLE
    Is a repetition of (start point, length), so divide it by even/odd and start point array
     Create an array of lengths.
     s[1:]: from 1 to the end
     s[1:][::2]: Skip 2 by s[1:] to get an array of extracted values.
    """

    mask = np.zeros(width * height, dtype=np.uint8)
    if mask_rle is not np.nan:
        s = mask_rle.split()
        array = np.asarray([int(x) for x in s])
        starts = array[0::2]
        lengths = array[1::2]

        for index, start in enumerate(starts):
            begin = int(start - 1)
            end = int(begin + lengths[index])
            mask[begin : end] = 1

    """
    img
    The pixel order of the line is from top to bottom from the left vertical line.
     It must be made and handed over.
     width/height should be made [height, width, ...] to fit the rows and columns.

     ex) When width=4, height=3

    s = [1,2,3,4,5,6,7,8,9,10,11,12]
        => 1,2,3 First row on the left, second row on 4,5,6

    s.reshape(4,3) :
    [[ 1  2  3]
     [ 4  5  6]
     [ 7  8  9]
     [10 11 12]]

    s.reshape(4,3).T :
    [[ 1  4  7 10]
     [ 2  5  8 11]
     [ 3  6  9 12]]
    """

    rle_mask = mask.reshape(width, height).T
    # print('>>> rle2mask mask.shape: ', rle_mask.shape)
    # print('>>> rle2mask mask: ', rle_mask)
    return rle_mask

# Test RLE functions
assert mask2rle(rle2mask(df['EncodedPixels'].iloc[0]))==df['EncodedPixels'].iloc[0]
assert mask2rle(rle2mask('1 1'))=='1 1'


def build_masks(rle_labels: pd.DataFrame, input_shape: Tuple[int, int, int]=(256, 1600, 4)):
    """

    :param rle_labels: pd.DataFrame ImageId, ClassId, RLE
    :param input_shape: (height, width) of array to return
    :return: masks #(256, 1600, 4)
    """
    masks = np.zeros(input_shape)
    # print('>>> build_masks')
    # print('>>> rle_labels.head()', rle_labels.head())
    for _, val in rle_labels.iterrows():
        # print('>>> build_masks: val[ImageId] :', val['ImageId'])
        # print('>>> build_masks: val[ClassId] :', val['ClassId'])
        masks[:, :, class_id2index(val['ClassId'])] = rle2mask(val['EncodedPixels'], input_shape)
        # print(">>> build_masks: masks ", masks.shape)
        # print('<<< build_masks')
    return masks #(256, 1600, 4)

def make_mask(row_id_in: int, df_in: pd.DataFrame,  input_shape_in: Tuple[int, int, int] = (256, 1600, 4)):
    """
    Given a row index, dataframe, shape, return image_id and mask (256, 1600, 4) from the dataframe `df`
    :param row_id_in: int
    :param df_in: pd.DatFrame
    :param input_shape_in: Tuple[int] = (256, 1600, 4))
    :return: fname, masks
    """
    fname = df_in.iloc[row_id_in].ImageId
    # print('>>>make_mask row_id_in: ', row_id_in, ' fname: ', fname)
    rle_labels = df_in[df_in['ImageId'] == fname][['ImageId', 'ClassId', 'EncodedPixels']]
    # print('>>>make_mask rle_labels', rle_labels)
    masks = build_masks(rle_labels, input_shape=input_shape_in) #(256, 1600, 4)
    # print('>>>make_mask masks: {0}'.format(masks))
    return fname, masks
assert mask2rle(np.zeros((256, 1600), np.float32)) == ''
def show_images(df_in: pd.DataFrame, img_dir: str, trained_df_in: pd.DataFrame = None):
    """
    Shows some images with most valid masks as possible and their masks
    If there is an a df to compare - shows also mask from this df
    :param df_in:
    :param img_dir:
    :param trained_df_in:
    """
    local_df = df_in
    local_trained_df = trained_df_in
    columns = 1
    if type(trained_df_in) == pd.DataFrame:
        rows = 15
    else:
        rows = 10
    
    
    fig = plt.figure(figsize=(20,80))
    
    def sorter(local_df):
        local_df = local_df.sort_values(by=['maskCount', 'ImageId'], ascending=False) # To show as many valid masks as possible
        grp = local_df['ImageId'].drop_duplicates()[0:rows]
        return grp

    ax_idx = 1
    for filename in sorter(df_in):
        if ax_idx > rows * columns * 2:
            break

        subdf = local_df[local_df['ImageId'] == filename].reset_index() # chose file with masks

        fig.add_subplot(rows * 2, columns, ax_idx).set_title(filename)
        img = cv2.imread(os.path.join(img_dir, filename ))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_2 = cv2.imread(os.path.join(img_dir, filename ))
        img_2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)

        # showing mask from first df
            
        ax_idx += 1
        fig.add_subplot(rows * 2, columns, ax_idx).\
            set_title(filename + '      Highlighted defects ClassIds: ' + str(subdf['ClassIds'][0]))

        colors = [(255, 51, 51),(255,255,51), (51,255,51), (51,51,255)]
        masks = build_masks(subdf, (256, 1600, 4)) # get masks (256, 1600, 4)
        masks_len = masks.shape[2] # get 4
        
        for i in range(masks_len):
            img[masks[:, :, i] == 1] = colors[i]

        plt.imshow(img)
        ax_idx += 1


        # showing mask from second df
        if type(trained_df_in) == pd.DataFrame:
            subdf_trained = local_trained_df[local_trained_df['ImageId'] == filename].reset_index() # chose file with masks
            fig.add_subplot(rows * 2, columns, ax_idx).\
                set_title('Trained  '+ filename + '      Highlighted defects ClassIds: ' + str(subdf['ClassIds'][0]))

            colors = [(204, 51, 51),(204,204,51), (51,204,51), (51,51,204)]
            masks = build_masks(subdf_trained, (256, 1600, 4)) # get masks (256, 1600, 4)
            masks_len = masks.shape[2] # get 4

            for i in range(masks_len):
                img_2[masks[:, :, i] == 1] = colors[i]

            plt.imshow(img_2)
            ax_idx += 1
        
    print("Class 1 = Red","Class 2 = Yellow","Class 3 = Green","Class 4 = Blue", sep='\n')
    plt.show()
    
show_images(df, TRAIN_IMG_DIR)
class SteelDataset(Dataset):
    """
    This class takes care for creating image dataset
    """
    def __init__(self,
                 df_in: pd.DataFrame,
                 data_folder_in: str,
                 mean_in: Optional[Tuple[float]],
                 std_in: Optional[Tuple[float]],
                 phase_in: str):
        """
        :param df_in: pd.DataFrame
        :param data_folder_in: str
        :param mean_in: Optional[Tuple[float]]
        :param std_in: Optional[Tuple[float]]
        :param phase_in: str
        """

        # print('>>> SteelDataset.__init__()')
        self.df = df_in
        self.root = data_folder_in
        # normalization function
        #mean, std = batch_mean_std(df, data_folder)
        
        self.mean = mean_in
        self.std = std_in
        self.phase = phase_in
        self.transforms = get_transforms(phase_in = self.phase, mean_in = self.mean, std_in = self.std)
        # print('>>> SteelDataset.__init__() type transforms', type(self.transforms))
        # print('>>> SteelDataset.__init__() transforms', self.transforms)
        self.indices = self.df.index.tolist()
        # print('>>> SteelDataset.__init__() fnames indices', self.indices)
        # print('<<< SteelDataset.__init__()')

    def __getitem__(self, idx: int):
        """
        :param idx: int
        :return: img, mask
        """
        image_name, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root,  image_name)
        # print('>>> SteelDataset.__getitem__() image_path = ', image_path)
        img = cv2.imread(image_path)
        # print('>>> SteelDataset.__getitem__() image_name = ', image_name)

        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image'] # 3x256x1600
        if self.phase == "test":
            return image_name, img
        else:    
            mask = augmented['mask'] # 256x1600x4
            # print('img shape' ,mask.size())
            # print('mask shape before permution' ,mask.size())
            mask = mask.permute(2, 0, 1) # expected permute(2, 0, 1) 4x256x1600, got ([4, 256, 1600]) ok
            # print('<<< SteelDataset.__getitem__() img = ', img.size(), 'mask = ', mask.size())

            return img, mask

    def __len__(self):
        return len(self.indices)


def get_transforms(phase_in: str, mean_in: Optional[Tuple[float]], std_in: Optional[Tuple[float]]):
    """
    Returns a list for transforms
    :param phase_in: str
    :param mean_in: Optional[Tuple[float]
    :param std_in: Optional[Tuple[float]
    :return: list_trfms
    """
    list_transforms = []
    if phase_in == "train":
        list_transforms.extend(
            [
                HorizontalFlip(p=0.25), # only horizontal flip as of now
                VerticalFlip(p=0.25), 
                RandomBrightness(p=0.25),  
                ShiftScaleRotate(p=0.25), 
                GaussNoise(p=0.25), 
                ElasticTransform(p=0.25)
            ]
        )
    list_transforms.extend(
        [
            Normalize(mean=mean_in, std=std_in),
            ToTensorV2()
        ]
    )
    list_trfms = Compose(list_transforms)
    # print('list_trfms: ', list_trfms)
    return list_trfms

def batch_mean_std(df_in: pd.DataFrame, data_folder_in: str):
    """
    Returns mean, std for whole df
    :param df_in: pd.DataFrame
    :param data_folder_in: str
    :return: images_mean, images_std
    """
    # thanks to ronaldokun from jovian.ml
    # https://jovian.ml/forum/t/assignment-4-in-class-data-science-competition/1564/268
    print('>>> batch_mean_std')
    mean_total = []
    std_total = []
    print('df len: {0}'.format(len(df)))
    #grp = df.groupby('ImageId')
    df_loc =  df_in['ImageId']
    print(df.head())
    
    for i, filename in df_loc.items():
        path = os.path.join(data_folder_in, filename)
        img = cv2.imread(os.path.join(data_folder_in, filename), cv2.COLOR_BGR2RGB)
        img = img/255.0 # scale the image to [0,1]

        #img = cv2.imread(str(file), cv2.COLOR_RGB2BGR)
        #img = img/255.0 # scale the image to [0,1]
        # 512x512x3 is reshaped to 512*512 x 3 then the mean is calculated in the first dimension
        mean_total.append(img.reshape(-1, 3).mean(0)) 
        std_total.append((img**2).reshape(-1, 3).mean(0))

    # Image stats
    images_mean =  np.array(mean_total).mean(0)
    images_std =  np.sqrt(np.array(std_total).mean(0) - images_mean**2)
    # print('df len: {0}'.format(len(df)))
    # print('batch_mean: {0}, batch_std: {1}'.format(images_mean, images_std))
    # print('<<< batch_mean_std')
    return images_mean, images_std
    

def provider(
    data_folder: str,
    df_in: pd.DataFrame,
    phase_in: str,
    mean_in: Optional[Tuple[float, float, float]] = None,
    std_in: Optional[Tuple[float, float, float]] = None,
    batch_size: int =8,
    num_workers: int =4,
):
    """
    Returns dataloader for the model training
    :param data_folder: str
    :param df_in: pd.DataFrame
    :param phase_in: str
    :param mean_in: Optional[Tuple[float]]
    :param std_in: Optional[Tuple[float]]
    :param batch_size: int
    :param num_workers: int
    :return: dataloader
    """
    #df = pd.read_csv(df_path) ### same
    # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
    #df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_')) ### same
    #df['ClassId'] = df['ClassId'].astype(int) ### same
    #df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels') ### ???
    #df['defects'] = df.count(axis=1) ### ???
    
    # print('>>> provider started')
    data_folder_loc = data_folder
    df_loc = df_in
    phase_loc = phase_in
    mean_loc = mean_in
    std_loc = std_in    
    
    if phase_loc != "test":
        train_df_loc, val_df_loc = train_test_split(df_loc, test_size=0.2, stratify=df_in["maskCount"], random_state=69)
        # print('>>> provider started (1), shapes: train_df = {0}, test_df = {1}'.format(train_df.shape, val_df.shape))
        # Split arrays or matrices into random train and test subsets
        df_loc = train_df_loc if phase_loc == "train" else val_df_loc
    # print('>>> provider started (2)')
       
    image_dataset = SteelDataset(df_in = df_loc, 
                                 data_folder_in = data_folder_loc, 
                                 mean_in = mean_loc, 
                                 std_in = std_loc, 
                                 phase_in = phase_loc
                                ) # img, mask
    # print('image_dataset (img, mask) len= ', len(image_dataset))
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    #print('dataloader len= ', len(dataloader))
    #print('dataloader = ', dataloader)
    # print('<<< provider finished ()')
    return dataloader

def get_default_device():
    """Pick GPU if available, else CPU"""
    
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        return torch.device('cuda')
    else:
        torch.set_default_tensor_type("torch.FloatTensor")
        return torch.device('cpu')
        
class Meter:
    """A meter to keep track of iou (Jacard index) throughout an epoch"""
    def __init__(self, phase: str, epoch: int, ):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.F_scores = []
        self.iou_scores = []
        self.Dice_scores = []

    def update(self, targets: torch.Tensor, outputs: torch.Tensor):
        """ 
        targets: torch.tensor([0..1])
        outputs: torch.tensor([-1..1])
         """
        probs = torch.sigmoid(outputs) # torch.tensor([0..1])
        
        F, iou, Dice = compute_batch(probs, targets, self.base_threshold)
        self.iou_scores.append(iou)
        self.F_scores.append(F)
        self.Dice_scores.append(Dice)

    def get_metrics(self):
        F = np.nanmean(self.F_scores)
        iou = np.nanmean(self.iou_scores)
        Dice = np.nanmean(self.Dice_scores)
        return F, iou, Dice

def epoch_log(phase, epoch, epoch_loss, meter, start):
    """logging the metrics at the end of an epoch"""

    F, iou, Dice  = meter.get_metrics()
    print("Phase: %s | Epoch: %d | Loss: %0.4f | F: %0.4f | IoU: %0.4f | Dice: %0.4f" % (phase, epoch, epoch_loss, F, iou, Dice))
    return F, iou, Dice 


def confusion_matrix_and_union(prediction: torch.Tensor, truth: torch.Tensor, threshold: int):
    """ 
    Coputes a confusion matrix elements and union for one ground truth mask and predicted mask
    TP, FP, FN, TN, U 
    """
    assert prediction.shape == truth.shape # [4x256x1600]
    
    #flatten label and prediction tensors
    prediction = prediction.view(-1) # [1638400]
    truth = truth.view(-1) # [1638400]
    
    prob = (prediction >= threshold).int()
    label = (truth).int()
    not_prob = (1-prob)
    not_label = (1-label)
    

    
    TP = (prob&label).sum().to(torch.float32)           # Will be zero if Truth=0 or Prediction=0
    FP = (prob&not_label).sum().to(torch.float32)       # Will be zero if Truth=0 or Prediction=1
    FN = (not_prob&label).sum().to(torch.float32)       # Will be zero if Truth=1 or Prediction=0
    TN = (not_prob&not_label).sum().to(torch.float32)   # Will be zero if Truth=1 or Prediction=1
    U = (prob|label).sum().to(torch.float32)            # Will be zero if both are 0
    # print('>>> confusion_matrix_and_union >>> TP: {0}, FP: {1}, FN: {2}, TN: {3}, U: {4}'.format(TP, FP, FN, TN, U))
    return TP, FP, FN, TN, U 

def compute_ious(TP: torch.Tensor, U: torch.Tensor):
    """computes iou for one ground truth mask and predicted mask"""
    
    iou = (TP + 1e-12) / (U + 1e-12)  # We smooth our devision to avoid 0/0
    
    return iou

def compute_F_score(TP: torch.Tensor, FP: torch.Tensor, FN: torch.Tensor, beta=1):
    """
    computes F metric for one ground truth mask and predicted mask
    """
    precision = torch.mean(TP / (TP + FP + 1e-12))  # We adding nearzero value to except dividing by zero
    recall = torch.mean(TP / (TP + FN + 1e-12))
    
    F = ((1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12))
    
    return F

def compute_Dice(prediction: torch.Tensor, truth: torch.Tensor, TP: torch.Tensor, threshold: int, reduction='mean'):
    """Computes a Dice score"""

    assert prediction.shape == truth.shape # [4x256x1600]
    
    #flatten label and prediction tensors
    prediction = prediction.view(-1) # [1638400]
    truth = truth.view(-1) # [1638400]
    
    prob = (prediction >= threshold).float()
    label = (truth).float()
    
    intersection = TP.sum()
    prob_sum = prob.sum()
    label_sum = label.sum()
    
    Dice = (2. * intersection + 1e-12) / (prob_sum + label_sum + 1e-12)

    return Dice


def compute_batch(prediction: torch.Tensor, truth: torch.Tensor, threshold: int):
    """computes means F IoU Dice for a batch of ground truth masks and predicted masks"""
    Fs = []
    ious = []
    Dices = []

    # batch shape [4x4x256x1600]
    for preds, labels in zip(prediction, truth):
        # [4x256x1600]
        # we can see a personal metric for each class in image
        
        TP, FP, FN, TN, U = confusion_matrix_and_union(preds, labels, threshold)

        Fs.append(np.array(compute_F_score(TP, FP, FN, beta=1)).mean()) # [4]
        ious.append(np.array(compute_ious(TP, U)).mean()) # [4]
        Dices.append(np.array(compute_Dice(preds, labels, TP, threshold)).mean()) # [4]
    # print('Fs: ',Fs)
    # print('ious: ',ious)
    # print('Dices: ', Dices)
    F = np.array(Fs).mean()
    iou = np.array(ious).mean()
    Dice = np.array(Dices).mean()
    # print('F: ',F)
    # print('iou: ',iou)
    # print('Dice: ', Dice)
    return F, iou, Dice

model = smp.Unet("resnet50", encoder_weights="imagenet", classes=4, activation=None)
#print(type(model))
# model # uncomment to take a deeper look
class Trainer(object):
    """This class takes care of training and validation of our model"""
    def __init__(self, epochs_in: int, model_in, df_in: pd.DataFrame):
        """
        :param epochs_in:
        :param model_in:
        :param df_in:
        """

        # print('>>> Trainer.__init()__')
        self.df_local = df_in
        self.num_workers = 6
        self.batch_size = {"train": 4, "val": 4}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 5e-4
        self.num_epochs = epochs_in # 20
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = get_default_device()
        
        self.net = model_in
        self.criterion = nn.BCEWithLogitsLoss() #
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr) #
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True) #
        self.net = self.net.to(self.device) 
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                data_folder=TRAIN_IMG_DIR,
                df_in=self.df_local,
                phase_in=phase,
                mean_in=(0.485, 0.456, 0.406),
                std_in=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers
            )
            for phase in self.phases
        }

        # print('>>> Trainer.__init()__ self.dataloaders ', self.dataloaders)
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.F_scores = {phase: [] for phase in self.phases}
        self.Dice_scores = {phase: [] for phase in self.phases}
        
        # print('<<< Trainer.__init()__')

        
    def forward(self, images, targets):
        # print('>>> Trainer.forward()')
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        # print('<<< Trainer.forward()')
        return loss, outputs

    def iterate(self, epoch, phase):
        # print('>>> Trainer.iterate()')
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        # print('>>> Trainer.iterate() dataloader: ', dataloader)
        running_loss = 0.0
        total_batches = len(dataloader)
        tk0 = tqdm(dataloader, total=total_batches) # progress bar
        self.optimizer.zero_grad()
        for itr, batch in enumerate(tk0): # replace `dataloader` with `tk0` for tqdm
            # TODO: запускать 
            images, targets = batch
            # print('>>> Trainer len(batch): ', len(batch)) # 1 ok
            # print('>>> Trainer images.shape: ', images.shape) #  # ([4, 3, 256, 1600])
            # print('>>> Trainer targets.shape: ', targets.shape)  # ([4, 4, 256, 1600])

            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            # print('>>> Trainer targets type: ',type(targets))
            # print('>>> Trainer targets: ',targets) # torch.tensor([0..1])
            # print('>>> Trainer outputs type: ',type(outputs))
            # print('>>> Trainer outputs: ',outputs) # torch.tensor([-1..1])
            meter.update(targets, outputs)
            tk0.set_postfix(loss=(running_loss / ((itr + 1)))) #
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        iou, F, Dice = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.iou_scores[phase].append(iou)
        self.F_scores[phase].append(F)
        self.Dice_scores[phase].append(Dice)
        torch.cuda.empty_cache()
        # print('<<< Trainer.iterate()')
        return epoch_loss

    def start(self):
        # print('>>> Trainer.start()')
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, TRAINED_WEIGHTS)
            print()
        # print('<<< Trainer.start()')
            
class Tester(object):
    def __init__(self, model_in, df_in: pd.DataFrame, image_root: str, weights_root_in: str = TRAINED_WEIGHTS):
        """
        Returns dataframe and make submission file like sample_submission.csv with rles and classes
        :param model_in: Unet model
        :param df_in: test_df
        :param image_root: str
        :weights_root: str
        """

        self.df = df_in
        self.num_workers = 2
        self.batch_size = 8
        self.device = get_default_device()
        self.best_threshold = 0.25 # 0.5
        self.weigths_root = weights_root_in
        self.state = torch.load(self.weigths_root)
        self.model = model_in
        self.model.to(self.device)
        self.model.eval()
        self.model.load_state_dict(self.state["state_dict"])

        print('best_threshold', self.best_threshold)
        self.min_size = 100 # 3500 # min total of pixels on the mask
        self.test_dataloader = provider(
                        data_folder=image_root,
                        df_in=self.df,
                        phase_in = 'test',
                        mean_in=(0.485, 0.456, 0.406),
                        std_in=(0.229, 0.224, 0.225),
                        batch_size=self.batch_size,
                        num_workers=self.num_workers
                    )

        self.predicted_pixels = [] # np.zeros((4, 256, 1600), np.float32)

    def process(self, probability, threshold, min_size):
        '''Post processing of each predicted mask, components with lesser number of pixels
        than `min_size` are ignored'''
        mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1] # (256, 1600)
        num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
        predictions = np.zeros((256, 1600), np.float32)
        
        for c in range(1, num_component):
            p = (component == c)
            if p.sum() > min_size:
                predictions[p] = 1 # (4, 256, 1600)
        return predictions 

    def start(self):
        for i, batch in enumerate(tqdm(self.test_dataloader)):
            fnames, images = batch 
            batch_preds = torch.sigmoid(model(images.to(self.device)))
            batch_preds = batch_preds.detach().cpu().numpy()
            # print('>>> Tester > images.shape: ', images.shape) # (4, 3, 256, 1600)
            # print('>>> Tester > batch_preds.shape: ', batch_preds.shape) # (4, 4, 256, 1600)
            for fname, pred_mask in zip(fnames, batch_preds):
                for cls, pred_mask in enumerate(pred_mask):
                    pred_mask = self.process(pred_mask, self.best_threshold, self.min_size)  # (4, 256, 1600)
                    rle = mask2rle(pred_mask)
                    if rle != '':
                        cls = index2class_id(cls)
                        pred_loc = [fname, cls, rle]
                        self.predicted_pixels.append(pred_loc)

        # save predictions to submission.csv
        self.df = pd.DataFrame(self.predicted_pixels, columns=['ImageId', 'ClassId', 'EncodedPixels'])      
        self.df.to_csv(SUBMISSION_FILE, index=False)
model_trainer = Trainer(epochs_in = 1, model_in = model, df_in = df)
# model_trainer.start()  # uncomment to start training
def plot(scores, name):
    """
    plots metric
    """
    plt.figure(figsize=(15,5))
    plt.plot(range(len(scores["train"])), scores["train"], label=f'train {name}')
    plt.plot(range(len(scores["train"])), scores["val"], label=f'val {name}')
    plt.title(f'{name} plot'); plt.xlabel('Epoch'); plt.ylabel(f'{name}')
    plt.legend()
    plt.show()

plot(model_trainer.losses, "BCE loss")
plot(model_trainer.iou_scores, "IoU score")
plot(model_trainer.F_scores, "F score")
plot(model_trainer.Dice_scores, "Dice score")
#prepairing new dataset like sample_submission.csv
predicted_train_df = df.iloc[:, :1].drop_duplicates()
predicted_train_df['EncodedPixels'] =  '1 409600'
predicted_train_df['ClassId'] =  0
predicted_train_df.head()
model_tester = Tester(model_in = model, df_in = predicted_train_df, image_root = TRAIN_IMG_DIR, weights_root_in = PRE_TRAINED_WEIGHTS) # remove to use inner weights:  weights_root_in = PRE_TRAINED_WEIGHTS
model_tester.start()
predicted_train_df = pd.read_csv(SUBMISSION_FILE)
predicted_train_df = survey(predicted_train_df)
predicted_train_df.head()
mask_count_df_pivot = pd.DataFrame({'ClassCount' : predicted_train_df.groupby('ImageId').size()})
mask_count_df_pivot = pd.DataFrame({'Num' : mask_count_df_pivot.groupby('ClassCount').size()})
mask_count_df_pivot.sort_values('ClassCount', ascending=True, inplace=True)
ClassId_count_df = predicted_train_df.set_index(["ImageId", "ClassId"]).count(level='ClassId')

total, counter = counter_func(ClassId_count_df)
print('Total strings: {0}, 1 class: {1}, 2 class: {2}, 3 class: {3}, 4 class: {4}'.format(total, *counter))
total, counter = counter_func(mask_count_df_pivot)
print('Total images: {0}, one class: {1}, two classes: {2}, three classes: {3}, four classes: {4}'.format(total, *counter))
show_images(df[:], TRAIN_IMG_DIR, predicted_train_df)
# model_tester = Tester(model_in = model, df_in = test_df, image_root = TEST_IMG_DIR)
model_tester = Tester(model_in = model, df_in = test_df, image_root = TEST_IMG_DIR, weights_root_in = PRE_TRAINED_WEIGHTS)
model_tester.start()
predicted_df = pd.read_csv(SUBMISSION_FILE)
predicted_df = survey(predicted_df)
predicted_df.head()
mask_count_df_pivot = pd.DataFrame({'ClassCount' : predicted_df.groupby('ImageId').size()})
mask_count_df_pivot = pd.DataFrame({'Num' : mask_count_df_pivot.groupby('ClassCount').size()})
mask_count_df_pivot.sort_values('ClassCount', ascending=True, inplace=True)
ClassId_count_df = predicted_df.set_index(["ImageId", "ClassId"]).count(level='ClassId')

total, counter = counter_func(ClassId_count_df)
print('Total strings: {0}, 1 class: {1}, 2 class: {2}, 3 class: {3}, 4 class: {4}'.format(total, *counter))
total, counter = counter_func(mask_count_df_pivot)
print('Total images: {0}, one class: {1}, two classes: {2}, three classes: {3}, four classes: {4}'.format(total, *counter))
show_images(predicted_df, TEST_IMG_DIR)
!pip install jovian --upgrade
import jovian
jovian.commit(project='asigment5-zero-to-gans-novikov-ms')