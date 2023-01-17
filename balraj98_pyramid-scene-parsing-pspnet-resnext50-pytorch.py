import os, cv2

import numpy as np

import pandas as pd

import random, tqdm

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



import torch

import torch.nn as nn

from torch.utils.data import DataLoader

import albumentations as album
!pip install -q -U segmentation-models-pytorch albumentations > /dev/null

import segmentation_models_pytorch as smp
DATA_DIR = '../input/stanford-background-dataset'



metadata_df = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))

metadata_df = metadata_df[['image_id', 'image_path', 'label_colored_path']]

metadata_df['image_path'] = metadata_df['image_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))

metadata_df['label_colored_path'] = metadata_df['label_colored_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))

# Shuffle DataFrame

metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)



# Perform 90/10 split for train / val

valid_df = metadata_df.sample(frac=0.1, random_state=42)

train_df = metadata_df.drop(valid_df.index)

len(train_df), len(valid_df)
class_dict = pd.read_csv(os.path.join(DATA_DIR, 'labels_class_dict.csv'))

# Get class names

class_names = class_dict['class_names'].tolist()

# Get class RGB values

class_rgb_values = class_dict[['r','g','b']].values.tolist()



print('All dataset classes and their corresponding RGB values in labels:')

print('Class Names: ', class_names)

print('Class RGB values: ', class_rgb_values)
# Useful to shortlist specific classes in datasets with large number of classes

select_classes = ['sky', 'tree', 'road', 'grass', 'water', 'building', 'mountain', 'foreground', 'unknown']



# Get RGB values of required classes

select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]

select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]



print('Selected classes and their corresponding RGB values in labels:')

print('Class Names: ', class_names)

print('Class RGB values: ', class_rgb_values)
# helper function for data visualization

def visualize(**images):

    """

    Plot images in one row

    """

    n_images = len(images)

    plt.figure(figsize=(20,8))

    for idx, (name, image) in enumerate(images.items()):

        plt.subplot(1, n_images, idx + 1)

        plt.xticks([]); 

        plt.yticks([])

        # get title from the parameter names

        plt.title(name.replace('_',' ').title(), fontsize=20)

        plt.imshow(image)

    plt.show()



# Perform one hot encoding on label

def one_hot_encode(label, label_values):

    """

    Convert a segmentation image label array to one-hot format

    by replacing each pixel value with a vector of length num_classes

    # Arguments

        label: The 2D array segmentation image label

        label_values

        

    # Returns

        A 2D array with the same width and hieght as the input, but

        with a depth size of num_classes

    """

    semantic_map = []

    for colour in label_values:

        equality = np.equal(label, colour)

        class_map = np.all(equality, axis = -1)

        semantic_map.append(class_map)

    semantic_map = np.stack(semantic_map, axis=-1)



    return semantic_map

    

# Perform reverse one-hot-encoding on labels / preds

def reverse_one_hot(image):

    """

    Transform a 2D array in one-hot format (depth is num_classes),

    to a 2D array with only 1 channel, where each pixel value is

    the classified class key.

    # Arguments

        image: The one-hot format image 

        

    # Returns

        A 2D array with the same width and hieght as the input, but

        with a depth size of 1, where each pixel value is the classified 

        class key.

    """

    x = np.argmax(image, axis = -1)

    return x



# Perform colour coding on the reverse-one-hot outputs

def colour_code_segmentation(image, label_values):

    """

    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments

        image: single channel array where each value represents the class key.

        label_values



    # Returns

        Colour coded image for segmentation visualization

    """

    colour_codes = np.array(label_values)

    x = colour_codes[image.astype(int)]



    return x
class BackgroundDataset(torch.utils.data.Dataset):



    """Stanford Background Dataset. Read images, apply augmentation and preprocessing transformations.

    

    Args:

        df (str): DataFrame containing images / labels paths

        class_rgb_values (list): RGB values of select classes to extract from segmentation mask

        augmentation (albumentations.Compose): data transfromation pipeline 

            (e.g. flip, scale, etc.)

        preprocessing (albumentations.Compose): data preprocessing 

            (e.g. noralization, shape manipulation, etc.)

    

    """

    def __init__(

            self, 

            df,

            class_rgb_values=None, 

            augmentation=None, 

            preprocessing=None,

    ):

        self.image_paths = df['image_path'].tolist()

        self.mask_paths = df['label_colored_path'].tolist()

        

        self.class_rgb_values = class_rgb_values

        self.augmentation = augmentation

        self.preprocessing = preprocessing

    

    def __getitem__(self, i):

        

        # read images and masks

        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)

        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)

        

        # one-hot-encode the mask

        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        

        # apply augmentations

        if self.augmentation:

            sample = self.augmentation(image=image, mask=mask)

            image, mask = sample['image'], sample['mask']

        

        # apply preprocessing

        if self.preprocessing:

            sample = self.preprocessing(image=image, mask=mask)

            image, mask = sample['image'], sample['mask']

            

        return image, mask

        

    def __len__(self):

        # return length of 

        return len(self.image_paths)
dataset = BackgroundDataset(train_df, class_rgb_values=select_class_rgb_values)

random_idx = random.randint(0, len(dataset)-1)

image, mask = dataset[2]



visualize(

    original_image = image,

    ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),

    one_hot_encoded_mask = reverse_one_hot(mask)

)
def get_training_augmentation():

    train_transform = [

        album.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),

        album.OneOf(

            [

                album.HorizontalFlip(p=1),

                album.VerticalFlip(p=1),

                album.RandomRotate90(p=1),

            ],

            p=0.5,

        ),

    ]

    return album.Compose(train_transform)





def get_validation_augmentation():

    # Add sufficient padding to ensure image is divisible by 32

    test_transform = [

        album.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),

    ]

    return album.Compose(test_transform)





def to_tensor(x, **kwargs):

    return x.transpose(2, 0, 1).astype('float32')





def get_preprocessing(preprocessing_fn=None):

    """Construct preprocessing transform    

    Args:

        preprocessing_fn (callable): data normalization function 

            (can be specific for each pretrained neural network)

    Return:

        transform: albumentations.Compose

    """   

    _transform = []

    if preprocessing_fn:

        _transform.append(album.Lambda(image=preprocessing_fn))

    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

        

    return album.Compose(_transform)
augmented_dataset = BackgroundDataset(

    train_df, 

    augmentation=get_training_augmentation(),

    class_rgb_values=select_class_rgb_values,

)



random_idx = random.randint(0, len(augmented_dataset)-1)



# Different augmentations on image/mask pairs

for idx in range(3):

    image, mask = augmented_dataset[idx]

    visualize(

        original_image = image,

        ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),

        one_hot_encoded_mask = reverse_one_hot(mask)

    )
ENCODER = 'resnext50_32x4d'

ENCODER_WEIGHTS = 'imagenet'

CLASSES = select_classes

ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation



# create segmentation model with pretrained encoder

model = smp.PSPNet(

    encoder_name=ENCODER, 

    encoder_weights=ENCODER_WEIGHTS, 

    classes=len(CLASSES), 

    activation=ACTIVATION,

)



preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
# Get train and val dataset instances

train_dataset = BackgroundDataset(

    train_df, 

    augmentation=get_training_augmentation(),

    preprocessing=get_preprocessing(preprocessing_fn),

    class_rgb_values=select_class_rgb_values,

)



valid_dataset = BackgroundDataset(

    valid_df, 

    augmentation=get_validation_augmentation(), 

    preprocessing=get_preprocessing(preprocessing_fn),

    class_rgb_values=select_class_rgb_values,

)



# Get train and val data loaders

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4)
# Set flag to train the model or not. If set to 'False', only prediction is performed (using an older model checkpoint)

TRAINING = True



# Set num of epochs

EPOCHS = 100



# Set device: `cuda` or `cpu`

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# define loss function

loss = smp.utils.losses.DiceLoss()



# define metrics

metrics = [

    smp.utils.metrics.IoU(threshold=0.5),

]



# define optimizer

optimizer = torch.optim.Adam([ 

    dict(params=model.parameters(), lr=0.0001),

])



# define learning rate scheduler (not used in this NB)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(

    optimizer, T_0=1, T_mult=2, eta_min=5e-5,

)



# load best saved model checkpoint from previous commit (if present)

if os.path.exists('../input/pyramid-scene-parsing-pspnet-resnext50-pytorch/best_model.pth'):

    model = torch.load('../input/pyramid-scene-parsing-pspnet-resnext50-pytorch/best_model.pth', map_location=DEVICE)
train_epoch = smp.utils.train.TrainEpoch(

    model, 

    loss=loss, 

    metrics=metrics, 

    optimizer=optimizer,

    device=DEVICE,

    verbose=True,

)



valid_epoch = smp.utils.train.ValidEpoch(

    model, 

    loss=loss, 

    metrics=metrics, 

    device=DEVICE,

    verbose=True,

)
%%time



if TRAINING:



    best_iou_score = 0.0

    train_logs_list, valid_logs_list = [], []



    for i in range(0, EPOCHS):



        # Perform training & validation

        print('\nEpoch: {}'.format(i))

        train_logs = train_epoch.run(train_loader)

        valid_logs = valid_epoch.run(valid_loader)

        train_logs_list.append(train_logs)

        valid_logs_list.append(valid_logs)



        # Save model if a better val IoU score is obtained

        if best_iou_score < valid_logs['iou_score']:

            best_iou_score = valid_logs['iou_score']

            torch.save(model, './best_model.pth')

            print('Model saved!')
# load best saved model checkpoint from the current run

if os.path.exists('./best_model.pth'):

    best_model = torch.load('./best_model.pth', map_location=DEVICE)

    print('Loaded PSPNet model from this run.')



# load best saved model checkpoint from previous commit (if present)

elif os.path.exists('../input/pyramid-scene-parsing-pspnet-resnext50-pytorch/best_model.pth'):

    best_model = torch.load('../input/pyramid-scene-parsing-pspnet-resnext50-pytorch/best_model.pth', map_location=DEVICE)

    print('Loaded PSPNet model from a previous commit.')
# create test dataloader (with preprocessing operation: to_tensor(...))

test_dataset = BackgroundDataset(

    valid_df, 

    augmentation=get_validation_augmentation(), 

    preprocessing=get_preprocessing(preprocessing_fn),

    class_rgb_values=select_class_rgb_values,

)



test_dataloader = DataLoader(test_dataset)



# test dataset for visualization (without preprocessing augmentations & transformations)

test_dataset_vis = BackgroundDataset(

    valid_df,

    class_rgb_values=select_class_rgb_values,

)



# get a random test image/mask index

random_idx = random.randint(0, len(test_dataset_vis)-1)

image, mask = test_dataset_vis[random_idx]



visualize(

    original_image = image,

    ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),

    one_hot_encoded_mask = reverse_one_hot(mask)

)
# Center crop padded image / mask to original image dims

def crop_image(image, true_dimensions):

    return album.CenterCrop(p=1, height=true_dimensions[0], width=true_dimensions[1])(image=image)
sample_preds_folder = 'sample_predictions/'

if not os.path.exists(sample_preds_folder):

    os.makedirs(sample_preds_folder)
for idx in range(len(test_dataset)):



    image, gt_mask = test_dataset[idx]

    image_vis = test_dataset_vis[idx][0].astype('uint8')

    true_dimensions = image_vis.shape

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

    # Predict test image

    pred_mask = best_model(x_tensor)

    pred_mask = pred_mask.detach().squeeze().cpu().numpy()

    # Convert pred_mask from `CHW` format to `HWC` format

    pred_mask = np.transpose(pred_mask,(1,2,0))

    # Get prediction channel corresponding to foreground

    pred_foreground_heatmap = crop_image(pred_mask[:,:,select_classes.index('foreground')], true_dimensions)['image']

    pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values), true_dimensions)['image']

    # Convert gt_mask from `CHW` format to `HWC` format

    gt_mask = np.transpose(gt_mask,(1,2,0))

    gt_mask = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), select_class_rgb_values), true_dimensions)['image']

    cv2.imwrite(os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"), np.hstack([image_vis, gt_mask, pred_mask])[:,:,::-1])

    

    visualize(

        original_image = image_vis,

        ground_truth_mask = gt_mask,

        predicted_mask = pred_mask,

        pred_foreground_heatmap = pred_foreground_heatmap

    )
test_epoch = smp.utils.train.ValidEpoch(

    model,

    loss=loss, 

    metrics=metrics, 

    device=DEVICE,

    verbose=True,

)



valid_logs = test_epoch.run(test_dataloader)

print("Evaluation on Test Data: ")

print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")

print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")
train_logs_df = pd.DataFrame(train_logs_list)

valid_logs_df = pd.DataFrame(valid_logs_list)

train_logs_df.T
plt.figure(figsize=(20,8))

plt.plot(train_logs_df.index.tolist(), train_logs_df.iou_score.tolist(), lw=3, label = 'Train')

plt.plot(valid_logs_df.index.tolist(), valid_logs_df.iou_score.tolist(), lw=3, label = 'Valid')

plt.xlabel('Epochs', fontsize=20)

plt.ylabel('IoU Score', fontsize=20)

plt.title('IoU Score Plot', fontsize=20)

plt.legend(loc='best', fontsize=16)

plt.grid()

plt.savefig('iou_score_plot.png')

plt.show()
plt.figure(figsize=(20,8))

plt.plot(train_logs_df.index.tolist(), train_logs_df.dice_loss.tolist(), lw=3, label = 'Train')

plt.plot(valid_logs_df.index.tolist(), valid_logs_df.dice_loss.tolist(), lw=3, label = 'Valid')

plt.xlabel('Epochs', fontsize=20)

plt.ylabel('Dice Loss', fontsize=20)

plt.title('Dice Loss Plot', fontsize=20)

plt.legend(loc='best', fontsize=16)

plt.grid()

plt.savefig('dice_loss_plot.png')

plt.show()