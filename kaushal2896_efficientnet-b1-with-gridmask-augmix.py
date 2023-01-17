!pip install --upgrade efficientnet-pytorch
import os

import gc

import random



import cv2

import numpy as np

import pandas as pd



from tqdm.auto import tqdm



import torch

from torch import nn

import albumentations

from torchvision import transforms

from torch.optim import lr_scheduler

import torch.utils.data as data_utils

from torch.utils.data import Dataset, DataLoader

from albumentations.core.transforms_interface import DualTransform



from efficientnet_pytorch import EfficientNet



from albumentations.core.transforms_interface import ImageOnlyTransform

from albumentations.augmentations import functional as F

from PIL import Image, ImageOps, ImageEnhance





from PIL import Image



from sklearn.model_selection import train_test_split



from matplotlib import pyplot as plt
# Constants

SEED = 1996

BATCH_SIZE = 256 

DIM = (128, 128)

SIZE = 128

HEIGHT = 137 

WIDTH = 236

PARTIAL_SIZE = 5000

VAL_SIZE = 0.01



# Set random seeds

random.seed(SEED)

np.random.seed(SEED)



# load files

IMG_PATH = '../input/grapheme-imgs-128x128/'

train_df = pd.read_csv('../input/bengaliai-cv19/train.csv')

test_df = pd.read_csv('../input/bengaliai-cv19/test.csv')

train_df['filename'] = train_df.image_id.apply(lambda filename: os.path.join(IMG_PATH + filename + '.png'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



train_df = train_df.drop(['grapheme'], axis=1)



# top 5 samples

train_df.head()
# TODO: Partial data. Remove for actual training

train_df = train_df[0:PARTIAL_SIZE]
train_df, val_df = train_test_split(train_df, test_size=VAL_SIZE, random_state=SEED, shuffle=False, stratify=None)

val_df = val_df.reset_index(drop=True)

print(train_df.shape, val_df.shape)
train_df.tail()
val_df.tail()
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

        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask

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
def int_parameter(level, maxval):

    """Helper function to scale `val` between 0 and maxval .

    Args:

    level: Level of the operation that will be between [0, `PARAMETER_MAX`].

    maxval: Maximum value that the operation can have. This will be scaled to

      level/PARAMETER_MAX.

    Returns:

    An int that results from scaling `maxval` according to `level`.

    """

    return int(level * maxval / 10)





def float_parameter(level, maxval):

    """Helper function to scale `val` between 0 and maxval.

    Args:

    level: Level of the operation that will be between [0, `PARAMETER_MAX`].

    maxval: Maximum value that the operation can have. This will be scaled to

      level/PARAMETER_MAX.

    Returns:

    A float that results from scaling `maxval` according to `level`.

    """

    return float(level) * maxval / 10.





def sample_level(n):

    return np.random.uniform(low=0.1, high=n)





def autocontrast(pil_img, _):

    return ImageOps.autocontrast(pil_img)





def equalize(pil_img, _):

    return ImageOps.equalize(pil_img)





def posterize(pil_img, level):

    level = int_parameter(sample_level(level), 4)

    return ImageOps.posterize(pil_img, 4 - level)





def rotate(pil_img, level):

    degrees = int_parameter(sample_level(level), 30)

    if np.random.uniform() > 0.5:

        degrees = -degrees

    return pil_img.rotate(degrees, resample=Image.BILINEAR)





def solarize(pil_img, level):

    level = int_parameter(sample_level(level), 256)

    return ImageOps.solarize(pil_img, 256 - level)





def shear_x(pil_img, level):

    level = float_parameter(sample_level(level), 0.3)

    if np.random.uniform() > 0.5:

        level = -level

    return pil_img.transform(pil_img.size,

                           Image.AFFINE, (1, level, 0, 0, 1, 0),

                           resample=Image.BILINEAR)





def shear_y(pil_img, level):

    level = float_parameter(sample_level(level), 0.3)

    if np.random.uniform() > 0.5:

        level = -level

    return pil_img.transform(pil_img.size,

                           Image.AFFINE, (1, 0, 0, level, 1, 0),

                           resample=Image.BILINEAR)





def translate_x(pil_img, level):

    level = int_parameter(sample_level(level), pil_img.size[0] / 3)

    if np.random.random() > 0.5:

        level = -level

    return pil_img.transform(pil_img.size,

                           Image.AFFINE, (1, 0, level, 0, 1, 0),

                           resample=Image.BILINEAR)





def translate_y(pil_img, level):

    level = int_parameter(sample_level(level), pil_img.size[0] / 3)

    if np.random.random() > 0.5:

        level = -level

    return pil_img.transform(pil_img.size,

                           Image.AFFINE, (1, 0, 0, 0, 1, level),

                           resample=Image.BILINEAR)





# operation that overlaps with ImageNet-C's test set

def color(pil_img, level):

    level = float_parameter(sample_level(level), 1.8) + 0.1

    return ImageEnhance.Color(pil_img).enhance(level)





# operation that overlaps with ImageNet-C's test set

def contrast(pil_img, level):

    level = float_parameter(sample_level(level), 1.8) + 0.1

    return ImageEnhance.Contrast(pil_img).enhance(level)





# operation that overlaps with ImageNet-C's test set

def brightness(pil_img, level):

    level = float_parameter(sample_level(level), 1.8) + 0.1

    return ImageEnhance.Brightness(pil_img).enhance(level)





# operation that overlaps with ImageNet-C's test set

def sharpness(pil_img, level):

    level = float_parameter(sample_level(level), 1.8) + 0.1

    return ImageEnhance.Sharpness(pil_img).enhance(level)





augmentations = [

    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,

    translate_x, translate_y

]



augmentations_all = [

    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,

    translate_x, translate_y, color, contrast, brightness, sharpness

]



def normalize(image):

    """Normalize input image channel-wise to zero mean and unit variance."""

    return image - 127



def apply_op(image, op, severity):

    #   image = np.clip(image, 0, 255)

    pil_img = Image.fromarray(image)  # Convert to PIL.Image

    pil_img = op(pil_img, severity)

    return np.asarray(pil_img)



def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.):

    """Perform AugMix augmentations and compute mixture.

    Args:

    image: Raw input image as float32 np.ndarray of shape (h, w, c)

    severity: Severity of underlying augmentation operators (between 1 to 10).

    width: Width of augmentation chain

    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly

      from [1, 3]

    alpha: Probability coefficient for Beta and Dirichlet distributions.

    Returns:

    mixed: Augmented and mixed image.

    """

    ws = np.float32(

      np.random.dirichlet([alpha] * width))

    m = np.float32(np.random.beta(alpha, alpha))



    mix = np.zeros_like(image).astype(np.float32)

    for i in range(width):

        image_aug = image.copy()

        depth = depth if depth > 0 else np.random.randint(1, 4)

        for _ in range(depth):

            op = np.random.choice(augmentations)

            image_aug = apply_op(image_aug, op, severity)

        # Preprocessing commutes since all coefficients are convex

        mix += ws[i] * image_aug

#         mix += ws[i] * normalize(image_aug)



    mixed = (1 - m) * image + m * mix

#     mixed = (1 - m) * normalize(image) + m * mix

    return mixed





class RandomAugMix(ImageOnlyTransform):



    def __init__(self, severity=3, width=3, depth=-1, alpha=1., always_apply=False, p=0.5):

        super().__init__(always_apply, p)

        self.severity = severity

        self.width = width

        self.depth = depth

        self.alpha = alpha



    def apply(self, image, **params):

        image = augment_and_mix(

            image,

            self.severity,

            self.width,

            self.depth,

            self.alpha

        )

        return image
class GraphemesDataset(Dataset):

    def __init__(self, train_df, dim, transform):

        self.train_df = train_df

        self.dim = dim

        self.transform = transform

        

    def __len__(self):

        return len(self.train_df)

    

    @staticmethod

    def to_grayscale(rgb_image):

        return np.dot(rgb_image[... , :3] , [0.299 , 0.587, 0.114])

    

    def __getitem__(self, index):

        # load the image file using cv2

        image = cv2.imread(os.path.join(IMG_PATH + self.train_df['image_id'][index] + '.png'))

        image = cv2.resize(image,  self.dim)

        

        if self.transform:

            res = self.transform(image=image)

            image = res['image'].astype(np.float32)

        else:

            image = image.astype(np.float32)



        image = self.to_grayscale(image)

        image /= 255

        image = image.reshape(-1, SIZE, SIZE)

        

        target_root = self.train_df['grapheme_root'].iloc[index]

        target_vowel = self.train_df['vowel_diacritic'].iloc[index]

        target_consonant = self.train_df['consonant_diacritic'].iloc[index]

        

        return image, target_root, target_vowel, target_consonant
transforms_train = albumentations.Compose([

    albumentations.OneOf([

        GridMask(num_grid=3, mode=0, rotate=15),

        GridMask(num_grid=3, mode=2, rotate=15),

    ], p=0.7),

    RandomAugMix(severity=4, width=3, alpha=1.0, p=0.7),

])

train_dataset = GraphemesDataset(train_df, DIM, transform=transforms_train)

train_loader = data_utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
def plot_images(images):

    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 8))

    count=0

    for row in ax:

        for col in row:

            col.imshow(images[count].reshape(SIZE, SIZE).cpu().detach().numpy().astype(np.float64))

            count += 1

    plt.show()
plot_images(next(iter(train_loader))[0])
# Generate the validation dataset

val_dataset = GraphemesDataset(val_df, DIM,transform=None)

val_loader = data_utils.DataLoader(val_dataset, batch_size=int(PARTIAL_SIZE*VAL_SIZE), shuffle=False)

X_val, Y_val_root, Y_val_vowel, Y_val_consonant = next(iter(val_loader))

X_val.shape
X_val = X_val.to(device, dtype=torch.float)

Y_val_root = Y_val_root.to(device)

Y_val_vowel = Y_val_vowel.to(device)

Y_val_consonant = Y_val_consonant.to(device)
class EfficientNetWrapper(nn.Module):

    def __init__(self):

        super(EfficientNetWrapper, self).__init__()

        

        # Load imagenet pre-trained model 

        self.effNet = EfficientNet.from_pretrained('efficientnet-b1', in_channels=1).to(device)

        

        # Appdend output layers based on our date

        self.fc_root = nn.Linear(in_features=1000, out_features=168)

        self.fc_vowel = nn.Linear(in_features=1000, out_features=11)

        self.fc_consonant = nn.Linear(in_features=1000, out_features=7)

        

    def forward(self, X):

        output = self.effNet(X)

        output_root = self.fc_root(output)

        output_vowel = self.fc_vowel(output)

        output_consonant = self.fc_consonant(output)

        

        return output_root, output_vowel, output_consonant
%%capture

model = EfficientNetWrapper().to(device)
LEARNING_RATE = 0.02

EPOCHS = 150

CUTMIX_ALPHA = 1
model = nn.DataParallel(model)

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0.00001, eps=1e-08)

criterion = nn.CrossEntropyLoss()
def clear_cache():

    gc.collect()

    torch.cuda.empty_cache()
def get_accuracy(root_preds, target_root, vowel_pred, target_vowel, consonant_pred, target_consonant):

    assert len(root_preds) == len(target_root) and len(vowel_pred) == len(target_vowel) and len(consonant_pred) == len(target_consonant)

    

    total = len(target_root) + len(target_vowel) + len(target_consonant)

    _, predicted_root = torch.max(root_preds.data, axis=1)

    _, predicted_vowel = torch.max(vowel_pred.data, axis=1)

    _, predicted_consonant = torch.max(consonant_pred.data, axis=1)

    

    del root_preds

    del vowel_pred

    del consonant_pred

    torch.cuda.empty_cache()



    correct = (predicted_root == target_root).sum().item() + (predicted_vowel == target_vowel).sum().item() + (predicted_consonant == target_consonant).sum().item()

    

    del target_root

    del target_vowel

    del target_consonant

    torch.cuda.empty_cache()

    return correct / total
total_steps = len(train_loader)

val_acc_list = []

plot_flag = True

for epoch in range(EPOCHS):

    for i, (x_train, target_root, target_vowel, target_consonant) in enumerate(train_loader):

        x_train = x_train.to(device, dtype=torch.float)

        target_root = target_root.to(device)

        target_vowel = target_vowel.to(device)

        target_consonant = target_consonant.to(device)



        if plot_flag:

            plot_flag = False

            plot_images(x_train)

        

        # Forward pass

        root_preds, vowel_pred, consonant_pred = model(x_train)

        loss = (criterion(root_preds, target_root) + criterion(vowel_pred, target_vowel) + criterion(consonant_pred, target_consonant)) / 3

        

        del x_train

        clear_cache()

        

        # Backpropagate

        optimizer.zero_grad()  # Reason: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

        loss.backward()

        optimizer.step()

    

#     lr_scheduler.step(loss.item())

    del root_preds

    del target_root

    del vowel_pred

    del target_vowel

    del consonant_pred

    del target_consonant

    clear_cache()



    # Calculate validation accuracy after each epoch

    # Predict on validation set

    root_val_preds, vowel_val_pred, consonant_val_pred = model(X_val)



    val_acc = get_accuracy(root_val_preds, Y_val_root, vowel_val_pred, Y_val_vowel, consonant_val_pred, Y_val_consonant)

    val_acc_list.append(val_acc)



    del root_val_preds

    del vowel_val_pred

    del consonant_val_pred

    clear_cache()



    print('Epoch [{}/{}], Loss: {:.4f}, Validation accuracy: {:.2f}%'

          .format(epoch + 1, EPOCHS, loss.item(), val_acc * 100))
plt.style.use('ggplot')

plt.figure()

plt.plot(np.arange(0, EPOCHS), val_acc_list, label='val_accuracy')



plt.title('Accuracy')

plt.xlabel('# of epochs')

plt.ylabel('Accuracy')

plt.legend(loc='upper right')

plt.show()