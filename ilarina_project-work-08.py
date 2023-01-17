import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage import morphology, color, io, exposure

import os
import tqdm
import matplotlib.pyplot as plt
import statistics as stat
model_dir = './'
size_suffix = '256'
model_name = 'UNet'
dataset_name = 'JSRT'
experiment_suffix = dataset_name + '_' + model_name + '_' + size_suffix
print(experiment_suffix)

path_to_model = '../input/resfrom7/MODEL_ACC_MAX_JSRT_UNet_256_CPU_Wall_Epoch1.hdf5'

image_zise = 256
## Load JSRT data
## from load_data_JSRT import loadDataJSRT

import numpy as np
from skimage import transform, io, img_as_float, exposure

"""
Data was preprocessed in the following ways:
    - resize to im_shape;
    - equalize histogram (skimage.exposure.equalize_hist);
    - normalize by data set mean and std.
Resulting shape should be (n_samples, img_width, img_height, 1).

It may be more convenient to store preprocessed data for faster loading.

Dataframe should contain paths to images and masks as two columns (relative to `path`).
"""

def loadDataJSRT(df, path, im_shape):
    """This function loads data preprocessed with `preprocess_JSRT.py`"""
    X, y = [], []
    for i, item in df.iterrows():
        img = io.imread(path + item[0])
        img = transform.resize(img, im_shape)
        img = np.expand_dims(img, -1)
        mask = io.imread(path + item[1])
        mask = transform.resize(mask, im_shape)
        mask = np.expand_dims(mask, -1)
        X.append(img)
        y.append(mask)
    X = np.array(X)
    y = np.array(y)
    X -= X.mean()
    X /= X.std()

    print('### Data loaded')
    print('\t{}'.format(path))
    print('\t{}\t{}'.format(X.shape, y.shape))
    print('\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), y.min(), y.max()))
    print('\tX.mean = {}, X.std = {}'.format(X.mean(), X.std()))
    return X, y
def IoU(y_true, y_pred):
    """Returns Intersection over Union score for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)

def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)

def masked(img, gt, mask, alpha=1):
    """Returns image with GT field outlined with red, predicted field filled with blue."""
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    
    #boundary = morphology.dilation(gt, morphology.disk(3)) - gt
    #np.bitwise_xor
    boundary = np.bitwise_xor(morphology.dilation(gt, morphology.disk(3)), gt)
    
    color_mask[mask == 1] = [0, 0, 1]
    color_mask[boundary == 1] = [1, 0, 0]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked

def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img
# Path to csv-file. File should contain X-ray filenames as first column,
# mask filenames as second column.
#csv_path = '../input/idx_JSRT_GOOD_MASKS_SHORT_KAGGLE.csv'
csv_path = '../input/jsrt-original-and-bone-masks/idx_JSRT_GOOD_MASKS_SHORT_KAGGLE.csv'
#csv_path = './idx_JSRT_GOOD_MASKS_SHORT_Titan.csv'
idx_file=pd.read_csv(csv_path)
idx_file.head()    
# Path to the folder with images. Images will be read from path + path_from_csv
# path = csv_path[:csv_path.rfind('/')] + '/'
#path = '/data/JSRT/'
#path = '../input/'

df = pd.read_csv(csv_path)

# Path to the folder with images. Images will be read from path + path_from_csv
path = csv_path[:csv_path.rfind('/')] + '/'

# Load test data
im_shape = (image_zise, image_zise)
X, y = loadDataJSRT(df, path, im_shape)

n_test = X.shape[0]
inp_shape = X[0].shape
%time 

# Load model
model_name = path_to_model
UNet = load_model(model_name)

# For inference standard keras ImageGenerator is used.
test_gen = ImageDataGenerator(rescale=1.)

pr_mask_filename = []
ious = np.zeros(n_test)
dices = np.zeros(n_test)
%time

path_to_predicted = './MASKS_BONES_predicted/'
try:
    os.stat(path_to_predicted)
except:
    os.mkdir(path_to_predicted) 

i = 0
for xx, yy in test_gen.flow(X, y, batch_size=1):
    img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))
    pred = UNet.predict(xx)[..., 0].reshape(inp_shape[:2])
    mask = yy[..., 0].reshape(inp_shape[:2])

    # Binarize masks
    gt = mask > 0.5
    pr = pred > 0.5

    # Remove regions smaller than 2% of the image
    pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))

    head, tail = os.path.split(df.iloc[i][0])
    io.imsave(path_to_predicted + tail, masked(img, gt, pr, 1))

    pr_mask_filename.append(tail)
    ious[i] = IoU(gt, pr)
    dices[i] = Dice(gt, pr)
    print(tail, ious[i], dices[i])

    i += 1
    if i == n_test:
        break

print( 'Mean IoU:', ious.mean() )
print( 'Mean Dice:', dices.mean() )
import pandas as pd
script_dir = "./"
    
file_name = 'IoU_Dice_' + experiment_suffix + '.txt'
file_path = os.path.join(script_dir, file_name)

data_table = pd.DataFrame()
data_table['Filename'] = pd.Series(pr_mask_filename)
data_table['IoU'] = pd.Series(ious)
data_table['Dice'] = pd.Series(dices)
data_table.to_csv(file_path, index=True, header=True)    
%time
#plt.hist([ious],15,label='IoU', alpha=0.75, color='red')
#plt.hist([dices],15,label='Dice', alpha=0.75, color='blue')
plt.hist([ious],10,label='IoU', alpha=0.75, color='red')
plt.hist([dices],10,label='Dice', alpha=0.75, color='blue')

plt.title("Histogram - IoU and Dice for Predicted Bone Masks")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.savefig('Fig_05_IoU_Dice_histo1_' + experiment_suffix + '.png')
plt.show()
plt.close()
print('Bones - Mean: {}; Std: {}'.format(stat.mean(ious), stat.stdev(ious)))
print('Lungs - Mean: {}; Std: {}'.format(stat.mean(dices), stat.stdev(dices)))
%time
plt.hist([ious,dices],10,label=['IoU','Dice'], color=['red','blue'])

plt.title("Histogram - IoU and Dice for Predicted Bone Masks")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.savefig('Fig_05_IoU_Dice_histo2_' + experiment_suffix + '.png')
plt.show()
plt.close()
print('Bones - Mean: {}; Std: {}'.format(stat.mean(ious), stat.stdev(ious)))
print('Lungs - Mean: {}; Std: {}'.format(stat.mean(dices), stat.stdev(dices)))


