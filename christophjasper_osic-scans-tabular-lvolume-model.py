import numpy as np
import pandas as pd
import pydicom
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
ROOT = "../input/osic-pulmonary-fibrosis-progression"
BATCH_SIZE=128
chunk = pd.read_csv(f"{ROOT}/test.csv")

sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]
sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")
chunk['WHERE'] = 'val'
sub['WHERE'] = 'test'
data = chunk
data['min_week'] = data['Weeks']
data.loc[data.WHERE=='test','min_week'] = np.nan
data['min_week'] = data.groupby('Patient')['min_week'].transform('min')
base = data.loc[data.Weeks == data.min_week]
base = base[['Patient','FVC']].copy()
base.columns = ['Patient','min_FVC']
base['nb'] = 1
base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
base = base[base.nb==1]
base.drop('nb', axis=1, inplace=True)
data = data.merge(base, on='Patient', how='left')
data = sub.merge(data.drop(['FVC','Percent', 'Age', 'Sex',
                            'SmokingStatus', 'WHERE', 'Weeks'], axis=1), on='Patient')
data['base_week'] = data['Weeks'] - data['min_week']
del base
from skimage.measure import label,regionprops
from skimage.segmentation import clear_border
from multiprocessing import Pool
class Detector:
    def __call__(self, x):
        raise NotImplementedError('Abstract') 

class ThrDetector(Detector):
    def __init__(self, thr=-400):
        self.thr = thr
        
    def __call__(self, x):
        x = pydicom.dcmread(x)
        img = x.pixel_array
        img = (img + x.RescaleIntercept) / x.RescaleSlope
        img = img < self.thr
        
        img = clear_border(img)
        img = label(img)
        areas = [r.area for r in regionprops(img)]
        areas.sort()
        if len(areas) > 2:
            for region in regionprops(img):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                        img[coordinates[0], coordinates[1]] = 0
        img = img > 0
        return np.int32(img)
  

class Integral:
    def __init__(self, detector: Detector):
        self.detector = detector
    
    def __call__(self, xs):
        raise NotImplementedError('Abstract')
        

class MeanIntegral(Integral):
    def __call__(self, xs):
        with Pool(4) as p:
            masks = p.map(self.detector, xs) 
        return np.mean(masks)
test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
test_data = {}
for p in test.Patient.values:
    test_data[p] = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/test/{p}/')
import gc
gc.collect()
integral = MeanIntegral(ThrDetector()) 
keys = [k for k in list(test_data.keys())]
def chunks(lst, n):
    chunk = []
    width = len(lst)//n
    count = 0
    for i in range(0, len(lst), width):
        if count == n-1:
            chunk.append(lst[i:len(lst)])
            break
        else:
            chunk.append(lst[i:i + width])
        count += 1
    return chunk

cus = chunks(keys, 4)
len(cus)
from joblib import delayed, Parallel, parallel_backend
volume = {}
def process(keylist):
    for k in tqdm(keylist, total=len(keylist)):
        x = []
        for i in test_data[k]:
            x.append(f'../input/osic-pulmonary-fibrosis-progression/test/{k}/{i}') 
        volume[k] = integral(x)
        x.clear()
        
Parallel(n_jobs=4, require='sharedmem')(
    delayed(process)(cus[i]) for i in range(len(cus)))
test_volume = pd.DataFrame.from_dict(volume, orient="index").reset_index()
test_volume = test_volume.rename(columns={"index": "Patient", 0: "Volume"})
for i in range(test_volume.shape[0]):
    data.loc[data['Patient'] == test_volume["Patient"][i], "Volume"] = test_volume["Volume"][i]
data = data.drop_duplicates()
data.isna().sum()
mean = data["Volume"].mean()
data["Volume"] = data["Volume"].fillna(mean)
import copy
from datetime import timedelta, datetime
import imageio
from matplotlib import cm
import numpy as np
from pathlib import Path
import pytest
import scipy.ndimage as ndimage
from scipy.ndimage.interpolation import zoom
from skimage import measure, morphology, segmentation
from time import time, sleep
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import warnings
test_dir = '/kaggle/input/osic-pulmonary-fibrosis-progression/test/'
model_dir = '/kaggle/input/encoder/diophantus.pt'
dest_dir = '/kaggle/working/dest/'
resize_dims = (40, 256, 256)
clip_bounds = (-1000, 200)
watershed_iterations = 1
pre_calculated_mean = 0.02865046213070556
latent_features = 10
batch_size = 16
learning_rate = 3e-5
num_epochs = 40
val_size = 0.2
!mkdir '/kaggle/working/dest/'
class CTScansDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.patients = [p for p in self.root_dir.glob('*') if p.is_dir()]
        self.transform = transform

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, metadata = self.load_scan(self.patients[idx])
        sample = {'image': image, 'metadata': metadata}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def save(self, path):
        t0 = time()
        Path(path).mkdir(exist_ok=True, parents=True)
        print('Saving pre-processed dataset to disk')
        sleep(1)
        cum = 0

        bar = trange(len(self))
        for i in bar:
            sample = self[i]
            image, data = sample['image'], sample['metadata']
            cum += torch.mean(image).item()

            bar.set_description(f'Saving CT scan {data.PatientID}')
            fname = Path(path) / f'{data.PatientID}.pt'
            torch.save(image, fname)

        sleep(1)
        bar.close()
        print(f'Done! Time {timedelta(seconds=time() - t0)*150}\n'
              f'Mean value: {cum / len(self)}')

    def get_patient(self, patient_id):
        patient_ids = [str(p.stem) for p in self.patients]
        return self.__getitem__(patient_ids.index(patient_id))

    @staticmethod
    def load_scan(path):
        slices = [pydicom.read_file(p) for p in path.glob('*.dcm')]
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            warnings.warn(f'Patient {slices[0].PatientID} CT scan does not '
                          f'have "ImagePositionPatient". Assuming filenames '
                          f'in the right scan order.')

        image = np.stack([s.pixel_array.astype(float) for s in slices])
        return image, slices[0]
class CropBoundingBox:
    @staticmethod
    def bounding_box(img3d: np.array):
        mid_img = img3d[int(img3d.shape[0] / 2)]
        same_first_row = (mid_img[0, :] == mid_img[0, 0]).all()
        same_first_col = (mid_img[:, 0] == mid_img[0, 0]).all()
        if same_first_col and same_first_row:
            return True
        else:
            return False

    def __call__(self, sample):
        image, data = sample['image'], sample['metadata']
        if not self.bounding_box(image):
            return sample

        mid_img = image[int(image.shape[0] / 2)]
        r_min, r_max = None, None
        c_min, c_max = None, None
        for row in range(mid_img.shape[0]):
            if not (mid_img[row, :] == mid_img[0, 0]).all() and r_min is None:
                r_min = row
            if (mid_img[row, :] == mid_img[0, 0]).all() and r_max is None \
                    and r_min is not None:
                r_max = row
                break

        for col in range(mid_img.shape[1]):
            if not (mid_img[:, col] == mid_img[0, 0]).all() and c_min is None:
                c_min = col
            if (mid_img[:, col] == mid_img[0, 0]).all() and c_max is None \
                    and c_min is not None:
                c_max = col
                break

        image = image[:, r_min:r_max, c_min:c_max]
        return {'image': image, 'metadata': data}
class ConvertToHU:
    def __call__(self, sample):
        image, data = sample['image'], sample['metadata']

        img_type = data.ImageType
        is_hu = img_type[0] == 'ORIGINAL' and not (img_type[2] == 'LOCALIZER')

        intercept = data.RescaleIntercept
        slope = data.RescaleSlope
        image = (image * slope + intercept).astype(np.int16)
        return {'image': image, 'metadata': data}
class Resize:
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        image, data = sample['image'], sample['metadata']
        resize_factor = np.array(self.output_size) / np.array(image.shape)
        image = zoom(image, resize_factor, mode='nearest')
        return {'image': image, 'metadata': data}
class Clip:
    def __init__(self, bounds=(-1000, 500)):
        self.min = min(bounds)
        self.max = max(bounds)

    def __call__(self, sample):
        image, data = sample['image'], sample['metadata']
        image[image < self.min] = self.min
        image[image > self.max] = self.max
        return {'image': image, 'metadata': data}
class MaskWatershed:
    def __init__(self, min_hu, iterations, show_tqdm):
        self.min_hu = min_hu
        self.iterations = iterations
        self.show_tqdm = show_tqdm

    def __call__(self, sample):
        image, data = sample['image'], sample['metadata']

        stack = []
        if self.show_tqdm:
            bar = trange(image.shape[0])
            bar.set_description(f'Masking CT scan {data.PatientID}')
        else:
            bar = range(image.shape[0])
        for slice_idx in bar:
            sliced = image[slice_idx]
            stack.append(self.seperate_lungs(sliced, self.min_hu,
                                             self.iterations))

        return {
            'image': np.stack(stack),
            'metadata': sample['metadata']
        }

    @staticmethod
    def seperate_lungs(image, min_hu, iterations):
        h, w = image.shape[0], image.shape[1]

        marker_internal, marker_external, marker_watershed = MaskWatershed.generate_markers(image)

        sobel_filtered_dx = ndimage.sobel(image, 1)
        sobel_filtered_dy = ndimage.sobel(image, 0)
        sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
        sobel_gradient *= 255.0 / np.max(sobel_gradient)

        watershed = morphology.watershed(sobel_gradient, marker_watershed)

        outline = ndimage.morphological_gradient(watershed, size=(3,3))
        outline = outline.astype(bool)


        blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 0, 1, 1, 1, 0, 0]]

        blackhat_struct = ndimage.iterate_structure(blackhat_struct, iterations)

        outline += ndimage.black_tophat(outline, structure=blackhat_struct)

        lungfilter = np.bitwise_or(marker_internal, outline)
        lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)

        segmented = np.where(lungfilter == 1, image, min_hu * np.ones((h, w)))

        return segmented 

    @staticmethod
    def generate_markers(image, threshold=-400):
        h, w = image.shape[0], image.shape[1]

        marker_internal = image < threshold
        marker_internal = segmentation.clear_border(marker_internal)
        marker_internal_labels = measure.label(marker_internal)

        areas = [r.area for r in measure.regionprops(marker_internal_labels)]
        areas.sort()

        if len(areas) > 2:
            for region in measure.regionprops(marker_internal_labels):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                        marker_internal_labels[coordinates[0], coordinates[1]] = 0

        marker_internal = marker_internal_labels > 0

        external_a = ndimage.binary_dilation(marker_internal, iterations=10)
        external_b = ndimage.binary_dilation(marker_internal, iterations=55)
        marker_external = external_b ^ external_a

        marker_watershed = np.zeros((h, w), dtype=np.int)
        marker_watershed += marker_internal * 255
        marker_watershed += marker_external * 128

        return marker_internal, marker_external, marker_watershed
class Normalize:
    def __init__(self, bounds=(-1000, 500)):
        self.min = min(bounds)
        self.max = max(bounds)

    def __call__(self, sample):
        image, data = sample['image'], sample['metadata']
        image = image.astype(np.float)
        image = (image - self.min) / (self.max - self.min)
        return {'image': image, 'metadata': data}
    

class ToTensor:
    def __init__(self, add_channel=True):
        self.add_channel = add_channel

    def __call__(self, sample):
        image, data = sample['image'], sample['metadata']
        if self.add_channel:
            image = np.expand_dims(image, axis=0)

        return {'image': torch.from_numpy(image), 'metadata': data}
    
    
class ZeroCenter:
    def __init__(self, pre_calculated_mean):
        self.pre_calculated_mean = pre_calculated_mean

    def __call__(self, tensor):
        return tensor - self.pre_calculated_mean
def show(list_imgs, cmap=cm.bone):
    list_slices = []
    for img3d in list_imgs:
        slc = int(img3d.shape[0] / 2)
        img = img3d[slc]
        list_slices.append(img)
    
    fig, axs = plt.subplots(1, 5, figsize=(15, 7))
    for i, img in enumerate(list_slices):
        axs[i].imshow(img, cmap=cmap)
        axs[i].axis('off')
        
    plt.show()
scans = CTScansDataset(
    root_dir=test_dir,
    transform=transforms.Compose([
        CropBoundingBox(),
        ConvertToHU(),
        Resize(resize_dims),
        Clip(bounds=clip_bounds),
        MaskWatershed(
            min_hu=min(clip_bounds),
            iterations=watershed_iterations,
            show_tqdm=False),
        Normalize(bounds=clip_bounds),
        ToTensor()
    ]))
scans.save(dest_dir)
class CTTensorsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.tensor_files = sorted([f for f in self.root_dir.glob('*.pt')])
        self.transform = transform

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        image = torch.load(self.tensor_files[item])
        if self.transform:
            image = self.transform(image)

        return {
            'patient_id': self.tensor_files[item].stem,
            'image': image
        }

    def mean(self):
        cum = 0
        for i in range(len(self)):
            sample = self[i]['image']
            cum += torch.mean(sample).item()

        return cum / len(self)

    def random_split(self, val_size: float):
        num_val = int(val_size * len(self))
        num_train = len(self) - num_val
        return random_split(self, [num_train, num_val])
scans = CTTensorsDataset(
    root_dir=dest_dir2,
    transform=ZeroCenter(pre_calculated_mean=pre_calculated_mean)
)
class VarAutoEncoder(nn.Module):
    def __init__(self, latent_features=latent_features):
        super(VarAutoEncoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv3d(1, 16, 3)
        self.conv2 = nn.Conv3d(16, 32, 3)
        self.conv3 = nn.Conv3d(32, 96, 2)
        self.conv4 = nn.Conv3d(96, 1, 1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=3, return_indices=True)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.fc1 = nn.Linear(10 * 10, latent_features)
        self.fc2 = nn.Linear(10 * 10, latent_features)
        
        # Decoder
        self.fc3 = nn.Linear(latent_features, 10 * 10)
        self.deconv0 = nn.ConvTranspose3d(1, 96, 1)
        self.deconv1 = nn.ConvTranspose3d(96, 32, 2)
        self.deconv2 = nn.ConvTranspose3d(32, 16, 3)
        self.deconv3 = nn.ConvTranspose3d(16, 1, 3)
        self.unpool0 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.unpool1 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.unpool2 = nn.MaxUnpool3d(kernel_size=3, stride=3)
        self.unpool3 = nn.MaxUnpool3d(kernel_size=2, stride=2)

    def encode(self, x, return_partials=True):
        # Encoder
        x = self.conv1(x)
        up3out_shape = x.shape
        x, i1 = self.pool1(x)

        x = self.conv2(x)
        up2out_shape = x.shape
        x, i2 = self.pool2(x)

        x = self.conv3(x)
        up1out_shape = x.shape
        x, i3 = self.pool3(x)

        x = self.conv4(x)
        up0out_shape = x.shape
        x, i4 = self.pool4(x)

        x = x.view(-1, 10 * 10)
        
        mu = self.fc1(x)
        log_var = self.fc2(x)
        
        if return_partials:
            
            return mu, log_var, up3out_shape, i1, up2out_shape, i2, up1out_shape, i3, \
                   up0out_shape, i4

        else:
            return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, log_var, up3out_shape, i1, up2out_shape, i2, \
        up1out_shape, i3, up0out_shape, i4 = self.encode(x)
        
        z = self.reparameterize(mu, log_var)
       
        # Decoder
        x = F.relu(self.fc3(z))
        x = x.view(-1, 1, 1, 10, 10)
        x = self.unpool0(x, output_size=up0out_shape, indices=i4)
        x = self.deconv0(x)
        x = self.unpool1(x, output_size=up1out_shape, indices=i3)
        x = self.deconv1(x)
        x = self.unpool2(x, output_size=up2out_shape, indices=i2)
        x = self.deconv2(x)
        x = self.unpool3(x, output_size=up3out_shape, indices=i1)
        x = self.deconv3(x)

        return x, mu, log_var
device = "cpu"
model = VarAutoEncoder(latent_features=latent_features).to(device) 
model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
model.eval()
for i in range(len(scans)):
    pid = scans[i]['patient_id']
    img = scans[i]['image'].unsqueeze(0).float().to(device)
    with torch.no_grad():
        mu, log_var = model.encode(img, return_partials=False)
        bg = log_var.squeeze(0)[0]
        for j in range(latent_features):
            data.loc[data['Patient'] == pid, f"l_feature_{j}"] = log_var.squeeze(0)[j].item()
COLS1 = 'Sex'
FE = []
new_cols1 = np.array(['Male', 'Female'])
for mod in new_cols1:
    FE.append(mod)
    data[mod] = (data[COLS1] == mod).astype(int)

COLS2 = 'SmokingStatus'
new_cols2 = np.array(['Ex-smoker', 'Never smoked', 'Currently smokes'])
for mod in new_cols2:
    FE.append(mod)
    data[mod] = (data[COLS2] == mod).astype(int)
data['age'] = (data['Age'] - 49) / ( 88 - 49)
data['BASE'] = (data['min_FVC'] - 1015 ) / (6399 - 1015)
data['week'] = (data['base_week'] - data['base_week'].min() ) / ( data['base_week'].max() - data['base_week'].min() )
data['percent'] = (data['Percent'] - 28.87 ) / ( 153.15 - 28.87)
FE += ['age','percent','week','BASE', 'Volume', "l_feature_0", "l_feature_1", "l_feature_2", "l_feature_3", "l_feature_4",
    "l_feature_5", "l_feature_6", "l_feature_7", "l_feature_8", "l_feature_9"]
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import sys
C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")

def score(y_true, y_pred):
    y_true = tf.compat.v1.to_float(y_true)
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32, dtype_hint=None, name=None)
    y_pred = tf.compat.v1.to_float(y_pred)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32, dtype_hint=None, name=None)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return K.mean(metric)

def qloss(y_true, y_pred):
    y_true = tf.compat.v1.to_float(y_true)
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32, dtype_hint=None, name=None)
    y_pred = tf.compat.v1.to_float(y_pred)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32, dtype_hint=None, name=None)
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)

def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)
    return loss

def make_model(nh):
    z = L.Input((nh,), name="Patient")
    x = L.Dense(100, activation="relu", name="d1")(z)
    x = L.Dense(50, activation="relu", name="d2")(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="relu", name="p2")(x)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 
                     name="preds")([p1, p2])
    
    model = M.Model(z, preds, name="CNN")
    model.compile(loss=mloss(0.8), optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False), metrics=[score])
    return model
ze = data[FE].values
nh = ze.shape[1]
pe = np.zeros((ze.shape[0], 3))
net1 = make_model(nh)
net2 = make_model(nh)
net3 = make_model(nh)
net4 = make_model(nh)
net5 = make_model(nh)
checkpoint_path1 = "/kaggle/input/precise/preciseNH20_fold_1.ckpt"
checkpoint_path2 = "/kaggle/input/precise/preciseNH20_fold_2.ckpt"
checkpoint_path3 = "/kaggle/input/precise/preciseNH20_fold_3.ckpt"
checkpoint_path4 = "/kaggle/input/precise/preciseNH20_fold_4.ckpt"
checkpoint_path5 = "/kaggle/input/precise/preciseNH20_fold_5.ckpt"
net1.load_weights(checkpoint_path1)
net2.load_weights(checkpoint_path2)
net3.load_weights(checkpoint_path3)
net4.load_weights(checkpoint_path4)
net5.load_weights(checkpoint_path5)
pe += net1.predict(ze, batch_size=128, verbose=0)/5
pe += net2.predict(ze, batch_size=128, verbose=0)/5
pe += net3.predict(ze, batch_size=128, verbose=0)/5
pe += net4.predict(ze, batch_size=128, verbose=0)/5
pe += net5.predict(ze, batch_size=128, verbose=0)/5
pe.shape
sub['FVC1'] = pe[:, 1]
sub['Confidence1'] = pe[:, 2] - pe[:, 0]
s = sub[["Patient_Week","FVC1","Confidence1"]]
s = s.rename(columns={"FVC1": "FVC", "Confidence1": "Confidence"})
s.to_csv("submission.csv", index=False)