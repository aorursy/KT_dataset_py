import os

from os import sys



import imageio

import numpy as np

import pandas as pd

from PIL import Image

from tqdm import tqdm_notebook as tqdm



import torch

import torch.utils.data as D



import torchvision

from torchvision import transforms as T



# Create a dataframe including one experiment's worth of samples

df = pd.read_csv('../input/train.csv').head(1106)  # HEPG-01
# First case: don't do any preprocessing

# Modeled on https://www.kaggle.com/leighplt/densenet121-pytorch

# This also serves as a base class for the datasets in cases 2 and 3.

class ImageDS(D.Dataset):

    def __init__(self, df, img_dir, channels=[1,2,3,4,5,6]):

        self.records = df.to_records(index=False)

        self.channels = channels

        self.img_dir = img_dir

        self.len = df.shape[0] * 2

        

    @staticmethod

    def _load_img_as_tensor(file_name):

        with Image.open(file_name) as img:

            return T.ToTensor()(img)



    def _get_img_path(self, index, site, channel):

        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate

        return '/'.join([self.img_dir,experiment,f'Plate{plate}',f'{well}_s{site}_w{channel}.png'])

    

    def _load_data(self, index, site):

        paths = [self._get_img_path(index, site, ch) for ch in self.channels]

        # Although we're normalizing here, the computational cost is insignificant

        normalize = T.Normalize(

            mean=[0.5] * 6,

            std=[0.5] * 6

        )

        return normalize(torch.cat([self._load_img_as_tensor(img_path) for img_path in paths]))

    

    def __getitem__(self, index):

        site = (index % 2) + 1

        index = index // 2

        return self._load_data(index, site)



    def __len__(self):

        """

        Total number of samples in the dataset

        """

        return self.len

    

def loop(data_loader):

    for _ in data_loader:

        pass

    
# Note that we're loading images directly from the input folder.

ds = ImageDS(df, '../input/train')

loader = D.DataLoader(ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

%timeit -r 3 loop(loader) 
!git clone https://github.com/recursionpharma/rxrx1-utils

!mkdir -p processed-data/jpg

sys.path.append('rxrx1-utils')

import rxrx.io as rio
# Option two: Convert to an rgb jpg

# Modified version of https://www.kaggle.com/xhlulu/recursion-2019-load-size-and-resize-images



def convert_to_rgb(df, img_dir='processed-data/jpg/', resize=False, new_size=224, extension='jpeg'):

    N = df.shape[0]

    for i in tqdm(range(N)):

        code = df['id_code'][i]

        experiment = df['experiment'][i]

        plate = df['plate'][i]

        well = df['well'][i]

        for site in [1, 2]:

            save_path = f'{img_dir}{code}_s{site}.{extension}'



            im = rio.load_site_as_rgb(

                'train', experiment, plate, well, site, 

                base_path='../input/'

            )

            im = im.astype(np.uint8)

            im = Image.fromarray(im)

            

            if resize:

                im = im.resize((new_size, new_size), resample=Image.BILINEAR)

            im.save(save_path)



class JpgImageDS(ImageDS):

    def __init__(self, df, img_dir):

        super().__init__(df, img_dir)

        

    def _get_img_path(self, index, site):

        code = self.records[index].id_code

        return f'{self.img_dir}{code}_s{site}.jpeg'

    

    def _load_data(self, index, site):

        return self._load_img_as_tensor(self._get_img_path(index, site))
convert_to_rgb(df)
# Option two: Load jpegs

ds = JpgImageDS(df, 'processed-data/jpg/')

loader = D.DataLoader(ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

%timeit -r 3 loop(loader)

!du -sh ../input/train/HEPG2-01/

!du -sh processed-data/jpg/
# Option 3: Create preprocessed numpy files.

# Code mostly from https://www.kaggle.com/gidutz/starter-kernel-recursion-pharmaceuticals



BASE_DIR = '../input'

OUTPUT_DIR = 'processed-data/npy/'

DATA_PATH_FORMAT = os.path.join(BASE_DIR, 'train/{experiment}/Plate{plate}/{well}_s{sample}_w{channel}.png')



df_pixel_stats = pd.read_csv(os.path.join(BASE_DIR, 'pixel_stats.csv')).set_index(['id_code','site', 'channel'])



def transform_image(sample_data, pixel_data, site):

    x=[]

    for channel in [1,2,3,4,5,6]:

        impath = DATA_PATH_FORMAT.format(experiment=sample.experiment,

                                        plate=sample_data.plate,

                                        well=sample_data.well,

                                        sample=site,

                                        channel=channel)

        # normalize the channel

        img = np.array(imageio.imread(impath)).astype(np.float64)

        img -= pixel_data.loc[channel]['mean']

        img /= pixel_data.loc[channel]['std']

        img *= 255 # To keep MSB



        x.append(img)



    return np.stack(x).T.astype(np.byte)





!mkdir -p {OUTPUT_DIR}

for _, sample in tqdm(df.iterrows(), total=len(df)):

    for site in [1, 2]:

        pixel_data = df_pixel_stats.loc[sample.id_code, site, :].reset_index().set_index('channel')

        x = transform_image(sample, pixel_data, site)

        np.save(os.path.join(OUTPUT_DIR, '{sample_id}_s{site}.npy').format(sample_id=sample.id_code, site=site), x)
class NpyImageDS(ImageDS):

    def __init__(self, df, img_dir):

        super().__init__(df, img_dir)

        

    def _get_img_path(self, index, site):

        sample_id = self.records[index].id_code

        return f'{self.img_dir}{sample_id}_s{site}.npy'

    

    def _load_data(self, index, site):

        return torch.Tensor(np.load(self._get_img_path(index, site)).astype(np.float32)/ 255.0)
ds = NpyImageDS(df, OUTPUT_DIR)

loader = D.DataLoader(ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

%timeit -r 3 loop(loader) 
!du -sh ../input/train/HEPG2-01/

!du -sh processed-data/npy/
!rm -r processed-data

!mkdir -p processed-data/raw/HEPG2-01

!cp -r ../input/train/HEPG2-01/ processed-data/raw/HEPG2-01/

ds = ImageDS(df, 'processed-data/raw/HEPG2-01')

loader = D.DataLoader(ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

%timeit -r 1 loop(loader) 
# Apparently these directories need to be removed to avoid an error.

!rm -r  rxrx1-utils

!rm -r processed-data