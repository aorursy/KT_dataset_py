from pathlib import Path

import os

import numpy as np

from glob import glob

from collections import Counter

from tqdm import tqdm_notebook as tqdm



import h5py

import json

import time

from scipy.misc import imread, imresize

from random import seed, choice, sample



import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

%matplotlib inline

print(os.listdir("../input"))
import torch

import torch.nn as nn

import torchvision.models as models

from torch.nn.utils.rnn import pack_padded_sequence

from torch.utils.data import Dataset, DataLoader

import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
# Data parameters

data_folder = ''  # folder with data files saved by create_input_files.py

data_name = 'flickr8k_5_cap_per_img_5_min_word_freq' # base name shared by data files
def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,

                       max_len=100):

    # Read Karpathy JSON

    with open(karpathy_json_path, 'r') as j:

        data = json.load(j)



    # Read image paths and captions for each image

    train_image_paths = []

    train_image_captions = []

    val_image_paths = []

    val_image_captions = []

    test_image_paths = []

    test_image_captions = []

    word_freq = Counter()



    for img in data['images']:

        captions = []

        for c in img['sentences']:

            # Update word frequency

            word_freq.update(c['tokens'])

            if len(c['tokens']) <= max_len:

                captions.append(c['tokens'])



        if len(captions) == 0:

            continue



        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(

            image_folder, img['filename'])



        if img['split'] in {'train', 'restval'}:

            train_image_paths.append(path)

            train_image_captions.append(captions)

        elif img['split'] in {'val'}:

            val_image_paths.append(path)

            val_image_captions.append(captions)

        elif img['split'] in {'test'}:

            test_image_paths.append(path)

            test_image_captions.append(captions)



    # Sanity check

    assert len(train_image_paths) == len(train_image_captions)

    assert len(val_image_paths) == len(val_image_captions)

    assert len(test_image_paths) == len(test_image_captions)



    # Create word map

    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]

    word_map = {k: v + 1 for v, k in enumerate(words)}

    word_map['<unk>'] = len(word_map) + 1

    word_map['<start>'] = len(word_map) + 1

    word_map['<end>'] = len(word_map) + 1

    word_map['<pad>'] = 0



    # Create a base/root name for all output files

    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'



    # Save word map to a JSON

    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:

        json.dump(word_map, j)



    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files

    seed(123)

    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),

                                   (val_image_paths, val_image_captions, 'VAL'),

                                   (test_image_paths, test_image_captions, 'TEST')]:



        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:

            # Make a note of the number of captions we are sampling per image

            h.attrs['captions_per_image'] = captions_per_image



            # Create dataset inside HDF5 file to store images

            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')



            print("\nReading %s images and captions, storing to file...\n" % split)



            enc_captions = []

            caplens = []



            for i, path in enumerate(tqdm(impaths)):



                # Sample captions

                if len(imcaps[i]) < captions_per_image:

                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]

                else:

                    captions = sample(imcaps[i], k=captions_per_image)



                # Sanity check

                assert len(captions) == captions_per_image



                # Read images

                img = imread(impaths[i])

                if len(img.shape) == 2:

                    img = img[:, :, np.newaxis]

                    img = np.concatenate([img, img, img], axis=2)

                img = imresize(img, (256, 256))

                img = img.transpose(2, 0, 1)

                assert img.shape == (3, 256, 256)

                assert np.max(img) <= 255



                # Save image to HDF5 file

                images[i] = img



                for j, c in enumerate(captions):

                    # Encode captions

                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [

                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))



                    # Find caption lengths

                    c_len = len(c) + 2



                    enc_captions.append(enc_c)

                    caplens.append(c_len)



            # Sanity check

            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)



            # Save encoded captions and their lengths to JSON files

            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:

                json.dump(enc_captions, j)



            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:

                json.dump(caplens, j)

create_input_files(dataset='flickr8k',

                   karpathy_json_path='../input/karpathy-caption/caption_datasets/dataset_flickr8k.json',

                   image_folder='../input/flickr8k-sau/flickr8k-sau/Flickr_Data/Images/',

                   captions_per_image=5,

                   min_word_freq=5,

                   output_folder='',max_len=50)
class CaptionDataset(Dataset):

    """

    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.

    """



    def __init__(self, data_folder, data_name, split, transform=None):

        """

        :param data_folder: folder where data files are stored

        :param data_name: base name of processed datasets

        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'

        :param transform: image transform pipeline

        """

        self.split = split

        assert self.split in {'TRAIN', 'VAL', 'TEST'}



        # Open hdf5 file where images are stored

        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')

        self.imgs = self.h['images']



        # Captions per image

        self.cpi = self.h.attrs['captions_per_image']



        # Load encoded captions (completely into memory)

        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:

            self.captions = json.load(j)



        # Load caption lengths (completely into memory)

        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:

            self.caplens = json.load(j)



        # PyTorch transformation pipeline for the image (normalizing, etc.)

        self.transform = transform



        # Total number of datapoints

        self.dataset_size = len(self.captions)



    def __getitem__(self, i):

        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image

        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)

        if self.transform is not None:

            img = self.transform(img)



        caption = torch.LongTensor(self.captions[i])



        caplen = torch.LongTensor([self.caplens[i]])



        if self.split is 'TRAIN':

            return img, caption, caplen

        else:

            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score

            all_captions = torch.LongTensor(

                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])

            return img, caption, caplen, all_captions



    def __len__(self):

        return self.dataset_size
batch_size = 32

workers = 1 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_loader = DataLoader(CaptionDataset(data_folder, data_name, 'TRAIN',

                                         transform=transforms.Compose([normalize])),

                          batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

val_loader = DataLoader(CaptionDataset(data_folder, data_name, 'VAL',

                                       transform=transforms.Compose([normalize])),

                        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
imgs, caps, caplens  = next(iter(train_loader))
imgs.size(),caps.size(), caplens.size()