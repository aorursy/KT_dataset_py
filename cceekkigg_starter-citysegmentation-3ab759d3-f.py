import os

import torch

import numpy as np

from PIL import Image

import torch.utils.data as data

from collections import namedtuple
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
class CitysSegmentation(data.Dataset):



    Sizedict = {

        'Berlin' : (8, 6, 628, 800),

        'London' : (8, 6, 536, 536),

    }



    def __init__(self, root, city, images=None, labels=None, image_set='train'):

        super(CitysSegmentation).__init__()

        self.root = root

        self.city = city

        self.images = images

        self.labels = labels

        self.image_set = image_set

        

        self.nx, self.ny, self.sx, self.sy = self.Sizedict[self.city] 

        self.Ntile = len(self.images) #number of tiles in dataset

        self.Nsample = self.nx*self.ny #number of samples in tiles

        self.Ndata = self.Ntile * self.Nsample # number of data(samples) in dataset

        

        if image_set=='train' or image_set=='test':

            self.n_images = len(self.images)

        else:

            raise RuntimeError('Key value setting error, image_set can only be "train" or "test"')

            

    def __getitem__(self, index):

        """

        Args:

            index (int): Index

        Returns:

            tuple: (image, label)

        """

        img_index = index//self.Nsample    #tile idx in the dataset

        sample_index = index%self.Nsample  #sample idx in the tile

        

        assert index<self.Ndata, '%s image index exceed sample size(%s)' %(self.image_set, self.Ndata)

        image = Image.open(self.images[img_index]).convert('RGB')

        label = Image.open(self.labels[img_index])

            

            

        tx_idx = sample_index%self.nx

        ty_idx = sample_index//self.nx

        timage = np.array(image)[ty_idx*self.sy:(ty_idx+1)*self.sy, tx_idx*self.sx:(tx_idx+1)*self.sx]

        tlabel = np.array(label)[ty_idx*self.sy:(ty_idx+1)*self.sy, tx_idx*self.sx:(tx_idx+1)*self.sx]

            

        return timage, tlabel



    def __len__(self):

        return len(self.images)

def show(x, outfile=None):

    if isinstance(x, np.ndarray):

        img = Image.fromarray(x)

    if outfile is not None:

        Image.fromarray(x).save(outfile, "PNG")

    return img



def load_dataset(root, city, n_train=None, n_test=None, 

                 image_format='_raster.png', label_format='_label.png'):



    images_dir = os.path.join(root, city, 'raster')

    labels_dir = os.path.join(root, city, 'label')

    images = []

    labels = []



    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):

        raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders'

                            ' are inside the "root" directory')



    for file in os.listdir(labels_dir):

        if label_format not in file:

            continue

        img_name = file.split('_label')[0]

        images.append(os.path.join(images_dir, img_name+image_format))

        labels.append(os.path.join(labels_dir, file))

    images = np.array(images)

    labels = np.array(labels)



    #train_test_split

    train = []

    test = []

    assert len(images)==len(labels), 'dataset error with inequal rasters and labels'



    idx_arr = np.arange(len(images))

    np.random.shuffle(idx_arr)

    if n_train is None:

        n_test = len(idx_arr)//3

        n_train = len(idx_arr)-n_test

    train = idx_arr[n_test:]

    test = idx_arr[:n_test]



    #generate train and test data

    train_gen = CitysSegmentation(root, city, images=images[train], labels=labels[train], image_set='train')

    test_gen = CitysSegmentation(root, city, images=images[test], labels=labels[test], image_set='test')



    return train_gen, test_gen
if __name__ == "__main__":

    train_gen, test_gen = load_dataset(root='/kaggle/input/berlin-aoi-dataset/data_CitySegmentation', city='Berlin')

    print(train_gen.Ndata, test_gen.Ndata)

    img, lab = train_gen[train_gen.Ndata-1]

    print(np.array(img).shape)
show(img)
show(lab)