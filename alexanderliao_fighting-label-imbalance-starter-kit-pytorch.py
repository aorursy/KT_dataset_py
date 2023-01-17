import sys
import os
print(sys.path)
sys.path.append('../../')
#from dependencies import DATA, CODE
from tqdm import tqdm
#from vision.augmentations import compute_center_pad, do_gamma
import cv2
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
import torch
import gc
DATA = '.'
CODE = '.'
class ProteinDatasetLoad(Dataset):

    def __init__(self, split, augment=null_augment, mode='train',size=256):
        super(ProteinDatasetLoad, self).__init__()
        self.split   = split
        self.mode   = mode
        self.augment = augment
        self.size = size
        self.truth = pd.read_csv(DATA+'train.csv')
        self.problist = [(27, 0.125),
                         (15, 0.05555555555555555),
                         (10, 0.043478260869565216),
                         (9, 0.02702702702702703),
                         (8, 0.022222222222222223),
                         (20, 0.007633587786259542),
                         (17, 0.006024096385542169),
                         (24, 0.003875968992248062),
                         (26, 0.003816793893129771),
                         (13, 0.002352941176470588),
                         (16, 0.002288329519450801),
                         (12, 0.0017482517482517483),
                         (22, 0.001579778830963665),
                         (18, 0.0013679890560875513),
                         (6, 0.001221001221001221),
                         (11, 0.001141552511415525),
                         (14, 0.0011312217194570137),
                         (1, 0.0010183299389002036),
                         (19, 0.0008665511265164644),
                         (3, 0.0008285004142502071),
                         (4, 0.0006844626967830253),
                         (5, 0.0004935834155972359),
                         (7, 0.0004434589800443459),
                         (23, 0.0004187604690117253),
                         (2, 0.00034916201117318437),
                         (21, 0.00033090668431502316),
                         (25, 0.00015130882130428205),
                         (0, 9.693679720822024e-05)]
        self.label_dict = {0:  'Nucleoplasm',
                        1:  'Nuclear membrane',
                        2:  'Nucleoli',
                        3:  'Nucleoli fibrillar center',
                        4:  'Nuclear speckles',
                        5:  'Nuclear bodies',
                        6:  'Endoplasmic reticulum',
                        7:  'Golgi apparatus',
                        8:  'Peroxisomes',
                        9:  'Endosomes',
                        10:  'Lysosomes',
                        11:  'Intermediate filaments',
                        12:  'Actin filaments',
                        13:  'Focal adhesion sites',
                        14:  'Microtubules',
                        15:  'Microtubule ends',
                        16:  'Cytokinetic bridge',
                        17:  'Mitotic spindle',
                        18:  'Microtubule organizing center',
                        19:  'Centrosome',
                        20:  'Lipid droplets',
                        21:  'Plasma membrane',
                        22:  'Cell junctions',
                        23:  'Mitochondria',
                        24:  'Aggresome',
                        25:  'Cytosol',
                        26:  'Cytoplasmic bodies',
                        27:  'Rods & rings' }

        split_file =  CODE + '/datasets/Protein/splits/' + split
        lines = read_list_from_file(split_file)

        self.ids    = []

        for l in tqdm(lines):                                                                                                                                                                 
            folder, name = l.split('/')
            self.ids.append(name)

        print(len(lines))
        def save_to_dir(i):
           if(i%1000==0): print("loaded sample {}".format(i))
           folder, name = lines[i].split('/')
           image_file = DATA+folder+"/" + name
           if not(os.path.isfile(DATA+folder+ "_np"+ str(size) +"/"+name+".npy")):
              r = cv2.imread(image_file+'_red.png',cv2.IMREAD_GRAYSCALE).astype("uint8")
              g = cv2.imread(image_file+'_green.png',cv2.IMREAD_GRAYSCALE).astype("uint8")
              b = cv2.imread(image_file+'_blue.png',cv2.IMREAD_GRAYSCALE).astype("uint8")
              y = cv2.imread(image_file+'_yellow.png',cv2.IMREAD_GRAYSCALE).astype("uint8")
              r = cv2.resize(r,(size,size))
              g = cv2.resize(g,(size,size))
              b = cv2.resize(b,(size,size))
              y = cv2.resize(y,(size,size))

              image = np.dstack((r,g,b,y))
              np.save(DATA+folder+ "_np"+ str(size) +"/"+name,image)

           else:

              image = np.load(DATA+folder+ "_np"+ str(size) +"/"+name+".npy")

        num_cores = multiprocessing.cpu_count()
        if self.split.find("valid")==-1:
            Parallel(n_jobs=num_cores, prefer="threads")(delayed(save_to_dir)(i) for i in range(len(lines)))
        else:
            import random
            self.images=Parallel(n_jobs=num_cores, prefer="threads")(delayed(save_to_dir)(i) for i in range(len(lines)))


        self.annotations  = []
        self.probweights = []
        if self.mode in ['train','valid']:
            for l in tqdm(lines):
                folder, file = l.split('/')
                self.folder = folder

                label_encode = self.truth.loc[self.truth['Id']==file]['Target']
                label_list = [[int(i) for i in s.split()] for s in label_encode]

                for tuplea in self.problist:
                    if tuplea[0] in label_list[0]:
                        self.probweights.append(tuplea[1])
                        break
                self.annotations.append( np.eye(len(self.label_dict),dtype=np.float)[label_list].sum(axis=0))
        elif self.mode in ['test']:
            self.annotations  = [[] for l in lines]

        #-------
        print('\tProteinDataset')
        print('\tsplit            = %s'%split)
        print('\tlen(self.ids) = %d'%len(self.ids))
        print('')


    def __getitem__(self, index):
        image = np.load(DATA+self.folder+"_np"+str(self.size)+"/"+self.ids[index]+".npy")
        annotations  = self.annotations[index]

        return self.augment(image, annotations, index)

    def __len__(self):
        return len(self.ids)

                                                     