import os

import numpy as np

import pandas as pd

import pydicom

import cv2
train_panda = pd.read_csv(os.path.join('/kaggle/input/rsna-intracranial-hemorrhage-detection/','stage_1_train.csv'))

train_panda.iloc[:6]
class dicom_generator:

    def __init__(self,panda,subset='train',batch_size=12):

        self.panda = panda

        self.length = len(panda)

        self.subset = subset

        self.batch_size = batch_size

        self.position = 0

        if (self.subset == 'test'):

            self.subpath = 'stage_1_test_images'

        else:

            self.subpath = 'stage_1_train_images'

            

    def __iter__(self):

        return self

        

    def __next__(self):

        X,y = np.empty((self.batch_size,512,512,1)),[]

        for i in range(self.batch_size):

            filepath = os.path.join('/kaggle/input/rsna-intracranial-hemorrhage-detection/',

                                    self.subpath,"_".join((self.panda['ID'].iloc[self.position]).split("_", 2)[:2])+'.dcm')

            dicom = pydicom.dcmread(filepath).pixel_array

            # here's good place to do your own image augmentation

            if (dicom.shape[0] != 512): # occasionally image sizes in this dataset vary

                dicom = cv2.resize(dicom,dsize=(512,512),interpolation=cv2.INTER_CUBIC)

            X[i] = np.expand_dims(np.expand_dims(dicom,axis=0),axis=3).astype(float)/10

            y.append(self.panda['Label'].iloc[self.position:self.position+6].transpose().to_numpy())

            self.position += 6

            if (self.position >= self.length):

                self.position = 0

        if (self.subset == 'test'):

            return X

        else:

            return (X,np.asarray(y))
my_generator = dicom_generator(train_panda)



example_X, example_y = next(my_generator)

print(example_X.shape)

print(np.asarray(example_y).shape)



from matplotlib import pyplot as plt

%matplotlib inline



plt.imshow(np.squeeze(example_X[0]))