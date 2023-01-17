# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.decomposition import IncrementalPCA



from PIL import Image



import torch

import torch.nn as nn

import torch.utils.data as D

import torch.nn.functional as F



import torchvision

from torchvision import transforms as T



from tqdm import tqdm



import matplotlib.colors as colors

from matplotlib import cm

import matplotlib.gridspec as gridspec

import gc

gc.collect()
data_path='../input'

csv_path='../'
#ImagesDS was taken from https://www.kaggle.com/leighplt/densenet121-pytorch

class ImagesDS(D.Dataset):

    def __init__(self, csv_file, img_dir, mode='train', site=1, channels=[1,2,3,4,5,6]):

        

        df = pd.read_csv(csv_file)

        self.records = df.to_records(index=False)

        self.channels = channels

        self.site = site

        self.mode = mode

        self.img_dir = img_dir

        self.len = df.shape[0]

        

    @staticmethod

    def _load_img_as_tensor(file_name):

        with Image.open(file_name) as img:

            return T.ToTensor()(img)



    def _get_img_path(self, index, channel):

        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate

        return '/'.join([self.img_dir,self.mode,experiment,f'Plate{plate}',f'{well}_s{self.site}_w{channel}.png'])

        

    def __getitem__(self, index):

        paths = [self._get_img_path(index, ch) for ch in self.channels]

        img = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths])

        

        if self.mode == 'train':

            return img, self.records[index].sirna

        else:

            return img, self.records[index].id_code



    def __len__(self):

        """

        Total number of samples in the dataset

        """

        return self.len

    

class ProcessWithIncrementalPCA():

    def __init__(self, data_path,csv_path):

        self.data_path=data_path

        self.csv_path=csv_path

        self.batch_names={'HEPG2':['HEPG2-01','HEPG2-02','HEPG2-03','HEPG2-04','HEPG2-05','HEPG2-06','HEPG2-07'],

                          'HUVEC':['HUVEC-01','HUVEC-02','HUVEC-03','HUVEC-04','HUVEC-05','HUVEC-06','HUVEC-07',

                                  'HUVEC-08','HUVEC-09','HUVEC-10','HUVEC-11','HUVEC-12','HUVEC-13','HUVEC-14',

                                  'HUVEC-15','HUVEC-16'],

                          'RPE':['RPE-01','RPE-02','RPE-03','RPE-04','RPE-05','RPE-06','RPE-07'],

                          'U2OS':['U2OS-01','U2OS-02','U2OS-03']                            

        }



    def create_csv_for_pca(self,nr_of_samples,cell_type):

        df=pd.read_csv(self.data_path+'/train.csv')

        if cell_type=='HEPG2':

            batches=self.batch_names['HEPG2']

        if cell_type=='HUVEC':

            batches=self.batch_names['HUVEC']

        if cell_type=='RPE':

            batches=self.batch_names['RPE']

        if cell_type=='U2OS':

            batches=self.batch_names['U2OS']

        df_lst=[]

        for batch in batches:

            ind=df['experiment']==batch

            sub=np.array(df[ind].index)

            sub_=list(df[ind].index)

            nr_of_samples_in_batch=len(sub_)

            generate_random_numbers=np.random.randint(0,nr_of_samples_in_batch,nr_of_samples)

            sub=sub[generate_random_numbers]

            df_lst=df_lst+list(sub)

        df_=df.loc[df_lst,:]

        df_.to_csv(self.csv_path+'train_pca.csv')



    

    def create_loader(self,nr_of_samples):

        dataset = ImagesDS(self.csv_path+'train_pca.csv', self.data_path)

        loader = D.DataLoader(dataset, batch_size=nr_of_samples, shuffle=False, num_workers=1)

        return loader

        

    def incremental_PCA(self,cell_type,nr_of_samples,nr_components):

        self.create_csv_for_pca(nr_of_samples,cell_type)

        loader=self.create_loader(nr_of_samples)

        pca_res=np.zeros((1,nr_components))

        ipca = IncrementalPCA(n_components=nr_components)

        for x, y in tqdm(loader):

            #Flatten into 1D vector of features for PCA

            x=x.flatten().view(nr_of_samples,1572864).numpy()

            ipca.partial_fit(x)

        loader=self.create_loader(nr_of_samples)

        for x, y in tqdm(loader):

            x=x.flatten().view(nr_of_samples,1572864).numpy()

            tr=ipca.transform(x)

            pca_res=np.vstack((pca_res,tr))

        pca_array=np.array(pca_res)

        components=ipca.components_

        return pca_array[1:,:],components

    

    def plot_PCs(self,pca_array,nr_components):

        data_to_plot=[]

        for i in range(0,nr_components):

            fig = plt.figure(1, figsize=(10, 5))

            for j in range(0,int(pca_array.shape[0]/nr_components)):

                # Create an axes instance

                ax = fig.add_subplot(111)

                # Create the boxplot

                data_to_plot.append(pca_array[j*100:(j+1)*100,i])

            bp = ax.boxplot(data_to_plot)

            plt.title('PC'+str(i))

            data_to_plot=[]

            plt.show()

            

    def plot_remove_dimensions(self,pca_array,components,remove_dims):

        for dim in remove_dims:

            components=np.delete(components,dim,axis=0)

            pca_array=np.delete(pca_array,dim,axis=1)

        reconstruction=components.T@pca_array.T

        reconstruction=reconstruction.reshape((6,512,512,pca_array.shape[0]))

        gs = gridspec.GridSpec(6, 5)

        fig = plt.figure(figsize=(24,20))

        for sample in range(0,5):

            for channel_dim in range(0,6):

                ax=fig.add_subplot(gs[channel_dim, sample])

                ax.imshow(reconstruction[channel_dim,:,:,sample].reshape(512,512))

                ax.set_yticks([])

                ax.set_xticks([])

        plt.show()

                

proc=ProcessWithIncrementalPCA(data_path,csv_path)

pca_array,components=proc.incremental_PCA('HEPG2',100,100)
#Plot the distribution of PC's according to batch. 

proc.plot_PCs(pca_array,100)
#Plot the full reconstructions

proc.plot_remove_dimensions(pca_array,components,[])
#Plot the reconstruction with the first PC removed

proc.plot_remove_dimensions(pca_array,components,[0])