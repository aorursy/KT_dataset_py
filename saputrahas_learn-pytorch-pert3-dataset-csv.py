#library kebutuhan
%matplotlib inline
import torch
import torchvision
import torchvision.transforms as transforms
import PIL
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import pandas as pd
import seaborn as sns
#melakukan pengecekan directory pada dataset
!ls -all /kaggle/input/
#pemanggilan data dari directory
partfile = Path('/kaggle/input/Iris.csv')

#code library pandas untuk membaca data csv
df = pd.read_csv(p)

#melakukan drop/hapus pada kolom Id
df = df.drop(columns='Id')

#???
df['Species'] = df['Species'].astype('category').cat.codes
feature = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
target = df[['Species']]
feature
class IrisDataset(torch.utils.data.Dataset):
    
    #class IrisDataset untuk membuat parameter yang akan digunakan nantinya yang dideklarasikan lewat parameter "__init__"
    def __init__(self, path, feature_columns, target_columns, transform=None):
        self.path = Path(path)
        self.dframe = pd.read_csv(self.path)
        self._do_normalizer()
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.transform = transform
    
    #???
    def _do_normalizer(self):
        self.dframe = self.dframe.drop(columns='Id')
        self.dframe['Species'] = self.dframe['Species'].astype('category').cat.codes
    
    # fungsi untuk menghitung panjang dari data yang ada
    def __len__(self):
        return len(self.dframe)
    
    #melakukan load data dengan mengambil data peritemnya 
    def __getitem__(self, idx):
        #??
        feature = self.dframe[self.feature_columns].iloc[idx].values
        #??
        target = self.dframe[self.target_columns].iloc[idx].values
        
        #melakukan transform jika ada dan "self.transform berasal dari parameter yang sudah dideklarasikan pada fungsi __init__"
        if self.transform:
            feature = self.transform(feature)
            target = self.transform(target)
        
        return feature, target

#melakukan convert dari numpy ke tensor
class NumpyToTensor(object):
    def __call__(self, param):
        return torch.from_numpy(param.astype(np.float32))
#melakukan pemanggilan data
p = '/kaggle/input/Iris.csv'
fc = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
tc = ['Species']

#melakukan transform "merubah numpy menjadi tensor"
tmft = transforms.Compose([NumpyToTensor()])

#iris merupak variabel untuk memanggil kembali dataset yang telah dibuat sebelumnya dengan nama IrisDataset
#path merupakan variabel untuk menampung p yaitu p adalah directory file 
#target_colmns ??
#transform merupakan variabel atau parameter yang telah kita buat sebelumnya di fungsi __init__ karena ----> lanjutan kebawah
#parameter transform sebelumnya kita isi None maka disinilah baru kegunaaan atau pengisian transform yaitu dengan tmft
#tmft sendiri juga merupakan variabel yang menampung sebuah fungsi atau kegunaan untuk melakukan convert terhadap 
#numpy menjadi tensor
iris = IrisDataset(path=p,feature_columns=fc, target_columns=tc, transform=tmft )

#loader merupakan variabel untuk menampung Dataloader
#iris merupakan varibel yang sudah dideklarasikan sebelumnya
#batch_size merupakan jumlah banyak data yang akan diload satu kali pengeloadtan
#shuffle kegunaannya untuk melakukan acak pada data apabilan bernilai True sedangkan kalau False maka data akan terurut dari index 0
#num_workers??
loader = torch.utils.data.DataLoader(iris, batch_size=16, shuffle=True, num_workers=0)
#??
load_iter = iter(loader)
x,y = load_iter.next()
x
# class NumpyToTensor(object):
#     def __call__(self, param):
#         return torch.from_numpy(param.astype(np.float32))
    
# class TensorToNumpy(object):
#     def __call__(self, param):
#         return param.numpy()
    
# a = np.array([1,5,6])
# tmft = transforms.Compose([
#     NumpyToTensor(),
#     TensorToNumpy()
# ])
# x = tmft(a)

# ntt = NumpyToTensor(),
# ttn = TensorToNumpy()
# x = ntt(a)
# x = ttn(x)
# x
