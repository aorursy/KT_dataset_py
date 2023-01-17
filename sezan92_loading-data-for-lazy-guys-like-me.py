import pandas as pd
import matplotlib.pyplot as plt
import PIL
import zipfile
import os
from tqdm import tqdm as tq
import numpy as np
class dataLoader():
    """Class for Loading Data
    Arguments:
        chars: list, a list of strings for showing the dataset zipped file
        mode: training or testing train dataset or test dataset
        extracted: boolean,If already the zip files are extracted or not
        This class expects the zipped files are kept inside 'numta' folder
   
    Usage:
        DL= dataLoader(mode='training',chars=['a'])
        image_gen=DL.img_gen() #Generator
        x,y=image_gen.next() #Batch of images, by default 100 images
        DL.visualise(10) #Plots 10th image from current batch.
    
    Contribution:
        Reader can edit this code according to their need or other dataset
        I urge to fillup the method augment to contribute :)
    
    """
    def __init__(self,mode,chars=None,folder='numta',extracted=False):
        if type(chars) !=list:
            raise ValueError("It must be a list")
        print("DataLoader initiated")
        print(help(dataLoader))
        self._mode=mode
        self._chars=chars
        self._folder=folder
        self._extracted=extracted
    def getchars(self):
        """Getting all chars list"""
        return self._chars
    def setchars(self,char):
        """Adding a char to already existing list
        char:character for dataset"""
        if type(char)==str:
            self._chars.append(char)
            self._extracted=False
        else:
            raise ValueError("It must be a String ")
    def unzip(self):
        """Extracts  all the images from all zipped files
        Extracted files will be saved into the folders of same name"""
        if self._extracted==False:
            
            for filename in tq(self._chars):
                target_folder=os.path.join(self._folder,self._mode+'-'+filename)
                if not os.path.exists(target_folder):
                    self._data = os.path.join(target_folder+'.zip')
                    self._zipped_data = zipfile.ZipFile(self._data)
                    os.makedirs(target_folder)
                    self._zipped_data.extractall(path=target_folder)
                else:
                    print("The %s.zip file already extracted"%(target_folder))
                self._extracted=True
        else:
            print("All zipped files are already extracted!")
    def img_gen(self,img_size=180,img_type='RGB',dims=3,req_num_of_images=100):
        """Generator Method for loading extracted images
        output: 
        X:numpy array of all images
        y:numpy array of all labels"""
        print("checking the folder")
        for filename in tq(self._chars):
            target_folder=os.path.join(self._folder,self._mode+'-'+filename)
            if not os.path.exists(target_folder):
                print("%s doesn't exist! Unziping those files using unzip method"%(target_folder))
                self.unzip()
        all_filenames_list=[]
        for filename in tq(self._chars):
            target_folder=os.path.join(self._folder,self._mode+'-'+filename)
            filenames = pd.read_csv(target_folder+'.csv',index_col=False)
            filenames =filenames[['filename','digit','database name']]
            all_filenames_list.append(filenames)
        all_files_df=pd.concat(all_filenames_list)
        
        X = np.zeros((req_num_of_images,img_size,img_size,dims))
        y = np.zeros(req_num_of_images)
        num=0 #index for X and y
        
        for index,row in all_files_df.iterrows():
            img_name=os.path.join(self._folder,row['database name'],row['filename'])
            img = PIL.Image.open(img_name)
            img = img.resize((img_size, img_size))
            img = np.array(img.convert(img_type), dtype=np.uint8).reshape((img_size,img_size,dims))
            X[num,:,:,:] =img
            y[num]=row['digit']
            num=num+1
            if num==req_num_of_images:
                self.X=X
                self.y=y
                yield X,y
                num=0

    def visualise(self,index):
        """Method for showing any image of current batch"""
        img =self.X[index]
        label = self.y[index]
        plt.imshow(img)
        plt.title(str(label))
    def augment(self):
        """ Method for augmenting data. Did nothing so that others can fill up 
        code according to their creativity :) """
        pass
DL= dataLoader(mode='training',chars=['a'],folder='../input/')
image_gen=DL.img_gen() #Generator
x,y=image_gen.__next__()#Batch of images, by default 100 images
DL.visualise(10) #Plots 10th image from current batch.
x,y=image_gen.__next__()#Batch of images, by default 100 images
DL.visualise(10) #P
