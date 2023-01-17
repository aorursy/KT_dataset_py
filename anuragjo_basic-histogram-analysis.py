import os

import pandas as pd

import numpy as np

import cv2

import PIL

from tqdm import tqdm

import matplotlib.pyplot as plt

from collections import Counter

import pickle
for folder in ['test','train','val']:

    for dirname,_,filename in os.walk(f'../input/chest-xray-pneumonia/chest_xray/{folder}'):

        if dirname.split('/')[-1]=='NORMAL':

            print(f"{folder} - NORMAL images count: {len(filename)}")

        elif dirname.split('/')[-1]=='PNEUMONIA':

            print(f"{folder} - PNEUMONIA images count: {len(filename)}")      
normal_img_path = '../input/chest-xray-pneumonia/chest_xray/train/NORMAL'

pneumonia_img_path = '../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA'



for color,folder_path in zip(['g','r'],[normal_img_path,pneumonia_img_path]):

    counter = Counter()

    for img_path in tqdm(os.listdir(folder_path)[:1300],total=1300,desc=f"{folder_path.split('/')[-1]} images processed: "):

        image = np.array(cv2.resize(cv2.cvtColor(cv2.imread(folder_path+'/'+img_path), cv2.COLOR_BGR2GRAY),(600,800)))

        counter+=Counter(image.flatten())



    freq_df = pd.DataFrame({'Intensity':list(counter.keys()),'Frequency':list(counter.values())})

    freq_df.sort_values('Intensity',inplace=True)

    freq_df = freq_df[freq_df.Intensity>0]  #discarding frequency count of 0 pixel intensity value

    

    ## Plotting

    plt.figure(figsize=(16,7))

    ax1 = plt.subplot(1,1,1,title=f"Histogram of {folder_path.split('/')[-1]} X-Ray Images (1300 images used)")

    ax1.bar(x=freq_df.Intensity,height=freq_df.Frequency,color=color)

    plt.plot()

    

    ## Saving

    try:

        with open(f"{folder_path.split('/')[-1]}_hist.png",'wb') as f:

            plt.savefig(f)

    except:

        pass

    

    with open(f"{folder_path.split('/')[-1]}_hist_dict.pkl",'wb') as f:

        pickle.dump(counter,f)     

    
with open(f"NORMAL_hist_dict.pkl",'rb') as f:

    counter_normal = pickle.load(f)

with open(f"PNEUMONIA_hist_dict.pkl",'rb') as f:

    counter_pneumonia = pickle.load(f)
freq_normal = pd.DataFrame({'Intensity':list(counter_normal.keys()),'Count':list(counter_normal.values())})

freq_normal.sort_values('Intensity',inplace=True)

freq_pneumonia = pd.DataFrame({'Intensity':list(counter_pneumonia.keys()),'Count':list(counter_pneumonia.values())})

freq_pneumonia.sort_values('Intensity',inplace=True)
freq_normal.head(10)
freq_pneumonia.head(10)
freq_normal['Intensity_bins'] = pd.cut(freq_normal.Intensity,bins=list(range(0,256,15)))

freq_bins_normal = freq_normal.groupby('Intensity_bins',as_index=False).sum()[['Intensity_bins','Count']]



freq_pneumonia['Intensity_bins'] = pd.cut(freq_pneumonia.Intensity,bins=list(range(0,256,15)))

freq_bins_pneumonia = freq_pneumonia.groupby('Intensity_bins',as_index=False).sum()[['Intensity_bins','Count']]
freq_bins_normal
freq_bins_pneumonia
freq_bins_pneumonia.to_csv('bincount_pneumonia.csv',index=False)

freq_bins_normal.to_csv('bincount_normal.csv',index=False)
  