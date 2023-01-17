# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import PIL
import matplotlib.pyplot as plt
import glob

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pics=[]
ims=[]
desc_stat=pd.DataFrame()
fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(15, 15), sharex=True, sharey=True)
for i, f in enumerate(glob.glob("/kaggle/input/cee-498-project8-pore-in-concrete/batch1/batch1/*.png")[0:100]):
    ax = axes[int(i/10)][int(i%10)]
    im = PIL.Image.open(f).convert("LA")
    ax.imshow(im)
    im_to_num = np.array(im)
    pix=im_to_num[:,:,0].reshape(im.size[0],im.size[1]).ravel()
    pics.append(pix)
    ims.append(im)
    df=pd.DataFrame(pix)
    df_stat=df.describe().T
    Porosity=df[df==0].count()/pix.shape[0]*100
    df_stat['Porosity']=Porosity
    df_stat['Range']=df_stat['max']-df_stat['min']
    df_stat['mode']=df.mode()
    desc_stat=pd.concat([desc_stat,df_stat],axis=0 )
    desc_stat = desc_stat.drop(columns=['25%', 'min', '75%', 'count'])    
desc_stat=desc_stat.reset_index()   
desc_stat = desc_stat.drop(columns=['index']) 

desc_stat.head()
desc_stat['mean'].hist(figsize=(10, 4), bins=20)
desc_stat['mean'].plot(kind='density', subplots=True, layout=(1, 2), 
                  sharex=False, figsize=(10, 4));
desc_stat['Porosity'].hist(figsize=(10, 4), bins=20)
desc_stat['Porosity'].mean()
pics2=[]
ims2=[]
desc_stat2=pd.DataFrame()
fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(15, 15), sharex=True, sharey=True)
for m, n in enumerate(glob.glob("/kaggle/input/cee-498-project8-pore-in-concrete/batch2/batch2/*.png")[0:100]):
    ax2 = axes[int(m/10)][int(m%10)]
    im2 = PIL.Image.open(n).convert("LA")
    ax2.imshow(im2)
    im_to_num2 = np.array(im2)
    pix2=im_to_num2[:,:,0].reshape(im2.size[0],im2.size[1]).ravel()
    pics2.append(pix2)
    ims2.append(im2)
    df2=pd.DataFrame(pix2)
    df_stat2=df2.describe().T
    Porosity2=df2[df2==0].count()/pix2.shape[0]*100
    df_stat2['Porosity']=Porosity2
    df_stat2['Range']=df_stat2['max']-df_stat2['min']
    df_stat2['mode']=df2.mode()
    desc_stat2=pd.concat([desc_stat2,df_stat2],axis=0 )
    desc_stat2 = desc_stat2.drop(columns=['25%', 'min', '75%', 'count'])    
desc_stat2=desc_stat2.reset_index()   
desc_stat2 = desc_stat2.drop(columns=['index'])

desc_stat2.head()
desc_stat2['mean'].hist(figsize=(10, 4), bins=20)
desc_stat2['mean'].plot(kind='density', subplots=True, layout=(1, 2), 
                  sharex=False, figsize=(10, 4));
desc_stat2['Porosity'].hist(figsize=(10, 4), bins=20)
desc_stat2['Porosity'].mean()