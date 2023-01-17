import pandas as pd

train = pd.read_csv("../input/diabetic-retinopathy-224x224-gaussian-filtered/train.csv")
from glob import glob

import os
glob('gaussian_filtered_images/Mild/*')
glob('gaussian_filtered_images/Mild/*')
os.listdir('../input/diabetic-retinopathy-224x224-gaussian-filtered/gaussian_filtered_images/gaussian_filtered_images/Mild')
glob('../input/diabetic-retinopathy-224x224-gaussian-filtered/gaussian_filtered_images/gaussian_filtered_images/Mild/*')
glob('../input/diabetic-retinopathy-224x224-gaussian-filtered/gaussian_filtered_images/gaussian_filtered_images/Mild/*[0-9].png')
import numpy as np
# 全ての.pngファイルパスのリストを取得

all_png_list=glob('../input/diabetic-retinopathy-224x224-gaussian-filtered/gaussian_filtered_images/gaussian_filtered_images/*/*')
all_png_list
# フォルダ名のリストを作成

type_list=[p.split('/')[5] for p in all_png_list]
np.unique(type_list)
extention_list=[p.split('.')[-1] for p in all_png_list]
np.unique(extention_list)
df=pd.DataFrame({'filepath':all_png_list,'type':type_list,'extention':extention_list})
df
df['type'].value_counts()
from tqdm import tqdm 
sum=0

for i in tqdm(range(0,1000000)):

    sum+=i

print(sum)
for idx, row in tqdm(df.iterrows()):

    pass
for idx,row in tqdm(df.iterrows(),total=len(df)):

    pass
import os

from pathlib import Path
p = Path('../input/diabetic-retinopathy-224x224-gaussian-filtered/gaussian_filtered_images/gaussian_filtered_images')
p
list(p.iterdir())
list(p.glob('*/*.png'))
list(p.glob('*/*.png'))[0].as_posix()
folder=list(p.iterdir())[3]

print(folder)

print(type(folder))
split_list=os.path.split(folder)

print(split_list)

file_path=os.path.join(folder,'test.txt')

print(file_path)
new_folder_path=os.path.join(p,'new_folder')

print(new_folder_path)
os.path.exists('new_folder')
# if文を条件にしてフォルダを作っていく

if not os.path.exists('new_folder'):

    os.makedirs('new_folder')
os.path.exists('new_folder')
from multiprocessing import Pool,cpu_count
cpu_num=4

p=Pool(processes=cpu_num)
# 物理コア数を取得

cpu_count()
p=Pool(processes=cpu_count()-1)