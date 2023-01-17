import pandas as pd 
import numpy as np 
import os  
from tqdm import tqdm
import pickle
path = '../input/widerface/PREPROCESSED_DATA.pickle'
dict_data =  pickle.load(open(path, 'rb'))
data = pd.DataFrame.from_dict(dict_data, orient='index').reset_index()
data.rename(columns={'index':'image_path'}, inplace=True)
data.head()
#JO BHI PATH HAI
#SARI IMAGES EK HI FOLDERS ME RAKHNI HAI ALG ALG SUBFOLDERS ME NHI
#IMAGES IS FOLDER ME "YOLOV5/images/train/{SARI IMAGES}"

path = 'YOLOV5/labels/train/'

#TRAIN ME AISA BNNA CAHIYE -> 'YOLOV5/labels/train/0_Parade_marchingband_1_849.txt'

for _, row in tqdm(data.iterrows(), total=len(x)):
    image_name = row['image_path']
    bboxes = row['bbx']
    image_height = float(row['image_height'])
    image_width = float(row['image_width'])
    full_path = f'{path}{image_name.txt}'
    f = open(full_path, 'w')
    for bbox in bboxes:
        x = float(bbox[0])
        y = float(bbox[1])
        w = float(bbox[2])
        h = float(bbox[3])
        
        x_center = x + w/2
        y_center = y + h/2
        
        x_center /= image_width
        y_center /= image_height
        w /= image_width
        h /= image_height
        
        #(class, x_center, y_center, width, height)
        file_sting_write = f'0, {x_center}, {y_center}, {w}, {h}\n'
        f.write(file_string_write)
    f.close()
