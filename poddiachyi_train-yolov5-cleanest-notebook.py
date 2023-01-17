import numpy as np 
import pandas as pd 
import os
from tqdm.auto import tqdm
import shutil as sh
import numpy as np
import random
import torch
csv_path = '../input/global-wheat-detection/train.csv'
dataset_path = '../input/global-wheat-detection'

IMG_SIZE = 1024
!cp -r ../input/yolov5trainstable/* ./
!cd ../input/yolostuff/ && pip install -r requirements.txt
!pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
def set_seed():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
def load_df(path):
    df = pd.read_csv(csv_path)

    bboxs = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))

    for i, column in enumerate(['x', 'y', 'w', 'h']):
        df[column] = bboxs[:,i]

    df.drop(columns=['bbox'], inplace=True)

    df['x_center'] = df['x'] + df['w'] / 2
    df['y_center'] = df['y'] + df['h'] / 2
    df['classes'] = 0

    df = df[['image_id','x', 'y', 'w', 'h','x_center','y_center','classes']]
    
    return df
df = load_df(csv_path)
df.head()
index = list(set(df.image_id))
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
        
def write_bboxes_to_ann(file, bboxes):
    for j in range(len(bboxes)):
        text = ' '.join(bboxes[j])
        file.write(text)
        file.write("\n")
        
        
def process_bboxes(ann_file, table):
    with open(ann_file_path, 'w+') as f:
        bboxes = table[['classes','x_center','y_center','w','h']].astype(float).values
        bboxes = bboxes / IMG_SIZE
        bboxes = bboxes.astype(str)
        write_bboxes_to_ann(f, bboxes)
        
!rm -r convertor/
val_index = index[0 : len(index)//5]
source = 'train'
for name, table in tqdm(df.groupby('image_id')):
    
    if name in val_index:
        phase = 'val2017/'
    else:
        phase = 'train2017/'
    
    full_labels_path = os.path.join('convertor', phase, 'labels')
    create_folder(full_labels_path)
    
    ann_file_path = os.path.join(full_labels_path, name + '.txt') # annotation file
    process_bboxes(ann_file_path, table)
        
    img_folder = os.path.join('convertor', phase, 'images')
    create_folder(img_folder)
    
    name_with_ext = name + '.jpg'
    img_src = os.path.join(dataset_path, source, name_with_ext)
    img_dst = os.path.join('convertor', phase, 'images', name_with_ext)
    sh.copy(img_src, img_dst)
set_seed()
!python train.py --img 512 --batch 16 --epochs 40 --data ../input/yolostuff/wheat0.yaml --cfg ../input/yolostuff/yolov5x.yaml --name yolov5x_fold0 --weights ../input/yolostuff/yolov5x.pt

create_folder('trained_models')
!cp -r runs/exp0_yolov5x_fold0/weights/ trained_models/
!rm -r runs
!rm -rf convertor
