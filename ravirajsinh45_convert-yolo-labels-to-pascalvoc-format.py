import numpy as np

import cv2

import matplotlib.pyplot as plt

import pandas as pd

import os

from tqdm import tqdm
path = '/kaggle/input/crop-and-weed-detection-data-with-bounding-boxes/agri_data/data/'
data = [os.path.join(path,x) for x in os.listdir(path)]
images = [data[x] for x in range(len(data)) if data[x].endswith('.jpeg')]
labels = [data[x] for x in range(len(data)) if data[x].endswith('.txt')]
#checking our conversion is working or not

plt.imshow(cv2.cvtColor(cv2.imread(images[0]),cv2.COLOR_BGR2RGB))
df = pd.DataFrame(columns=['image_name','object','x_cen','y_cen','w','h'])

i=0

for id in tqdm(images):

    label = open(id[:-5]+'.txt','r') 

    a = label.read()

    label.close()

    label = a.split('\n')

    cleaned_label = label[:-1]

    

    for name in cleaned_label:

        ind_label = name.split(' ')

        df.loc[i,'image_name'] = id[78:]

        df.loc[i,'object'] = int(ind_label[0])

        df.loc[i,'x_cen'] = float(ind_label[1])

        df.loc[i,'y_cen'] = float(ind_label[2])

        df.loc[i,'w'] = float(ind_label[3])

        df.loc[i,'h'] = float(ind_label[4])

        i += 1

    
df.head()
#column name for pascal-voc dataframe

column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
pascal_voc = pd.DataFrame(columns=column_name)

for i in tqdm(range(len(df))):

    

    pascal_voc.loc[i,'filename'] = df.loc[i,'image_name']

    pascal_voc.loc[i,'width'] = 512

    pascal_voc.loc[i,'height']= 512  

    if df.loc[i,'object'] == 0:    

        pascal_voc.loc[i,'class'] = 'crop'

    else:

        pascal_voc.loc[i,'class'] = 'weed'

    pascal_voc.loc[i,'xmin']  = int((df.loc[i,'x_cen'] - df.loc[i,'w']/2)*512)

    pascal_voc.loc[i,'ymin']  = int((df.loc[i,'y_cen'] - df.loc[i,'h']/2)*512)

    pascal_voc.loc[i,'xmax']  = int((df.loc[i,'x_cen'] + df.loc[i,'w']/2)*512)

    pascal_voc.loc[i,'ymax']  = int((df.loc[i,'y_cen'] + df.loc[i,'h']/2)*512) 

    
pascal_voc
any_index_number = 55 #change this number for different images

test_img = cv2.cvtColor(cv2.imread(path+pascal_voc.loc[any_index_number,'filename']),cv2.COLOR_BGR2RGB)

test_df = pascal_voc[pascal_voc['filename']==pascal_voc.loc[any_index_number,'filename']].reset_index(drop=True)

for i in range(len(test_df)):

    

    rec = cv2.rectangle(test_img, (test_df.loc[i,'xmin'],test_df.loc[i,'ymin']), (test_df.loc[i,'xmax'],test_df.loc[i,'ymax']), (255,0,0), 1, 1) 

    text = cv2.putText(rec, test_df.loc[i,'class'], (test_df.loc[i,'xmin'],test_df.loc[i,'ymin']-10), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,0), 2, cv2.LINE_AA)

    

plt.figure(figsize=(8,8))    

plt.imshow(text)

plt.axis('off')

plt.show()


pascal_voc.to_csv('pascal_voc_format.csv',index=False)