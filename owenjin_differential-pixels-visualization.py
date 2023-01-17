import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
base_dir = '/kaggle/input/alaska2-image-steganalysis/'
img_dirs = ['Cover','JUNIWARD', 'JMiPOD',  'UERD']
def check_img(img):
    img_c = mpimg.imread(os.path.join(base_dir,"Cover",img))
    img_u = mpimg.imread(os.path.join(base_dir,"UERD",img))
    img_j = mpimg.imread(os.path.join(base_dir,"JUNIWARD",img))
    img_jm = mpimg.imread(os.path.join(base_dir,"JMiPOD",img))
    
    img_c_u = (img_c-img_u)*10000
    img_c_j = (img_c-img_j)*10000
    img_c_jm = (img_c-img_jm)*10000
    
    return [img_c, img_c_j,img_c_jm,img_c_u]


all_files = {}
for id in img_dirs:
    if id=="Test":
        continue
    lst = []
    for file in os.listdir(os.path.join(base_dir, id)):
        lst.append(file)
    all_files[id]=lst

files_df = pd.DataFrame(all_files, columns=all_files.keys())
files_df.head(5)
n_rows=4
n_cols=4
ax_n = 0

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, 16))
image_list = []
image_idx = np.random.randint(files_df.shape[0],size=4)
  

for f in files_df.iloc[image_idx,0]:

    image_list.extend(check_img(f))
        

for fname in (image_list) :
    ax_n += 1
    axes[(ax_n-1) // n_cols, (ax_n-1) % n_rows ].imshow(fname)
#     ax.imshow(img)
    axes[(ax_n-1) // n_cols, (ax_n-1) % n_rows ].set_title(img_dirs[(ax_n-1) % n_rows])