# General packages
import os
import pandas as pd
import numpy as np

import cv2
import openslide
import PIL
from PIL import Image
import zipfile

import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from IPython.display import Image, display
# Location of the training images
path = '../input/prostate-cancer-grade-assessment'

# Location of training labels
train = pd.read_csv(f'{path}/train.csv')#.set_index('image_id')
test = pd.read_csv(f'{path}/test.csv')#.set_index('image_id')
submission = pd.read_csv(f'{path}/sample_submission.csv')

#load suspicious cases ( no mask on images)
suspicious = pd.read_csv(f'../input/suspicious-data-panda/suspicious_test_cases.csv')

# image and mask directories
data_dir = f'{path}/train_images'
mask_dir = f'{path}/train_label_masks'

#remove all suspicious data of training set
df_train= train.copy().set_index('image_id')

for j in df_train.index:
    for i in suspicious['image_id']:
        if i == j:
            df_train.drop([i], axis=0, inplace = True)
            
df_train
#overlay masks on the slide
def overlay_mask_on_slide(slide, mask, center='radboud', alpha=0.8, max_size=(400, 400)):
    """Show a mask overlayed on a slide.
    
    """

    if center not in ['radboud', 'karolinska']:
        raise Exception("Unsupported palette, should be one of [radboud, karolinska].")

    # Load data from the highest level
    slide_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])
    mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])

    # Mask data is present in the R channel
    mask_data = mask_data.split()[0]
    # Create alpha mask
    alpha_int = int(round(255*alpha))
    if center == 'radboud':
        alpha_content = np.less(mask_data.split()[0], 2).astype('uint8') * alpha_int + (255 - alpha_int)
    elif center == 'karolinska':
        alpha_content = np.less(mask_data.split()[0], 1).astype('uint8') * alpha_int + (255 - alpha_int)
        
    alpha_content = PIL.Image.fromarray(alpha_content)
    preview_palette = np.zeros(shape=768, dtype=int)
    
    if center == 'radboud':
        # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}
        preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)
    elif center == 'karolinska':
        # Mapping: {0: background, 1: benign, 2: cancer}
        preview_palette[0:9] = (np.array([0, 0, 0, 0, 1, 0, 1, 0, 0]) * 255).astype(int)
    
    mask_data.putpalette(data=preview_palette.tolist())
    mask_rgb = mask_data.convert(mode='RGB')
    
    overlayed_image = PIL.Image.composite(image1=slide_data, image2=mask_rgb, mask=alpha_content)
    
    return(overlayed_image)
#make tiles on a image
def tile(img, sz=128, N=16):
    shape = np.array(img).shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                 constant_values=255)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    return img
save_dir = "../train_overlay/"
os.makedirs(save_dir, exist_ok=True)

#select the number of images to overlay
for imge in tqdm(df_train.index):
    
    # select the good provider corresponding at the picture
    prov = df_train.loc[imge,'data_provider']
    
    #select the label for the image
    label = df_train.loc[imge, 'isup_grade']
    
    #open slide and mask images
    slide = openslide.OpenSlide(f'../input/prostate-cancer-grade-assessment/train_images/{imge}.tiff')
    mask = openslide.OpenSlide(f'../input/prostate-cancer-grade-assessment/train_label_masks/{imge}_mask.tiff')
    
    #overlay mask on a slide
    im1 = overlay_mask_on_slide(slide, mask, center= prov )
    
    #close slide and mask
    slide.close()
    mask.close()
    
    #create tiles 
    image= tile(im1, sz=128, N=16)
    
    #concatenate each tile on a picture
    image = cv2.hconcat([cv2.vconcat([image[0], image[1], image[2], image[3]]), 
                             cv2.vconcat([image[4], image[5], image[6], image[7]]), 
                             cv2.vconcat([image[8], image[9], image[10], image[11]]), 
                             cv2.vconcat([image[12], image[13], image[14], image[15]])])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    matplotlib.image.imsave( save_dir+f'{imge}.png', image)
    
!ls ../train_overlay/


display(Image(filename='../train_overlay/3046035f348012fdba6f7c53c4faa16e.png'))

!tar -czf tiles_overlay_mask.tar.gz ../train_overlay/*.png