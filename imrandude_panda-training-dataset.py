#Usual Imports

import skimage.io

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import PIL

import cv2

import torch

from pathlib import Path

import pandas as pd

from tqdm import tqdm
ID = '867e8aabc9463fb736371750b3082132'

patch_size = 224

stride=patch_size
def get_tensor_patch(path,path_size=patch_size,stride=patch_size):

    image = skimage.io.MultiImage(path)

    image = cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB)

    input = torch.from_numpy(image)

    input.transpose_(0, 2).shape

    return input.data.unfold(0, 3, 3).unfold(1, patch_size, stride).unfold(2, patch_size, stride).squeeze(0)
# Image file

file_path = f'../input/prostate-cancer-grade-assessment/train_images/'+ID+'.tiff'

patches = get_tensor_patch(file_path)

patches.shape
# Mask file

mask_path = f'../input/prostate-cancer-grade-assessment/train_label_masks/'+ID+'_mask.tiff'

mask_patches = get_tensor_patch(mask_path)

mask_patches.shape
# Save Masks having greater than threshold pixels

threshold = patch_size*patch_size*0.8

final = list()
mask = mask_patches[(2,6)][-1].numpy()
%timeit cv2.imwrite('cv_mask.png', mask) 

%timeit matplotlib.image.imsave('matplot_mask.png', mask)
!rm -rf *.png
%%time

for i in range(mask_patches.size(0)):

    for j in range(mask_patches.size(1)):

        mask = mask_patches[(i,j)][-1].numpy()

        if np.count_nonzero(mask) > threshold:

            output = final.append([i,j])

            cv2.imwrite(ID+'_'+str(i)+'_'+str(j)+'_mask.png', mask) 

            #matplotlib.image.imsave(ID+'_'+str(i)+'_'+str(j)+'_mask.png', mask)
len(final)
#https://joseph-long.com/writing/colorbars/

#https://stackoverflow.com/questions/42396927/how-to-adjust-size-of-two-subplots-one-with-colorbar-and-another-without-in-py

from mpl_toolkits.axes_grid1 import make_axes_locatable

def combine(tensor,aTensor):

    fig, axarr = plt.subplots(ncols=2,figsize=(8, 8))

    im1=axarr[0].imshow(tensor.numpy().transpose(1, 2, 0))

    divider = make_axes_locatable(axarr[0])

    cax1 = divider.append_axes("right", size="5%", pad=0.05)

    cax1.axis('off')

    im2=axarr[1].imshow(aTensor[-1].numpy())

    divider = make_axes_locatable(axarr[1])

    cax2 = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im2, cax=cax2)
i=tuple(final[20])

combine(patches[i],mask_patches[i])
%%time

for i,j in final:

     cv2.imwrite(ID+'_'+str(i)+'_'+str(j)+'.png', patches[(i,j)].numpy().transpose(1, 2, 0))

    #matplotlib.image.imsave(ID+'_'+str(i)+'_'+str(j)+'.png', patches[(i,j)].numpy().transpose(1, 2, 0))
i,j=final[55]

from pylab import imread,subplot,imshow,show

import matplotlib.pyplot as plt



image = imread(ID+'_'+str(i)+'_'+str(j)+'.png')

plt.imshow(image)
!pip -q install imutils
from imutils import paths

import argparse

import cv2



threshold=100.0



def variance_of_laplacian(image):

    return cv2.Laplacian(image, cv2.CV_64F).var()
imagePath=ID+'_'+str(i)+'_'+str(j)+'.png'

image = cv2.imread(imagePath)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

fm = variance_of_laplacian(gray)

text = "Not Blurry"

if fm < threshold:

    text = "Blurry"



plt.title(text)

plt.imshow(image)

fm
!rm -rf *.png
ids = ['867e8aabc9463fb736371750b3082132','14b6e5e738edba8fb2a18cb95dc9af6a','e573d5bd029fefceb5463707eb41a03a',

       '2f307cc31799996529e44c0bcc87b9dd','1329fffac9982cca468efe133cfacf50','3acfefa2f174d9f3f555cae12f5394ee',

       '237f5ce41bb73c120e10024a66602ea7','62274c435774e0ed7a219c9c3b693bce','4fdb930e10a03f26ab85cabf770b5ec6',

       '125ed8b10ae6c064811a76636e949515','f2d3ffcb3aa600272d70fe13b62a56e9','545d14287912a7a53a64fe279a92097e',

       'b22a805186b5c95212db780854e5d2fe','f6df497ddcb8f033d7c28c05fe242aaa','99ce1fa103014b5486f882bb82994acf',

       '623155cecfa8cc671054be004d6cb2cf','4dcd3e9109bd390a86bb0f72cfc80a8f','bb575d53749fe2b8cb9ef5381d968808',

       '2a560b45f1696770fecd76bb8ca75c59','f50e9b75a0d7c2e4ee32bbfb10eb7164','59aa903a1ceb3e9793bedda411a4a2cd',

       'dfae643568b01d1f4ad5d86bac2f7994','25a213edbcc763a0b37c60d060c028a6','fcf954f576b072584d4271144e57f6d3',

       'a5a27468f2a7a299cb8b952abb941705','677998d30fbf0b08e100dc31f2fcc0ee','e6fe8d683d9d7f092c03992a23d901cc',

       'e5c110d0ec8087bc89e6f6a528a690ce','208025030fedd3a2e858890eff150f36','8ef59d7cb53bf92b5c589f7377edb0d7',

       'ee84768fd0b939e9fc7369652cba4a35','8dc0d30245d16902077f9d5da16f1475','08abd36dda089b53088bb7bf7558336a',

       'e9e062c3fcf872abdf5f44dc2c55345a','c35a5c6dd47f3c37f339ff296ef1c6fe','1090bc9626ae4dec99207c3b64a1a050',

       'f86f614f61dce71662f527ac82b3170b','f800a3979b283e4b697fa2aade0c1947','ebd6034772ec8ea179c5a1802c6b886e',

       '0c5c2d16c0f2e399b7be641e7e7f66d9','87ebd3322722faa0adfc6a3efb9e9bbb','a45c077803194bf256f58964f457d5cb',

       '89d67babd435cc76b1366455fd89044f','84d0f286cb639e7ceb3b185c62776100','1d2e39029f1d50a22bd157b3fc4f6b16',

       '44a0e59d540e47bd78f739e49f4b0680','fd545ee9fe42ed0eeb216fd4be3af3d2','1f16f094f5488694af5785d517bda17b',

       '620261483418740878617c4e04015457','30dbd061bed3964c478e64269bb5870e','ee1605f1c4e68d7a52a07fae529e14f8',

       '2e24b2e477e5650b425ae9d58eff5654','26346b4e789c4f7c871f5f0e689b41d5','79c8f162b231aebb6c639afa3f4395ae',

       '83388ee25283c6bdc0cd5ed50b5fb50d','3ed6a37e2e5a9a48ebfe9a8f6e2f1a39','2e2d3ed2069dd704dde97c3d6d194b94',

       '2edf6cc7696c91d86eb86413fa9c82d7','5d438b93171c54575c3248d799952a8c','c7793b26c758ddf3e9d6561c31d394d4',

       'd76c8504812e17930194614c25ba21fd','0e46418f938971abebfda741d6840e9b','3625e6a2deae10c191a3fc3b6fde9fdf',

       '28e51d6764c6d71b78406be1fb53b76b','0b373388b189bee3ef6e320b841264dd','630ead3f5eee9ca3d292d65a785d480b',

       'b790c95b8efb85a4e600533cf0444937','d676924aa4d2c3259e47e91955fe161f','531ce6a7fd3f8699de151e1f07b57970',

       '4aac539cdb139eb4549013b591084c4f','06eda4a6faca84e84a781fee2d5f47e1','da07bf2ac3a36b2ff58e35561f15613f',

       '753743ff6e00c61a67516723909f71d7','93e2ce38e743146a128afb9ff3a61383','264e5e8732450598190acec51da65cdc',

       'bacad594d4d9ad882eda3a6738326d52','b9c08ad54a17440c8901a5d5516a53c6','1ab5733eafcd0a2baede5e06d4195915',

       'e948b70d1e102a039a6b93a4be16b323','b4ec9258133fcb21d9c99fc2ae9d134e','e0c370969267c34db431130a881e72f4',

       '1894dd1e67ddfc62a309e78ef7ffc5e3','9ffb0d108038b5e91cf647c4f4b61318','a3fa13e9535729ec9cb08f17bc8c03cc',

       '3e6e9a534a8d92999185e99dc4293bee']
threshold = patch_size*patch_size*0.95
%%time

for a in tqdm(range(len(ids))):

    ID=ids[a]

    file_path = f'../input/prostate-cancer-grade-assessment/train_images/'+ID+'.tiff'

    patches = get_tensor_patch(file_path)

    mask_path = f'../input/prostate-cancer-grade-assessment/train_label_masks/'+ID+'_mask.tiff'

    mask_patches = get_tensor_patch(mask_path)

    final = list()

    for i in range(mask_patches.size(0)):

        for j in range(mask_patches.size(1)):

            mask = mask_patches[(i,j)][-1].numpy()

            if np.count_nonzero(mask) > threshold:

                output = final.append([i,j])

                cv2.imwrite(ID+'_'+str(i)+'_'+str(j)+'_mask.png', mask)



    for i,j in final:

        cv2.imwrite(ID+'_'+str(i)+'_'+str(j)+'.png', patches[(i,j)].numpy().transpose(1, 2, 0))

    
!mkdir -p masks



import glob

import os

import shutil



for file_name in glob.glob('*mask.png'):

    new_path = os.path.join('masks', file_name)

    shutil.move(file_name, new_path)

!mkdir -p images

for file_name in glob.glob('*.png'):

    new_path = os.path.join('images', file_name)

    shutil.move(file_name, new_path)
# https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/

from pathlib import Path

INPUT_PATH='./images'

files = list(Path(INPUT_PATH).rglob('*.png'))

final_list=list()



for i in tqdm(range(len(files))):

    dict1 = {}

    image = cv2.imread(str(files[i]))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    fm = variance_of_laplacian(gray)

    blur = 0

    if fm < 100.0:

        blur = 1

        #print(str(files[i]),blur,fm)

        out = [str(files[i]),blur,fm]

        final_list.append(out)

    
df = pd.DataFrame(final_list,columns=['filename', 'blur', 'laplace_var'])

df.shape
i=1500

image = imread(df['filename'][i])

plt.title(df['laplace_var'][i])

plt.imshow(image)
for i in tqdm(range(len(df['filename']))):

    Path(df['filename'][i]).unlink()
import re



for i in tqdm(range(len(df['filename']))):

    x = re.sub("images", "masks", df['filename'][i])

    x = re.sub(".png", "_mask.png", x)

    Path(x).unlink()
!tar -czf masks.gz masks  --remove-files

!tar -czf images.gz images --remove-files
