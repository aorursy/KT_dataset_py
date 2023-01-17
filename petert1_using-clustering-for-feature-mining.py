import numpy as np

import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt

import scipy.misc as sc



%matplotlib inline
pngfile = np.array(Image.open("train/Black-grass/0ace21089.png")) # load single image and convert into nparray

plt.imshow(pngfile)

h=pngfile.shape[0] # get height

w=pngfile.shape[1] # get width

pngfile=pngfile.reshape([h*w,3])
np.random.seed(1) # To be sure you will get the same clusters



from sklearn.cluster import KMeans



kmeans = KMeans(init='k-means++', n_clusters=20, n_init=1)

kmeans.fit(pngfile) # Clustering



pred=kmeans.predict(pngfile[:,:]) # Extracting pixel's cluster numbers

pred=pred.reshape([h,w]) # reshaping the pixels for visualizetion

plt.imshow(pred) # show the result
res=(pred==12)*1 # if pixel number is 12, the result will be 1, otherwise - 0.

plt.imshow(res)
pngfile = Image.open("train/Maize/a1d7080b1.png")

pngfile=np.array(pngfile)

h=pngfile.shape[0]

w=pngfile.shape[1]

pngfile=pngfile.reshape([h*w,3])

pred=kmeans.predict(pngfile[:,:])

pred=(pred.reshape([h,w])==12)*1

plt.imshow(pred)
import glob

from tqdm import tqdm_notebook as tqdm



folder_list=[]

for filename in glob.iglob('train/**', recursive=False):

    c=filename.split('\\')

    folder_list.append(c[len(c)-1])

    

f=open("train.csv",'w') 



with tqdm(total=len(folder_list)) as pbar:    

    for folder in folder_list:

        print(folder)

        pbar.update(1)

        for filename in glob.iglob('train/' + folder + '/*.png', recursive=False):

            pngfile = np.array(Image.open(filename))

            h=pngfile.shape[0]

            w=pngfile.shape[1]

            

            if(pngfile.shape[2]!=3):

                continue

                

            pngfile=pngfile.reshape([h*w,3])

            pred=kmeans.predict(pngfile[:,:])

            pred=pred.reshape([h,w])



            res=(pred==12)*1          

            res=sc.imresize(res,[100,100]).reshape([10000])



            f.write(folder+'\t')

            for i in range(9999):

                f.write(str(res[i])+'\t')  

            f.write(str(res[9999])+'\n')

            f.write(folder+'\t')

            for i in range(9999):

                f.write(str(res2[i])+'\t')  

            f.write(str(res2[9999])+'\n')

f.close()
f=open("test.csv",'w') 



for filename in glob.iglob('test/*.png', recursive=False):

    pbar.update(1)

    pngfile = np.array(Image.open(filename))

    h=pngfile.shape[0]

    w=pngfile.shape[1]

    pngfile=pngfile.reshape([h*w,3])

    pred=kmeans.predict(pngfile[:,:])

    pred=pred.reshape([h,w])



    res=(pred==12)*1

                        

    res=sc.imresize(res,[100,100]).reshape([10000])



    for i in range(9999):

        f.write(str(res[i])+'\t')  

    f.write(str(res[9999])+'\n')

    

f.close()