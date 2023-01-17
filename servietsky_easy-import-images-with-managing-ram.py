import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import tqdm

from PIL import Image

import glob

import skimage.measure

import gc
# nbr_images : the quantity of images that we want to import



# df : - If a DataFrame that contains images exists and we want to concatenate 

#        with the images that this function is going to import, we add it. 

#        the return of the function is the existing dataframe concatenated with the new.

#      - If NaN the function creates a new DataFrame which contains the imported images.



# file_name : The name of the file into which we want to import these images. Cover, JMiPOD, JUNIWARD or UERD.



# from_ : From which image you can start importing. If 1000 entered, the function will start importing from the thousandth image.



# status : It is a gadget, it takes 'neg' or 'pos', if 'neg' it adds an output column at the end of the DataFrame equal to 0,

#          if pos it adds an output column equal to 1. it is to distinguish whether the image hides a message or not.







def img_reader(nbr_images = 10, df = None, file_name = 'Cover', from_ = 0, status = 'neg') :

    from_ = from_

    nbr_images  = nbr_images

    image_list = []

    i=0

    j=0

    df = df

    file_name = file_name

    for filename in tqdm.tqdm(glob.glob('../input/alaska2-image-steganalysis/'+file_name+'/*.jpg')): 

        if j >= from_ :

            im=mpimg.imread(filename)

            im=skimage.measure.block_reduce(im, (2,2,1), np.max) # Drop this step to not apply the image pooling.

            image_list.append(np.sum(im.reshape((d3, d1*d2)), axis = 0).tolist()) # d3 = 3, d1 and d2 = 256, without pooling this may be 3 * 512 * 512, np.sum() is for sum rgb. 

            i+=1

            if i%1000 == 0 : # is for concat DataFrame by batch of 100 images.

                if df is None:

                    df = pd.DataFrame(image_list).astype('int16')

                    del image_list

                    gc.collect()

                    image_list = []

                else :

                    df = pd.concat([df , pd.DataFrame(image_list).astype('int16')])

                    del image_list

                    gc.collect()

                    image_list = []

                    if i == nbr_images :    

                        del image_list

                        gc.collect()

                        break

        j=j+1

        

    if status == 'neg' :

        df['output'] = 0

        df['output'] = df['output'].astype('int16')

        gc.collect()

    else :

        df['output'] = 1

        df['output'] = df['output'].astype('int16')

        gc.collect()

        

    return df
# Here i recover an image and I apply the pooling on it in order to recover the final dimensions d1, d2 and d3.



img=mpimg.imread('../input/alaska2-image-steganalysis/Cover/00001.jpg')

test_pool = skimage.measure.block_reduce(img, (2,2,1), np.max)

d1, d2, d3 = test_pool.shape

del test_pool

gc.collect()
# I call this function 4 times:



# the first to receive 12,000 images from image 1 of the folder Cover, label them 0 and store them in df_neg.



df_neg = img_reader(nbr_images = 12000, df = None, file_name = 'Cover', from_ = 0, status = 'neg')

print('import Cover Done !')



# Thereafter i recover 4,000 images from JMiPOD starting with image number 1, then 4,000 from JUNIWARD starting from image number 4,000 and 4,000 from UERD starting from image number 8000.



# At the end i have a dataframe 'df_pos' with 4000 images of JMiPOD followed by 4000 images of JUNIWARD followed by 4000 images of UERD which makes 12000 images in total labeled 1.



df_pos = img_reader(nbr_images = 4000, df = None, file_name = 'JMiPOD', from_ = 0, status = 'pos')

print('JMiPOD Done!')

df_pos = img_reader(nbr_images = 4000, df = df_pos, file_name = 'JUNIWARD', from_ = 4000, status = 'pos')

print('JUNIWARD Done!')

df_pos = img_reader(nbr_images = 4000, df = df_pos, file_name = 'UERD', from_ = 8000, status = 'pos')

print('UERD Done!')

print('df_neg info :')

display(df_neg.info())



print('df_pos info :')

display(df_pos.info())
print('df_neg head :')

display(df_neg.head())



print('df_pos head :')

display(df_pos.head())
# Lets save them in pkl format for next.



df_neg.to_pickle('df_neg.pkl')

df_pos.to_pickle('df_pos.pkl')
# concatenate all, free up space.



df_train = pd.concat([df_pos, df_neg], ignore_index = True).astype('int16')

del df_pos, df_neg

gc.collect()