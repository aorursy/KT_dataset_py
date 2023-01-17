import PIL

import pandas as pd

import numpy as np

import os

import cv2

import glob

import scipy

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import skimage.io

from skimage.color import rgb2gray

from skimage.transform import rescale, resize
input_path = "../input/prostate-cancer-grade-assessment"
train_images_path = os.path.join(input_path,"train_images")

images_im = glob.glob(os.path.join(train_images_path, "*tiff"))

images_im.sort()



train_images_path = os.path.join(input_path,"train_label_masks")

images_mask = glob.glob(os.path.join(train_images_path, "*tiff"))

images_mask.sort()



image_label = pd.read_csv(input_path + "/train.csv")["image_id"]



train_size = len(images_im)

mask_size = len(images_mask)

print(train_size, mask_size)
save_extra_images = [None]*(train_size-mask_size)
t1=str.split(str.split(images_im[200],"/")[-1],".")[0]

t2=str.split(str.split(images_mask[200],"/")[-1],"_")[0]



print(t1)

print(t2)
images_im_2nd = images_im





print("Mask count :"+ str(len(images_mask)))

print("Image before removal :"+ str(len(images_im)))

j=0

for i in range(len(images_mask)):

    t1=str.split(str.split(images_im_2nd[i],"/")[-1],".")[0]

    t2=str.split(str.split(images_mask[i],"/")[-1],"_")[0]

    #print(i)

    if t1!=t2: 

        save_extra_images[j] = images_im_2nd[i]

        images_im_2nd.remove(images_im_2nd[i])

        i=i-1

        j+=1

print("Image after removal :"+ str(len(images_im_2nd)))

print("Images without masks : "+str(train_size-mask_size)+ ".. 5 are listed below :")

print(save_extra_images[:5])
t1=str.split(str.split(images_im_2nd[200],"/")[-1],".")[0]

t2=str.split(str.split(images_mask[200],"/")[-1],"_")[0]



print(t1)

print(t2)
a= skimage.io.MultiImage(images_mask[11])



a=rgb2gray(a[2])

plt.imshow(a)

for i in range(1,25):

    b= skimage.io.MultiImage(images_mask[i])

    b= rgb2gray(b[2])    

    a= np.unique(a)

    b= np.unique(b)

    #print(a, b)

    c= np.concatenate((a,b),axis=0)

    #print(c)

    c= np.sort(np.unique(c))

    a= c

print(c)

    #print("***************")

def AND_pic(instance, c=c, images_im_2nd=images_im_2nd, images_mask=images_mask, plot_it=True):

    sample= skimage.io.MultiImage(images_im_2nd[instance])

    sample=sample[2]

    sample2= skimage.io.MultiImage(images_mask[instance])

    sample2=rgb2gray(sample2[2])

    

    c1= (c[1]+c[0])/2

    binary_mask= (sample2>c1)*1

    

    #print(sample.shape)

    #print(sample2.shape)

    #print(binary_mask.shape)

    binary_mask_3d = np.zeros((sample.shape[0],sample.shape[1],sample.shape[2]))

    for i in range(3):

        binary_mask_3d[:,:,i]= binary_mask



    sample_masked = np.multiply(sample,binary_mask_3d).astype(int)

    if plot_it:

        plt.imshow(sample_masked)

        plt.show()

        plt.imshow(sample)

    return sample_masked, binary_mask
def b_box_gen(binary_array):

    x_sum = np.sum(binary_array,axis=0)

    x1 = np.argmax((x_sum >0.5)*1)

    noise= np.array(range(len(x_sum))) * 1e-15

    x2 = np.argmax(((x_sum >0.5)*1)+noise)

    y_sum = np.sum(binary_array,axis=1)

    y1 = np.argmax((y_sum >0.5)*1)

    noise= np.array(range(len(y_sum))) * 1e-15

    y2 = np.argmax(((y_sum >0.5)*1)+noise)

    """

    add_x = (x2-x1)*0.001//1

    add_y = (y2-y1)*0.001//1

    x1-= add_x

    x2+= add_x

    y1-= add_y

    y2+= add_y

    """

    return x1, y1, x2, y2

sample_masked, binary_mask= AND_pic(5)
def square_image(img, target, plot_it= True):

     

    if len(img.shape)%2:

        img= scipy.ndimage.zoom(img, ((target)/img.shape[0],(target)/img.shape[1], 1))

    else:

        factor= 0.08

        enlarge = target/factor

        img = resize(img, (enlarge,enlarge), anti_aliasing=True)

        img = rescale(img, factor,  anti_aliasing=False)

        img = (img>((np.min(img)+np.max(img))/2))*1

    if plot_it:

        plt.imshow(img)

        plt.show()

    

    return img
target = 250

binary_mask_2 = square_image(binary_mask, target)

sample_masked_2 = square_image(sample_masked, target)
x1, y1, x2, y2 = b_box_gen(binary_mask_2)



fig,ax = plt.subplots(1)

ax.imshow(sample_masked_2)

rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,edgecolor='r',facecolor='none')

ax.add_patch(rect)



plt.show()
def final_box_gen(instance, target, plot_it= False):

    sample, binary= AND_pic(instance,  plot_it = plot_it)

    binary_2 = square_image(binary, target, plot_it = plot_it)

    sample_2 = square_image(sample, target, plot_it = plot_it)

    x1, y1, x2, y2 = b_box_gen(binary_2)

    box = np.zeros((4,1))

    box[:,0] = [x1, y1, x2, y2]

    if plot_it:

        fig,ax = plt.subplots(1)

        ax.imshow(sample_2)

        rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,edgecolor='r',facecolor='none')

        ax.add_patch(rect)

        plt.show()

    

    return sample_2, box
target_image_size = 300

example, box = final_box_gen(0, target_image_size, False)



#print(example.shape)

#print(box.shape)

#print(len(images_im_2nd))



total_x_data = np.zeros((len(images_im_2nd), target_image_size, target_image_size, 3 ), dtype=int)

total_y_data = np.zeros((len(images_im_2nd), 4, 1))





for i in range(500):          #use range(len(images_im_2nd)) for full dataset

    if i==774 or i==5211:        

        continue 

                               # these data samples have mismatched image and mask shape...

                               # you may fix it by reshaping them...I skipped it

    total_x_data[i,:,:,:], total_y_data[i,:,:] = final_box_gen(i, target_image_size, False)

    #total_x_data[i,:,:,:] = total_x_data[i,:,:,:].astype(int)

    print("Entry : "+str(i)+" : "+str.split(str.split(images_im_2nd[i],"/")[-1],".")[0])

save_dir = "/kaggle/a_images/"

os.makedirs(save_dir, exist_ok=True)