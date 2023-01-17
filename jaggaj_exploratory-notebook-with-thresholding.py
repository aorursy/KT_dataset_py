from skimage.io import imread
from skimage import color, filters, measure
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
%matplotlib inline
#glob all of the training images
img_list = glob.glob("/home/john/Data_Sci/COMP_540/Project/train_images/*/images/*")
print(len(img_list))
#Let's determine how many different image sizes we have, and look at an example of each image size

num_images=len(img_list)
img_shapes = {}
avg_image = {}
sample_image = {}
for i in range(num_images):
 #  img_index = np.random.randint(len(img_list)-1)
    img_index = i
    img_path = img_list[img_index]
    img = imread(img_path)
#    print("Image ", img_index)
    if img.shape not in img_shapes:
        img_shapes[img.shape] = 1
        sample_image[img.shape] = np.copy(img)
#        avg_image[img.shape] = np.copy(img)
    else:
        img_shapes[img.shape] += 1
#        avg_image[img.shape] += np.copy(img)

#We tried to plot the average image for each image size, but got essentially static, due
#to the non-uniformity of the locations of the nuclei
print("There are %d shapes" % len(img_shapes))
for shape, num_images in img_shapes.items():
#    avg_image[shape] =  avg_image[shape]/num_images
    print(shape,num_images)
    print("sample image with shape ", shape)
#    plt.imshow(avg_image[shape])
    plt.imshow(sample_image[shape])

    plt.show()
#glob all of the test images
img_list = glob.glob("/home/john/Data_Sci/COMP_540/Project/test_images/*/images/*")
#Let's determine how many different image sizes we have, and look at an example of each image size

num_images=len(img_list)
img_shapes = {}
sample_image = {}
for i in range(num_images):
    img_index = i
    img_path = img_list[img_index]
    image_id = img_path.split("/")[-1].split('.')[0] #extract image id from path
    img = imread(img_path)

    if img.shape not in img_shapes:
        img_shapes[img.shape] = 1
        sample_image[img.shape] = np.copy(img)

    else:
        img_shapes[img.shape] += 1


print("There are %d shapes" % len(img_shapes))
for shape, num_images in img_shapes.items():
    print(shape,num_images)
    print("sample image with shape ", shape)
    plt.imshow(sample_image[shape])
    plt.show()
def rle_encoding(dots): #this function modified from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return [str(i) for i in run_lengths]
#Now let's use basic otsu thresholding to identify nuclei in the images

img_list = sorted(glob.glob("/home/john/Data_Sci/COMP_540/Project/test_images/*/images/*"))

num_images = len(img_list)
data = pd.DataFrame(columns=["ImageId", "EncodedPixels"])
for i in range(num_images):
    img_index = i
    img_path = img_list[img_index]
    image_id = img_path.split("/")[-1].split('.')[0] #extract image id from path
    img = color.rgb2gray(imread(img_path)) #read as black and white for simple thresholding segmentation
    val = filters.threshold_otsu(img)
    nuc_mask = img > val
    nuc_labels = measure.label(nuc_mask,background=0) #label individual objects (nuclei)
    #Get RLE encoding for EACH nucleus in a given image
    for label in np.unique(nuc_labels)[1:]:
        nuc_pix = np.where(nuc_labels.T.flatten() == label)[0]  #np.where returns a tuple, 
       #so we have to take the first element, which is the array we want
 #       rle = rle_from_mask(nuc_pix)
        rle = rle_encoding(nuc_pix)
        #only keep rle's that are clearly not noise, e.g. more than a minimum number of pixels
        min_pix = 20 #Perhaps we could cross-validate the min_pix value on the training set.
        num_pix = sum([float(num) for num in rle[1::2]])
        if num_pix > min_pix:
            data = data.append({"ImageId":image_id, "EncodedPixels": " ".join(rle)}, ignore_index=True)
data.to_csv("submission.csv", index=False)


