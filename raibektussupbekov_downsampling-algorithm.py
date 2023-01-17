# So I created this function below to show how the downsampling algorithm works



import numpy as np

import matplotlib.pyplot as plt



def downsample_image(img:np.array, ratio: float) -> np.array:

    

    """

    Downsamples or scales down an image.

    

    Keyword arguments:

    img -- the image data, 

           Numpy ndarray has shape either

           * (H, W) for grayscale images

           or

           * (H, W, 3) for RGB images

           or

           * (H, W, 4) for RGBA images

    ratio -- the extent to scale down the image

    

    Return:

    the downsampled copy of the image data,

    Numpy ndarray has the same shape and dtype as the input

    """

    

    height, width = img.shape[:2]



    height_new = height // ratio

    width_new = width // ratio

    

    height_scale = height / height_new

    width_scale = width / width_new

    

    # RGB(A)

    if len(img.shape) > 2:

        img_new = np.zeros((height_new, width_new, img.shape[2]), dtype=img.dtype)

        for channel in range(img.shape[2]):

            for x_new in range(height_new):

                for y_new in range(width_new):

                    img_new[x_new, y_new, channel] = img[round(x_new*height_scale), round(y_new*width_scale), channel]

    # Grayscale

    else:

        img_new = np.zeros((height_new, width_new), dtype=img.dtype)

        for x_new in range(height_new):

            for y_new in range(width_new):

                img_new[x_new, y_new] = img[round(x_new*height_scale), round(y_new*width_scale)]

                

    return img_new

    

    
 !wget -O "orig_example_1.jpg" "https://cdn.getyourguide.com/img/tour_img-2179467-148.jpg" 
img_orig = plt.imread("orig_example_1.jpg")

img_downsampled = downsample_image(img_orig, 3)



fig, axes = plt.subplots(1, 2, figsize=(15, 30))

axes[0].set_title("Original-1")

axes[0].imshow(img_orig)

axes[1].set_title("Downsampled-1")

axes[1].imshow(img_downsampled, interpolation="nearest")
!wget  -O "orig_example_2.jpg" "https://media.cntraveler.com/photos/58f984542867946a9cbe1f11/master/pass/praca-do-comercio-lisbon-GettyImages-648812458.jpg"   
img_orig = plt.imread("orig_example_2.jpg")

img_downsampled = downsample_image(img_orig, 5)



fig, axes = plt.subplots(1, 2, figsize=(30, 50))

axes[0].set_title("Original-2")

axes[0].imshow(img_orig)

axes[1].set_title("Downsampled-2")

axes[1].imshow(img_downsampled, interpolation="nearest")