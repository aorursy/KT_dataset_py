# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2

import numpy as np

from matplotlib import pyplot as plt

image=cv2.imread('../input/natural-images/data/natural_images/cat/cat_0353.jpg')

plt.imshow(image,cmap="gray"),plt.axis("off")

plt.show()
image_grey=cv2.imread('../input/natural-images/data/natural_images/cat/cat_0353.jpg',cv2.IMREAD_GRAYSCALE)

max_output_value=255

neighbourhood_size=99

subtract_from_mean=10

image_binarized=cv2.adaptiveThreshold(image_grey,max_output_value,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,neighbourhood_size,subtract_from_mean)

plt.imshow(image_binarized,cmap="gray"),plt.axis("off")

plt.show()
img_flower=cv2.imread('../input/natural-images/data/natural_images/flower/flower_0449.jpg')

plt.imshow(img_flower,cmap="gray"),plt.axis("off")

plt.show()
median_intensity=np.median(img_flower)

lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity)) 

upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))



image_canny = cv2.Canny(img_flower, lower_threshold, upper_threshold)

plt.imshow(image_canny, cmap="gray"), plt.axis("off") 

plt.show()

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

image_gray = np.float32(image_gray)



block_size = 2 

aperture = 29 

free_parameter = 0.04





detector_responses = cv2.cornerHarris(image_gray,block_size,  aperture, free_parameter)



detector_responses = cv2.dilate(detector_responses, None)

threshold = 0.02 

image[detector_responses > threshold * detector_responses.max()] = [255,255,255]



image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



plt.imshow(image_gray, cmap="gray"), plt.axis("off") 

plt.show()
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



corners_to_detect = 10 

minimum_quality_score = 0.05 

minimum_distance = 25

corners = cv2.goodFeaturesToTrack(image_gray, corners_to_detect,  minimum_quality_score, minimum_distance)



corners = np.float32(corners)



for corner in corners:   

    x, y = corner[0]    

    cv2.circle(image, (x,y), 10, (255,255,255), -1)

   





image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(image_rgb, cmap='gray'), plt.axis("off") 

plt.show()

image_10X10=cv2.resize(image_grey,(10,10))

image_10X10.flatten()
plt.imshow(image_10X10, cmap="gray"), plt.axis("off") 

plt.show()

image_color=cv2.imread('../input/natural-images/data/natural_images/cat/cat_0353.jpg',cv2.IMREAD_COLOR)
image_color_10X10=cv2.resize(image_color,(10,10))

image_color_10X10.flatten().shape