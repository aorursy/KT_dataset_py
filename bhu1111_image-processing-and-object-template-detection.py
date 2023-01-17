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
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from PIL import Image
pic = Image.open('/kaggle/input/intro-computer-vision/00-puppy.jpg')
pic
type(pic)
pic_arr = np.asarray(pic)

pic_arr.shape
pic_arr
plt.imshow(pic_arr)
pic_red = pic_arr.copy()
pic_arr[:, :, 0]
#pic_red[:, :, 0] = 0    # Zero out contribution from green

pic_red[:, :, 2] = 0    # Zero out contribution from blue
plt.imshow(pic_red)
plt.imshow(pic_arr[:, :, 0])
import cv2
img = cv2.imread('/kaggle/input/intro-computer-vision/00-puppy.jpg')
img.shape
img_bgr = cv2.imread('/kaggle/input/intro-computer-vision/00-puppy.jpg')

plt.imshow(img_bgr)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
img_gray = cv2.imread('/kaggle/input/intro-computer-vision/00-puppy.jpg',cv2.IMREAD_GRAYSCALE)

plt.imshow(img_gray)
img_gray.shape
img_gray = cv2.imread('/kaggle/input/intro-computer-vision/00-puppy.jpg',cv2.IMREAD_GRAYSCALE)

plt.imshow(img_gray,cmap='gray')
img_gray.shape
## Saving images

cv2.imwrite('my_new_picture.jpg',img_gray)
full = cv2.imread('/kaggle/input/intro-computer-vision/sammy.jpg')

full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)
plt.imshow(full)
face= cv2.imread('/kaggle/input/intro-computer-vision/sammy_face.jpg')

face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

plt.imshow(face)
# The Full Image to Search

full = cv2.imread('/kaggle/input/intro-computer-vision/sammy.jpg')

full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)





# The Template to Match

face= cv2.imread('/kaggle/input/intro-computer-vision/sammy_face.jpg')

face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)





# All the 6 methods for comparison in a list

# Note how we are using strings, later on we'll use the eval() function to convert to function

methods = ['cv2.TM_CCOEFF', 

           'cv2.TM_CCOEFF_NORMED', 

           'cv2.TM_CCORR',

           'cv2.TM_CCORR_NORMED', 

           'cv2.TM_SQDIFF', 

           'cv2.TM_SQDIFF_NORMED']
height, width, channels = face.shape
for m in methods:

    

    # Create a copy of the image

    full_copy = full.copy()

    

    # Get the actual function instead of the string

    method = eval(m)



    # Apply template Matching with the method

    res = cv2.matchTemplate(full_copy,face,method)

    

    # Grab the Max and Min values, plus their locations

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    

    # Set up drawing of Rectangle

    

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum

    # Notice the coloring on the last 2 left hand side images.

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:

        top_left = min_loc    

    else:

        top_left = max_loc

        

    # Assign the Bottom Right of the rectangle

    bottom_right = (top_left[0] + width, top_left[1] + height)



    # Draw the Red Rectangle

    cv2.rectangle(full_copy,top_left, bottom_right, (255,0,0), 10)



    # Plot the Images

    plt.subplot(121)

    plt.imshow(res)

    plt.title('Result of Template Matching')

    

    plt.subplot(122)

    plt.imshow(full_copy)

    plt.title('Detected Point')

    plt.suptitle(m)

    

    

    plt.show()

    print('\n')

    print('\n')
nadia = cv2.imread('/kaggle/input/intro-computer-vision/Nadia_Murad.jpg',0)

denis = cv2.imread('/kaggle/input/intro-computer-vision/Denis_Mukwege.jpg',0)

solvay = cv2.imread('/kaggle/input/intro-computer-vision/solvay_conference.jpg',0)
plt.imshow(nadia,cmap='gray')
plt.imshow(denis,cmap='gray')
plt.imshow(solvay,cmap='gray')
face_cascade = cv2.CascadeClassifier('/kaggle/input/intro-computer-vision/haarcascade_frontalface_default.xml')
def detect_face(img):

    

  

    face_img = img.copy()

  

    face_rects = face_cascade.detectMultiScale(face_img) 

    

    for (x,y,w,h) in face_rects: 

        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 5) 

        

    return face_img
plt.imshow(detect_face(denis),cmap='gray')
plt.imshow(detect_face(nadia),cmap='gray')
# Gets errors!

plt.imshow(detect_face(solvay),cmap='gray')
def adj_detect_face(img):

    

    face_img = img.copy()

  

    face_rects = face_cascade.detectMultiScale(face_img,

                                               scaleFactor=1.2, 

                                               minNeighbors=5) 

    

    for (x,y,w,h) in face_rects: 

        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10) 

        

    return face_img
# Doesn't detect the side face.

plt.imshow(adj_detect_face(solvay),cmap='gray')
eye_cascade = cv2.CascadeClassifier('/kaggle/input/intro-computer-vision/haarcascade_eye.xml')
def detect_eyes(img):

    

    eye_img = img.copy()

  

    eyes = eye_cascade.detectMultiScale(eye_img) 

    

    

    for (x,y,w,h) in eyes: 

        cv2.rectangle(eye_img, (x,y), (x+w,y+h), (255,255,255), 10) 

        

    return eye_img
plt.imshow(detect_eyes(nadia),cmap='gray')