# Haar-cascade XML classifier

!wget -P /kaggle/working/ https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
!ls ../input/deepfake-detection-challenge
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



files =[]

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## Imports

import cv2

import numpy as np 

import math
def lerp(a, b, p):

    '''Linear interpretation in 1D

    Args:

        a (int): point A

        b (int): point B

        c (float): percentage



    Returns:

        int: position

    '''

    return math.floor(a + (b - a) * p)



# TODO: add docs

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):

    '''Resizes Image respecitng aspect ratio (source: https://stackoverflow.com/a/44659589)

    Args:

        

    Returns:

        

    '''

    # initialize the dimensions of the image to be resized and

    # grab the image size

    dim = None

    (h, w) = image.shape[:2]



    # if both the width and height are None, then return the

    # original image

    if width is None and height is None:

        return image



    # check to see if the width is None

    if width is None:

        # calculate the ratio of the height and construct the

        # dimensions

        r = height / float(h)

        dim = (int(w * r), height)



    # otherwise, the height is None

    else:

        # calculate the ratio of the width and construct the

        # dimensions

        r = width / float(w)

        dim = (width, int(h * r))



    # resize the image

    resized = cv2.resize(image, dim, interpolation = inter)



    # return the resized image

    return resized
path_to_video = files[1]

margin = 80

threshold = 10
files[1]
cap = cv2.VideoCapture(path_to_video)

face_cascade = cv2.CascadeClassifier('/kaggle/working/haarcascade_frontalface_default.xml')



first_frame = True

x = -1

while(True):

    # Capture frame-by-frame

    ret, frame = cap.read()



    # Our operations on the frame come here

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    gray_re = image_resize(gray, width=400)

    

    height, width = gray.shape

    height_re, width_re = gray_re.shape



    ratio = height / height_re



    # Detect the faces

    faces = face_cascade.detectMultiScale(gray_re, 1.1, 5)



    # Draw the rectangle around each face

    maxArea = 0

    minDist = np.inf

    

    

    for (_x, _y, _w, _h) in faces:

        # cv2.rectangle(gray, (_x, _y), (x+fw, y+fh), (255, 0, 0), 2)

        if  first_frame and _w*_h > maxArea:

            x, y, w, h = _x, _y, _w, _h

    

        elif (_x - _xp) ** 2 + (_y - _yp) ** 2 < minDist:

            x, y, w, h = _x, _y, _w, _h

            minDist = (_x - _xp) ** 2 + (_y - _yp) ** 2  

            

        maxArea = w*h

        _xp = x

        _yp = y





    #If one or more faces are found, draw a rectangle around the

    #largest face present in the picture

    if maxArea > 0 :

        if minDist < threshold:

            # add margin

            x = max(math.floor(x * ratio - margin), 0)

            y = max(math.floor(y * ratio - margin), 0)

        if first_frame:

            fw = math.floor(w * ratio + 2 * margin)

            fh = math.floor(h * ratio + 2 * margin) 

        

    # get rectangle

    if not first_frame:

        lerp_p = .3

        if minDist < threshold:

            x, y = lerp(xp, x, lerp_p), lerp(yp, y, lerp_p)

        else:

            x, y = xp, yp

        



    if x >= 0:

        patch = gray[y:y+fh, x:x+fw]

        # Convert patch to feature vector



        xp, yp = x, y

    

    # Display the resulting frame

    try:

        print(x, y, w, h)

#         cv2.imshow('frame', patch)

        # cv2.imshow('frame', gray)

    except:

        print('finding face')

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break



    # check if faces were found

    if (x >= 0):

        first_frame = False

    





# When everything done, release the capture

cap.release()

cv2.destroyAllWindows()