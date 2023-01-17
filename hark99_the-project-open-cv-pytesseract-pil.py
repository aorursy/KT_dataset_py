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

        print (os.path.join(dirname, filename))

        

        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import zipfile

import PIL

from PIL import Image,ImageDraw

from IPython.display import display

import pytesseract

import cv2 as cv

import numpy as np



# loading the face detection classifier

face_cascade = cv.CascadeClassifier('/kaggle/input/eye-face-classifier/utf-8haarcascade_frontalface_default.xml')
large_images = []

for images in range (0,13):

    large_images.append('/kaggle/input/news-large-images/a-{}.png'.format(images))

    #print ('starting from image index {}'.format(images))            
# Functions for detecting faces and msking contact sheet



thumb_size = (128, 128)





def show_faces(fname):

    cv_image = cv.imread(fname)

    pil_img = Image.open(fname)

    cv_image_gray = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)

    

    faces = face_cascade.detectMultiScale(cv_image_gray, 1.35) #last scalefactor 1.35

    

    face_list = []

    for x,y,w,h in faces:

        cropped = pil_img.crop((x, y, x + w, y + h))

        cropped.thumbnail(thumb_size)

        face_list.append(cropped)



    return face_list





def make_contact_sheet(face_list):



    w_each, h_each = thumb_size



    w_sheet = w_each * 5

    h_sheet = h_each * (len(face_list) // 5)

    if(len(face_list) % 5 != 0):

        h_sheet += h_each

    

    sheet = PIL.Image.new(face_list[0].mode, (w_sheet, h_sheet))

    

    x = 0

    y = 0

    

    for face in face_list:

        sheet.paste(face, (x, y) )

        

        if x + w_each == sheet.width:

            x = 0

            y += h_each

        else:

            x += w_each

    

    return sheet
large_images_file = []

for fname in large_images:

    fdict_large_images = {}

    fdict_large_images['name'] = fname

    fdict_large_images['img'] = Image.open(fname)

    fdict_large_images['text'] = pytesseract.image_to_string(Image.open(fname).convert('L')).lower()

    fdict_large_images['faces'] = show_faces(fname)

    large_images_file.append(fdict_large_images)
#query = input("Kindly Enter Name (Mark/mark) to be searched for: ")

query = 'Mark'

for file in large_images_file:

    if query.lower() in file['text']:

        

        print("Results found in file " + file['name'])

        

        if len(file['faces']) == 0:

            print("But there were no faces in that file!")

        

        else:

            display(make_contact_sheet(file['faces']))
small_images = []

for images in range (0,4):

    small_images.append('/kaggle/input/newspaper-images/a-{}.png'.format(images))

    #print ('starting from image index {}'.format(images))  

                    
small_images_file = []

for fname in small_images:

    fdict_small_images = {}

    fdict_small_images['name'] = fname

    fdict_small_images['img'] = Image.open(fname)

    fdict_small_images['text'] = pytesseract.image_to_string(Image.open(fname).convert('L')).lower()

    fdict_small_images['faces'] = show_faces(fname)

    small_images_file.append(fdict_small_images)
#query = input("Kindly Enter Name (Christopher/christopher) to be searched for: ")

query = 'Christopher'

for file in small_images_file:

    if query.lower() in file['text']:

        

        print("Results found in file " + file['name'])

        

        if len(file['faces']) == 0:

            print("But there were no faces in that file!")

        

        else:

            display(make_contact_sheet(file['faces']))