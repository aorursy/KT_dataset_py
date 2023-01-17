import os

import re

import sys

import time

import json

import requests

import urllib.request

import numpy as np

import pandas as pd

from bs4 import BeautifulSoup
def log_progress(sequence, every=None, size=None, name='Items'):

    from ipywidgets import IntProgress, HTML, VBox

    from IPython.display import display



    is_iterator = False

    if size is None:

        try:

            size = len(sequence)

        except TypeError:

            is_iterator = True

    if size is not None:

        if every is None:

            if size <= 200:

                every = 1

            else:

                every = int(size / 200)     # every 0.5%

    else:

        assert every is not None, 'sequence is iterator, set every'



    if is_iterator:

        progress = IntProgress(min=0, max=1, value=1)

        progress.bar_style = 'info'

    else:

        progress = IntProgress(min=0, max=size, value=0)

    label = HTML()

    box = VBox(children=[label, progress])

    display(box)



    index = 0

    try:

        for index, record in enumerate(sequence, 1):

            if index == 1 or index % every == 0:

                if is_iterator:

                    label.value = '{name}: {index} / ?'.format(

                        name=name,

                        index=index

                    )

                else:

                    progress.value = index

                    label.value = u'{name}: {index} / {size}'.format(

                        name=name,

                        index=index,

                        size=size

                    )

            yield record

    except:

        progress.bar_style = 'danger'

        raise

    else:

        progress.bar_style = 'success'

        progress.value = index

        label.value = "{name}: {index}".format(

            name=name,

            index=str(index or '?')

        )
class Point:



    def __init__(self, xcoord=0, ycoord=0):

        self.x = xcoord

        self.y = ycoord



class Rectangle:

    def __init__(self, bottom_left, top_right, colour):

        self.bottom_left = bottom_left

        self.top_right = top_right

        self.colour = colour



    def intersects(self, other):

        return not (self.top_right.x < other.bottom_left.x or self.bottom_left.x > other.top_right.x or self.top_right.y < other.bottom_left.y or self.bottom_left.y > other.top_right.y)
r1=Rectangle(Point(1,1), Point(2,2), 'blue')

r3=Rectangle(Point(1.5,0), Point(1.7,3), 'red')

r1.intersects(r3)
def scrape_log(code,debug):

    timer = time.time()

    base_url = 'https://nhentai.net/g/{}/'

    ret = requests.get(base_url.format(code))

    soup = BeautifulSoup(ret.content, "lxml")

    gallery = eval(soup.text.split('var gallery = new N.gallery(')[-1].split(');')[0])

    num_pages = gallery['num_pages']

    media_id = gallery['media_id']

    ext_li = ['jpg', 'png', 'jpeg', 'bmp', 'JPG', 'JPEG', 'PNG', 'BMP']

    ext = None

    images=[]

    for pi in log_progress(range(num_pages), every = 1):

        img_id = pi + 1

        if ext is None:

            for ext in ext_li:

                try:

                    img_url = f'https://i.nhentai.net/galleries/{media_id}/{img_id}.{ext}'

                    break

                except urllib.error.HTTPError:

                    continue

        else:

            img_url = f'https://i.nhentai.net/galleries/{media_id}/{img_id}.{ext}'

        try:

            url_response = urllib.request.urlopen(img_url)

            img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)

            image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

            image = cv2.cvtColor(image,cv2.COLOR_BGRA2BGR)

            images.append(image)

        except urllib.error.HTTPError:

                    continue

    return images
import cv2

import matplotlib.pyplot as plt

from skimage.transform import rotate

import math

import numpy as np

eye_cascade = cv2.CascadeClassifier('../input/eye.xml')

face_cascade = cv2.CascadeClassifier('../input/face.xml')

def detect(image,debug=False,eyesize=0.1):  

    faces = face_cascade.detectMultiScale(image,scaleFactor = 1.2, minNeighbors = 1, minSize = (80, 80))

    images=[]

    for i,(x, y, w, h) in enumerate(faces):

        #cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi = image[y:y+h, x:x+w]

        #(h, w) = roi.shape[:2]

        #print(h,w)

        #center = (w / 2, h / 2)

        #scale = 1.0

        #M = cv2.getRotationMatrix2D(center, -42, scale)

        #roi = cv2.warpAffine(roi, M, (h, w))

        eyes = eye_cascade.detectMultiScale(roi,scaleFactor = 1.01, minNeighbors = 5, minSize = (20, 20))

        eyes = sorted(eyes, key = lambda x: -1*x[2])    

        for i,(ex1, ey1, ew1, eh1) in enumerate(eyes):

            r1=Rectangle(Point(ex1,ey1), Point(ex1+ew1, ey1+eh1), 'blue') 

            for j,(ex2, ey2, ew2, eh2) in enumerate(eyes):

                if(i==j):

                    pass

                else:                    

                    r2=Rectangle(Point(ex2,ey2), Point(ex2+ew2, ey2+eh2), 'blue')

                    print(i,' vs ',j,' : ',r1.intersects(r2))

                    if(r1.intersects(r2)):

                        eyes.pop(j)

        for i,(ex, ey, ew, eh) in enumerate(eyes):

            print()

            cv2.rectangle(roi, (ex, ey), (ex+ew, ey+eh), (255,255,0), 2)            

                    

        if(len(eyes)>0):

            images.append(roi)  

            if(debug==True):                

                plt.figure()

                plt.imshow(roi)   

    return images
query = 'https://nhentai.net/tag/ahegao/popular'

ret = requests.get(query)

soup = BeautifulSoup(ret.text, 'html.parser')
g = []

for link in soup.find_all('a'):

    if re.match(r'^\/g/[0-9]*/', link.get('href')):

        numbers = re.findall('\d+',link.get('href')) 

        id=int(numbers[0])

        g.append(id)

print(g)
images=scrape_log(108419,debug=True)
detect(images[3],debug=True,eyesize=0.1);