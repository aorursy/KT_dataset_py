import cv2

print(cv2.__version__)

import urllib.request

from matplotlib import pyplot as plt

%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from PIL import Image

jpgfile = Image.open("../input/-12345/donald.jpg")

 

plt.imshow(jpgfile)
pics = cv2.imread("../input/-12345/donald.jpg")

plt.imshow(pics)
from pylab import rcParams



def plot_show(image,title = '', gray=False, size = (12,10)):

    temp = image

    # fix color

    if gray==False:

        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)

    #change image size

    rcParams['figure.figsize'] = size[0],size[1]

    #remove axis ticks

    plt.axis('off')

    plt.title('title')

    plt.imshow(temp)

    plt.show

    
plot_show(pics)
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

haarcascade_nam = "haarcascade_frontalface_default.xml"



urllib.request.urlretrieve(haarcascade_url,haarcascade_nam)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faces_list = detector.detectMultiScale(pics)

print(faces_list)
for faces in faces_list:

    (x,y,w,h)= faces

    cv2.rectangle(pics,(x,y),(x+w,y+h), (0,255,0),3)

plot_show(pics)