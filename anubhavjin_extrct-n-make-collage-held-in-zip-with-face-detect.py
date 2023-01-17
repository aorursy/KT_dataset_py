# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Visiulazation
import matplotlib.pyplot as plt

#image processing
import cv2

#extracting zippped file
import tarfile

#systems
import os
print(os.listdir("../input/haarcascades"))
print(os.listdir("../input/lfwpeople"))
class faceDetector():
    def __init__(self,facecascadepath):
        self.faceCascade=cv2.CascadeClassifier(facecascadepath)
    
    def detect(self,image,scaleFactor=1.1,minNeighbors=5,minSize=(30,30)):
        rects=self.faceCascade.detectMultiScale(image,scaleFactor=scaleFactor,
                                                minNeighbors=minNeighbors,
                                                minSize=minSize)
        return rects
frontal_cascade_path='/kaggle/input/haarcascades/haarcascade_frontalface_default.xml'
fd=faceDetector(frontal_cascade_path)
def show_image(image):
    plt.figure(figsize=(18,15))
    #Before showing image, bgr color order transformed to rgb order
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()
def detect_face(image, scaleFactor, minNeighbors, minSize):
    image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=fd.detect(image_grey)
    for(x,y,w,h) in faces:
        c2.reactangle(image,(x,y),(x+w,y+h),(127,255,0),3)
    show_image(image)
class FetchLFW:
    def  __init__(self,path):
        self.path=path
    
    def _initalize(self,dim):
        self.dim_of_photo_gallery=dim
        self.number_of_images=self.dim_of_photo_gallery*self.dim_of_photo_gallery
        
        total_number_images=13233
        self.random_face_indexes=np.arange(total_number_images)
        np.random.shuffle(self.random_face_indexes)
        self.n_random_face_indexes=self.random_face_indexes[:self.number_of_images]
        
    def get_lfw_image(self,dim=5):
        self._initalize(dim)
        self.lfw_images=self._get_images()
        return self.lfw_images
    def _get_images(self):
        image_list=[]
        tar=tarfile.open(path,'r:gz')
        counter=0
        for tarinfo in tar:
            tar.extract(tarinfo.name)
            if tarinfo.name[-4:]=='.jpg':
                if counter in self.n_random_face_indexes:
                    image=cv2.imread(tarinfo.name,cv2.IMREAD_COLOR)
                    image=cv2.resize(image,None,fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
                    image_list.append(np.array(image))
                counter+=1
            if tarinfo.isdir():
                pass
            else:
                os.remove(tarinfo.name)
        tar.close()
        return np.array(image_list)
path="../input/lfwpeople/lfw-funneled.tgz"
fetchLFW=FetchLFW(path)
dimension=15
images=fetchLFW.get_lfw_image(dim=dimension)
images.shape
def get_photo_gallery():
    counter=0
    himages=[]
    vimages=[]
    for i in range(dimension):
        for j in range(dimension):
            himages.append(images[counter])
            counter+=1
        himage=np.hstack((himages))
        vimages.append(himage)
        himages=[]
    images_matrix=np.vstack((vimages))
    return images_matrix
photo_gallery=get_photo_gallery()
print("photo_gallery:{}".format(photo_gallery.shape))
show_image(photo_gallery)
fd=faceDetector(frontal_cascade_path)
face_counter=0
for image_org in images:
    image_gray=cv2.cvtColor(image_org,cv2.COLOR_BGR2GRAY)
    faceRect=fd.detect(image_gray,
                       scaleFactor=1.1,
                       minNeighbors=5,
                       minSize=(30,30))
    #print("I found {} faces".format(len(faceRect)))
    first_detection=False
    for (x,y,w,h) in faceRect:
        if first_detection==False:
            face_counter+=1
            cv2.rectangle(image_org,(x,y),(x+w,y+h),(127,255,0),2)
            first_detection=True
        else:
            print("Second detection ignored in a image")

print("{} images have been scaned".format(dimension*dimension))
print("{} faces have been detected".format(face_counter))
photo_gallery=get_photo_gallery()
show_image(photo_gallery)
