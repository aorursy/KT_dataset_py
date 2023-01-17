import numpy as np # linear algebra
from skimage import io
from matplotlib import pyplot
import os

def get_item(person,pose):
    """ Returns an image of FacePix dataset.
    Inputs:
        person: Person number, in range(1,30).
        pose: Pose angle of face image, in range(-90,+90)"""
    
    facepix_path = "../input/facepix/FacePix/"
    img_path = facepix_path + "/" + str(person) + "("+ str(pose) +").jpg"
    img = io.imread(img_path)
    return img

# Reading 20'th person's face image in pose -45 degree
img = get_item(20,-45) 
pyplot.imshow(img,cmap='gray')
pyplot.show()