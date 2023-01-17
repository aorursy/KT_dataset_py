# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from PIL import Image as im

from PIL import ImageDraw 

from matplotlib.pyplot import imshow

import random

import math

import numpy as np
def drawVoronoiDiagram(width,height,noOfLabels):

    

    img = im.new("RGB",(width,height))

    

    label_x = random.sample(range(width),noOfLabels)

    label_y = random.sample(range(height),noOfLabels)

    

    label_r = random.sample(range(256),noOfLabels)

    label_g = random.sample(range(256),noOfLabels)

    label_b = random.sample(range(256),noOfLabels)

    

    for i in range(width):

        for j in range(height):

            

            minDist = math.sqrt((label_x[0] - i)**2+(label_y[0]-j)**2)

            minIdx  = 0

            

            for itr in range(noOfLabels):

                

                if math.sqrt((label_x[itr] - i)**2+(label_y[itr]-j)**2) < minDist:

                    minDist = math.sqrt((label_x[itr] - i)**2+(label_y[itr]-j)**2)

                    minIdx  = itr

                    

                img.putpixel((i,j),(label_r[minIdx],label_g[minIdx],label_b[minIdx]))

            

            

    draw = ImageDraw.ImageDraw(img)

    

    for itr in range(noOfLabels):

        draw.ellipse([label_x[itr]-3,label_y[itr]-3,label_x[itr]+3,label_y[itr]+3])

        

    

    img.save("voronoiSampleImg.png","PNG")

    imshow(np.asarray(img))

    

    
drawVoronoiDiagram(700,700,25)