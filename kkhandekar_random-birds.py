# --- Libraries ---



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Visiulazation

import matplotlib.pyplot as plt



#Read Image

import cv2



#Random Libary

import random



#Open URL

from urllib.request import urlopen
# --- Load Data ---

url = '../input/birdsongrecognitiondetails/bird_details.csv'

data = pd.read_csv(url, header='infer')
def random_bird(readFlag=cv2.IMREAD_COLOR):



    rec_cnt = data.image.count()    #get species count

    rand_indx = random.randrange(0,rec_cnt)   #generate a random number between 0 & rec_cnt

    

    #get the link image  & read it

    lnk = data.image[rand_indx]

    resp = urlopen(lnk)

    image = np.asarray(bytearray(resp.read()), dtype="uint8")

    image = cv2.imdecode(image, readFlag)

    

    #display the image

    plt.figure(figsize=(10, 8))

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.xticks([])

    plt.yticks([])

    

    #Add the bird name to the image

    

    font = {'family': 'serif', 'color': 'black', 'weight': 'bold', 'size': 25,}

    

    

    plt.text(100, 150 , data.species[rand_indx].capitalize(), horizontalalignment = 'center',

             verticalalignment = 'center', bbox=dict(facecolor='red', alpha=0.3),       

             fontdict = font)

    

    plt.show()

#Display Random Bird with Species Name

random_bird()