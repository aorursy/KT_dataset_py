# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/test/"))
print(os.listdir("../input/train"))
""# Any results you write to the current directory are saved as output.
import os
from PIL import Image
import numpy as np
from keras.models import load_model
import time
from tqdm import tqdm


def createSubmission(filename,coords,classes,test_directory = '../input/test/test/'):
    """
        @brief Function for creating a submission    
        @param filename filename for the competition, without ending
        @param coords predicted x,y  coorinates
        @param classes predicted classes in categorical encoding
        @param test_directory directory where the test images are located
    """
    
    # check the data
    if coords.shape != (225,2):
        raise ValueError('coords must have shape (225,2)')
 
    if classes.shape != (225,31):
        raise ValueError('classes must have shape (225,31)') 
    
    files = os.listdir(test_directory)
    
    with open(filename + '.csv','w') as f:
        
        f.write('Nr,X,Y,' + ",".join(['C' + str(i) for i in range(0,31)])+'\n')
    
        for n,coords,id_ in zip(classes,coords,files):
            
            # set the maxium class to 1 
            d = np.zeros(shape=(31,))
            d[np.argmax(n[0])] = 1.0

            class_ = str(d.tolist()).replace('[','').replace(']','').replace(' ','')
            x_coords = coords[0] 
            y_coords = coords[1]
            
            string = str(id_) + ',' + str(x_coords) + ',' + str(y_coords) + ',' + str(class_) + '\n'
           
                
            f.write(string)   
            

            
# using random numbers for prediction
classes = np.random.rand(225,31)
classes[classes >= 0.5] = 1.0
classes[classes != 1.0] = 0

createSubmission("submission.csv",np.random.rand(225,2),classes)
