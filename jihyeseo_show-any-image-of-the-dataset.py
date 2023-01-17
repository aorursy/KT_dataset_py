# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
from PIL import Image
import matplotlib.pyplot as plt
def display(subjectId, clothId, photoId):
    subjectId = str(subjectId).zfill(2)
    clothId = str(clothId).zfill(2)
    photoId = str(photoId).zfill(4)
    img_array = np.array(Image.open("../input/somaset/somaset/" + subjectId + "/" + clothId + "/" + photoId + ".jpg"))
    plt.imshow(img_array)
    return 

#subjectId : in range 1 to 48
# clothId : in range 1 to 8
# photoId : in range 1 to 250
display(1,6,30)
display(48,8,250)
display(48,8,249)
display(48,8,1)
