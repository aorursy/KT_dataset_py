!tar -xvf /kaggle/input/diabeticretinopathydetectionpretraineddata/DiabeticRetinopathyDetection.tar.xz 
import os

os.chdir('/kaggle/working/DiabeticRetinopathyDetection')
!bash classify.sh sample/Level-4.jpeg
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

image = mpimg.imread("sample/Level-4.jpeg")

plt.imshow(image)

plt.show()