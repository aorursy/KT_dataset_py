# This Python 3 environment comes with many helpful analytics libraries installed

import  numpy as np

import pandas as pd

import os

import cv2

import matplotlib.pyplot as plt

%matplotlib inline
train_dir = '..input/kermany2018/oct2017/OCT2017 /train'

valid_dir = '..input/kermany2018/oct2017/OCT2017 /val'

test_dir = '..input/kermany2018/oct2017/OCT2017 /test'

classes = ['CNV','DME','DRUSEN','NORMAL']
image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/CNV/CNV-13823-3.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'CNV',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'orange', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'CNV',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/DME/DME-30521-4.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'DME',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'blue', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'DME',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/DRUSEN/DRUSEN-95633-4.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'DRUSEN',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'green', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'DRUSEN',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/NORMAL/NORMAL-469935-33.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'NORMAL',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'red', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'NORMAL',fontsize=15)

plt.tight_layout()

plt.show()
image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/CNV/CNV-13823-4.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'CNV',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'orange', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'CNV',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/DME/DME-30521-41.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'DME',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'blue', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'DME',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/DRUSEN/DRUSEN-1016042-4.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'DRUSEN',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'green', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'DRUSEN',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/NORMAL/NORMAL-1004480-7.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'NORMAL',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'red', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'NORMAL',fontsize=15)

plt.tight_layout()

plt.show()
image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/CNV/CNV-28682-1.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'CNV',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'orange', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'CNV',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/DME/DME-82328-6.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'DME',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'blue', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'DME',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/DRUSEN/DRUSEN-1130960-46.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'DRUSEN',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'green', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'DRUSEN',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/NORMAL/NORMAL-1015755-20.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'NORMAL',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'red', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'NORMAL',fontsize=15)

plt.tight_layout()

plt.show()
image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/CNV/CNV-53018-6.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'CNV',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'orange', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'CNV',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/DME/DME-323904-4.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'DME',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'blue', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'DME',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/DRUSEN/DRUSEN-1146923-12.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'DRUSEN',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'green', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'DRUSEN',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/NORMAL/NORMAL-1016042-59.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'NORMAL',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'red', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'NORMAL',fontsize=15)

plt.tight_layout()

plt.show()
image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/CNV/CNV-103044-171.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'CNV',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'orange', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'CNV',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/DME/DME-439112-1.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'DME',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'blue', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'DME',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/DRUSEN/DRUSEN-1173253-8.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'DRUSEN',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'green', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'DRUSEN',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/NORMAL/NORMAL-1027133-10.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'NORMAL',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'red', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'NORMAL',fontsize=15)

plt.tight_layout()

plt.show()
image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/CNV/CNV-163081-98.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'CNV',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'orange', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'CNV',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/DME/DME-462675-51.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'DME',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'blue', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'DME',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/DRUSEN/DRUSEN-1193659-4.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'DRUSEN',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'green', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'DRUSEN',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/NORMAL/NORMAL-1058176-1.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'NORMAL',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'red', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'NORMAL',fontsize=15)

plt.tight_layout()

plt.show()
image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/CNV/CNV-471800-4.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'CNV',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'orange', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'CNV',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/DME/DME-539366-2.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'DME',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'blue', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'DME',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/DRUSEN/DRUSEN-1487749-44.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'DRUSEN',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'green', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'DRUSEN',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/NORMAL/NORMAL-1067440-4.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'NORMAL',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'red', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'NORMAL',fontsize=15)

plt.tight_layout()

plt.show()
image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/CNV/CNV-9997680-25.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'CNV',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'orange', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'CNV',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/DME/DME-626033-1.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'DME',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'blue', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'DME',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/DRUSEN/DRUSEN-1793499-25.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'DRUSEN',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'green', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'DRUSEN',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/NORMAL/NORMAL-1356232-1.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'NORMAL',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'red', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'NORMAL',fontsize=15)

plt.tight_layout()

plt.show()
image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/CNV/CNV-9935363-10.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'CNV',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'orange', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'CNV',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/DME/DME-1597899-5.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'DME',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'blue', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'DME',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/DRUSEN/DRUSEN-2257047-66.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'DRUSEN',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'green', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'DRUSEN',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/NORMAL/NORMAL-1597899-3.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'NORMAL',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'red', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'NORMAL',fontsize=15)

plt.tight_layout()

plt.show()
image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/CNV/CNV-9884539-13.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'CNV',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'orange', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'CNV',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/DME/DME-8231523-10.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'DME',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'blue', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'DME',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/DRUSEN/DRUSEN-3281144-8.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'DRUSEN',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'green', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'DRUSEN',fontsize=15)

plt.tight_layout()

plt.show()



image = cv2.imread('../input/kermany2018/oct2017/OCT2017 /train/NORMAL/NORMAL-1690970-4.jpeg',0)

plt.subplot(121)

plt.imshow(image, 'gray')

_= plt.xlabel( 'NORMAL',fontsize=15)

plt.subplot(122)

_= plt.hist(image.ravel(), bins = 256, color = 'red', )

_= plt.title( 'Intensity Value' ,fontsize=15 )

_= plt.ylabel( 'Count',fontsize=15)

_= plt.xlabel( 'NORMAL',fontsize=15)

plt.tight_layout()

plt.show()