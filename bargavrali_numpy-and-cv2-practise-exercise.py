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
#
# Hi, I am practising Data Science Concepts and below are small Programs that will  
# Create image drawings of 
# a Line shape, 
# a Square shape 
# a Car shape
#
#
import numpy as np
import cv2 #computer vision
import numpy as np
import cv2 #computer vision
data=np.ones((100,100))
data[20:40,20:40]=0
cv2.imshow("Creating Square",data)
cv2.waitKey()
cv2.destroyAllWindows()
import numpy as np
import cv2 #computer vision
data=np.ones((100,100))
data[50:52,10:90]=0
cv2.imshow("Creating Line",data)
cv2.waitKey()
cv2.destroyAllWindows()
import numpy as np
import cv2 #computer vision
data=np.zeros((100,100))
data[50,30:70]=1
data[51,29:71]=1
data[52,28:72]=1
data[53,27:73]=1
data[54,26:74]=1
data[55,25:75]=1
data[56,24:76]=1
data[57,23:77]=1
data[58,22:78]=1
data[59,21:79]=1
data[60,20:80]=1
data[60:75,10:90]=1
data[75:77,20:30]=1
data[75:77,70:80]=1
data[77:78,21:29]=1
data[77:78,71:79]=1
data[78:79,22:28]=1
data[78:79,72:78]=1
data[79:80,23:27]=1
data[79:80,73:77]=1
cv2.imshow("Creating car",data)
cv2.waitKey()
cv2.destroyAllWindows()