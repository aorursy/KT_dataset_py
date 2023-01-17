# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
from PIL import Image
import matplotlib.pyplot as plt
!ls '../input/somaset/somaset/01/06/'
img_array = np.array(Image.open('../input/somaset/somaset/01/06/0026.jpg'))
plt.imshow(img_array)
