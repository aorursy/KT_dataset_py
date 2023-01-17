import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import cv2
import matplotlib.pyplot as plt  

train_tiffs = os.listdir("../input/train/")
print(train_tiffs[:5])

for tiff in train_tiffs[:20]:
    im = cv2.imread("../input/train/" + tiff, cv2.IMREAD_GRAYSCALE)
    
    plt.imshow(im, cmap='gray')
    plt.show()

