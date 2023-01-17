import cv2

import matplotlib.pyplot as plt

import os
PATH = "../input/sun2012-subset/Images/Images"
img_paths = os.listdir(PATH)
import random
i = random.randrange(0,len(img_paths))



img = cv2.imread(os.path.join(PATH, img_paths[i]))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

plt.show()
import pandas as pd

import numpy as np
df_imgs = pd.read_pickle("../input/sun2012-subset/SUN2012.pkl")

df_imgs.head()
i = random.randrange(0,len(df_imgs))



img = np.reshape(np.array(df_imgs["Images"][i]),(224,224,3))

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)

plt.show()