import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import pandas as pd
import numpy as np
from PIL import Image
import io
import os
os.listdir("../input/wheres-waldo/Hey-Waldo/")
wdir = "../input/wheres-waldo/Hey-Waldo/64/"
waldos = np.array([np.array(imread(wdir + "waldo/"+fname)) for fname in os.listdir(wdir + "waldo")])
notwaldos = np.array([np.array(imread(wdir + "notwaldo/"+fname)) for fname in os.listdir(wdir + "notwaldo")])
plt.imshow(waldos[1])
data = []
for im in waldos:
    data.append(im.flatten('F'))
df1 = pd.DataFrame(data)
df1['waldo'] = 1
data = []
for im in notwaldos:
    data.append(im.flatten('F'))
df2 = pd.DataFrame(data)
df2['waldo'] = 0
frames = [df1, df2]
allwaldos = pd.concat(frames)
allwaldos.to_csv('all_waldo64.csv',index=False)
d = df1.iloc[1].drop('waldo').values.astype('uint8').reshape(3, 64, 64).transpose().reshape(64,64, 3)
plt.imshow(d)
(waldos[1] == d).all()