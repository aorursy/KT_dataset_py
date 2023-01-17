#imports

import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt
def brightness_augment(img, factor=0.5): 

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #convert to hsv

    hsv = np.array(hsv, dtype=np.float64)

    hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform()) #scale channel V uniformly

    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values

    rgb = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)

    return rgb

    
num_samples = 3

df_data = pd.read_csv('../input/train_v2.csv')

subset = df_data.sample(num_samples).reset_index(drop = True)



plt.figure(figsize=(15,15))

for i,j in zip(range(num_samples), range(num_samples)):

    plt.subplot(num_samples,2,i+1+j)

    img = cv2.imread('../input/train-jpg/{}.jpg'.format(subset['image_name'][i]))

    plt.imshow(img)

    plt.subplot(num_samples,2,i+2+j)

    plt.imshow(brightness_augment(img))

plt.show()