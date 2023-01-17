from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Finding the size of the images
im = Image.open("../input/driving_dataset/driving_dataset/0.jpg")
width, height = im.size
print('Original size: width {}, height {}'.format(width, height))
# Load txt file and create 'data' ndarray with <img_name, steer_command> 
txt_labels = pd.read_csv('../input/driving_dataset/driving_dataset/data.txt', sep=" ", header=None)
data = txt_labels.values
# Sample images
plt.figure(figsize=(20,50))
rand_idx = np.random.choice(data.shape[0], size=20, replace=False)
path = '../input/driving_dataset/driving_dataset/'
right_side = False
for i, idx_img in enumerate(rand_idx, 1):
    img_name = path + data[idx_img][0] 
    img = Image.open(img_name)
    if not right_side:
        plt.subplot(10,2,i)
        right_side = True
    else:
        plt.subplot(10,2,i)
        right_side = False
    plt.title('Img: {}, Steer: {}'.format(data[idx_img][0], data[idx_img][1]))
    plt.imshow(img)
steer = data[:][:,1]
steer = steer.astype(np.float16)
steer.shape
n, bins, patches = plt.hist(steer, 10, facecolor='g', alpha=0.75)
plt.xlabel('Steering')
plt.ylabel('Frequency')
plt.title('Distribution of Steering Commands')
bins = [ round(.05 * round(float(i)/.05, 2)) for i in bins ]
plt.xticks(bins)
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
# Refining hist size 
n, bins, patches = plt.hist(steer, bins=80, facecolor='g', alpha=0.75)
plt.xlabel('Steering')
plt.ylabel('Frequency')
plt.title('Distribution of Steering Commands')
bins = [ round(.05 * round(float(i)/.05, 2)) for i in bins ]
plt.xlim([0, bins[28]])
plt.xticks(bins[9:28])
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
