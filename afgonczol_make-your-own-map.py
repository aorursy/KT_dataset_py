import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
import os
print(os.listdir("../input"))
from tqdm import tqdm

# Any results you write to the current directory are saved as output.
img = cv2.imread("../input/cat-image/download.jpg", cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='Greys_r');
img.shape
cutoff = 80
cityid = 0
cities = np.array([-1, 0, 0]).reshape(1, 3)
for i, row in tqdm(enumerate(img)):
    for j, col in enumerate(row):
        if col > cutoff:
            cities = np.concatenate((cities, np.array([cityid, j, img.shape[1] - i]).reshape(1,3)))
            cityid +=1
cities = cities[1:]
cities = pd.DataFrame(cities)
cities.columns = ['cityId', 'X', 'Y']
fig = plt.figure(figsize=(4.8, 3.6) )
ax = fig.gca()
ax.set_facecolor('Black')
ax.set_xticks([])
ax.set_yticks([])
plt.scatter(cities.X, cities.Y, color='White', marker=".", alpha=.1);

cities.to_csv('cat.csv')

