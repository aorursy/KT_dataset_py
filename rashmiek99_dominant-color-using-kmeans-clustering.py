import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import cv2

import warnings

warnings.filterwarnings('ignore')
image = cv2.imread('/kaggle/input/kmeans/color.jfif')



plt.imshow(image);
image.shape
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(with_mean = 0,with_std=1)
r = []

g = []

b = []



for line in image:

    for pixel in line:

        temp_b , temp_g,temp_r = pixel

        

        r.append(temp_r)

        g.append(temp_g)

        b.append(temp_b)

    

df = pd.DataFrame({'red': r,'green': g,'blue': b})



df['scaled_red'] = scaler.fit_transform(df[['red']])

df['scaled_green'] = scaler.fit_transform(df[['green']])

df['scaled_blue'] = scaler.fit_transform(df[['blue']])



df.head()
X = df[['scaled_red','scaled_green','scaled_blue']].values

X
SSE = []



for cluster in range(2,8): 

    kmeans = KMeans(n_clusters=cluster,random_state=42)

    kmeans.fit(X)

    

    pred_clusters = kmeans.predict(X)

    SSE.append(kmeans.inertia_)

    

frame = pd.DataFrame({'Cluster':range(2,8) , 'SSE':SSE})

print(frame)
plt.figure(figsize=(5,5))

plt.plot(frame['Cluster'],frame['SSE'],marker='o')

plt.title('Clusters Vs SSE')

plt.xlabel('No of Clusters')

plt.ylabel('Intertia')

plt.show()
#Fit and predict for k = 4

k=4

kmeans = KMeans(n_clusters=k)

kmeans.fit(X)

k_pred = kmeans.predict(X)



#These are the centroids of the clusters

cluster_centers = kmeans.cluster_centers_

cluster_centers
colors = []



r_std, g_std, b_std = df[['red', 'green', 'blue']].std()



for cluster_center in cluster_centers:

    scaled_r, scaled_g, scaled_b = cluster_center

    

    colors.append((

    scaled_r * r_std /255,

    scaled_g * g_std / 255,

    scaled_b * b_std/ 255

    ))

    

plt.imshow([colors])

plt.show()
res = cluster_centers[k_pred.flatten()]

result_image = res.reshape((image.shape))



im_bgr = result_image[:, :, [2, 1, 0]] #restoring the image in bgr form



rescale_image = scaler.inverse_transform(im_bgr).astype(int) #rescaling it back to original



figure_size = 15

plt.figure(figsize=(figure_size,figure_size))

plt.subplot(1,2,1),plt.imshow(image)

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2),plt.imshow(rescale_image)

plt.title('Segmented Image when K = 4'), plt.xticks([]), plt.yticks([])

plt.show()