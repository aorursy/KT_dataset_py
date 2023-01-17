import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import random
from scipy import stats
import os
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
image_path = '../input/i2a2-bone-age-regression/images/'
test_df = pd.read_csv('../input/i2a2-bone-age-regression/test.csv')
image_name = test_df['fileName'][random.randint(0,test_df.shape[0])]
print(image_name)
im = Image.open(os.path.join(image_path,image_name))
print(im.format, im.size, im.mode)
plt.imshow(im)
im_arr = np.asarray(im)
statsval = stats.describe(np.ndarray.flatten(im_arr[0:10]))
print(statsval)
im_enh = ImageEnhance.Contrast(im)
im_enh = im_enh.enhance(10)
plt.imshow(im_enh)
im_enh_bw = im_enh.convert('1')
plt.imshow(im_enh_bw)
im_enh_bw_flip = ImageOps.flip(im_enh_bw)
plt.imshow(im_enh_bw_flip)
im_arr = np.asarray(im_enh_bw_flip, dtype=int)
print(im_arr)
im_arr.shape
im_cont_x = []
im_cont_y = []
for i in (range(0, im_arr.shape[0])):
    for j in (range(0, im_arr.shape[1])):
        if im_arr[i][j] > 0:
            #line = 0
            im_cont_y.append(i)
            im_cont_x.append(j)
plt.scatter(im_cont_x,im_cont_y)
im_cont_x_arr = np.asarray(im_cont_x)
im_cont_y_arr = np.asarray(im_cont_y)
im_cont_arr = np.stack((im_cont_x_arr, im_cont_y_arr),axis=1)
print(im_cont_arr.shape)
kmeans = KMeans(n_clusters=2, random_state=0).fit(im_cont_arr)
labels = kmeans.labels_

print(labels.shape)
print(labels)
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
 
ax.scatter(im_cont_x_arr, im_cont_y_arr,  c=labels, alpha=0.3, edgecolors='none')
ax.legend()
ax.grid(True)

plt.show()
clf = LogisticRegression(random_state=0).fit(im_cont_arr, labels)
clf.predict(im_cont_arr)

print(clf.coef_)
print(clf.intercept_)
grafico_a = []
grafico_b = []
for a in range(0,im_arr.shape[1]):
    b = (-clf.intercept_-clf.coef_[0][0]*a)/clf.coef_[0][1]
    if (b >= 0) and (b <= im_arr.shape[0]): 
        grafico_a.append(a)
        grafico_b.append(int(b))
plot, ax = plt.subplots()

    
ax.scatter(im_cont_x_arr, im_cont_y_arr,  c=labels, alpha=0.3, edgecolors='none')


ax.grid(True)


ax.plot(grafico_a,grafico_b)

plt.show()
plot, ax = plt.subplots()
plt.imshow(ImageOps.flip(im))
im_arr_crop = np.asarray(im_enh, dtype=int)
print(im_arr_crop.shape)
plot, ax = plt.subplots()
plt.imshow(ImageOps.flip(im))
grafico_a = []

grafico_b = []
for a in range(0,im_arr.shape[1]):
    b = (-clf.intercept_-clf.coef_[0][0]*a)/clf.coef_[0][1]
    if (b >= 0) and (b <= im_arr.shape[0]): 
        grafico_a.append(a)
        grafico_b.append(int(b))

ax.plot(grafico_a,grafico_b)
plt.show()
grafico_a_arr = np.asarray(grafico_a)
grafico_b_arr = np.asarray(grafico_b)
grafico_arr = np.stack((grafico_a_arr, grafico_b_arr),axis=1)
im_final_1 = np.asarray(ImageOps.flip(im)).copy()
c=0
for i in range(0,grafico_arr.shape[0]):
    
    a = grafico_arr[i][0]
    b = grafico_arr[i][1]  
    
    for j in range(c,b):
        for n in range(0,a):
            im_temp = im_final_1
            im_temp[j][n] = random.randint(statsval[1][0],statsval[1][1])
    c = b
im_crop = Image.fromarray(im_temp)
im_crop = im_crop.crop((grafico_a_arr.min(),0,im_arr.shape[0],im_arr.shape[1]))
plt.imshow(im_crop)
im_crop_unflip = ImageOps.flip(im_crop)
plt.imshow(bbbbbbb)
