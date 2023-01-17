import numpy as np

import pandas as pd

import cv2

import os

import matplotlib

# matplotlib.use('nbagg')

import matplotlib.pyplot as plt

%matplotlib inline

# %matplotlib inline

# import mpld3

# mpld3.enable_notebook()
base_path = '/kaggle/input/alaska2-image-steganalysis/'

algorithm = ('Cover(Unaltered)', 'JMiPOD', 'UERD', 'JUNIWARD')

fig, axes = plt.subplots(nrows=4, ncols=4, figsize = (11,11) )

np.random.seed(55)

for i,id in enumerate(np.random.randint(0,75001,4)):

    id = '{:05d}'.format(id)    

    cover_path = os.path.join(base_path, 'Cover', id + '.jpg')

    jmipod_path = os.path.join(base_path, 'JMiPOD', id + '.jpg')

    uerd_path = os.path.join(base_path, 'UERD', id + '.jpg')

    juniward_path = os.path.join(base_path, 'JUNIWARD', id + '.jpg')

    cover_img = plt.imread(cover_path)

    jmipod_img = plt.imread(jmipod_path)

    uerd_img = plt.imread(uerd_path)

    juniward_img = plt.imread(juniward_path)

    axes[i,0].imshow(cover_img)

    axes[i,1].imshow(jmipod_img)

    axes[i,2].imshow(uerd_img)

    axes[i,3].imshow(juniward_img)

    axes[i,0].set(ylabel=id+'.jpg')



for i,algo in enumerate(algorithm):

    axes[0,i].set(title=algo) 

for ax in axes.flat:

    ax.set(xticks=[], yticks=[])

plt.show()
cover_hist = {}

jmipod_hist = {}

uerd_hist = {}

juniward_hist = {}

color = ('b','g','r')

for i,col in enumerate(color):

    cover_hist[col] = cv2.calcHist([cover_img],[i],None,[256],[0,256])

    jmipod_hist[col] = cv2.calcHist([jmipod_img],[i],None,[256],[0,256])

    uerd_hist[col] = cv2.calcHist([uerd_img],[i],None,[256],[0,256])

    juniward_hist[col] = cv2.calcHist([juniward_img],[i],None,[256],[0,256])

    

fig_hist, axes_hist = plt.subplots(nrows=2, ncols=2, figsize=(12,12))

for ax, hist, algo in zip(axes_hist.flat, [cover_hist, jmipod_hist, uerd_hist, juniward_hist], algorithm):

    ax.plot(hist['r'], color = 'r', label='r')

    ax.plot(hist['g'], color = 'g', label='g')

    ax.plot(hist['b'], color = 'b', label='b')

    ax.set(ylabel='# of pixels', xlabel='Pixel value(0-255)', title=algo)

    ax.legend()

fig_hist.subplots_adjust(wspace=0.4, hspace=0.3)

fig_hist.suptitle('Histogram of a sample (' + id + '.jpg)', fontsize=20)

    #     ax.xlim([0,256])

plt.show()
fig, ax = plt.subplots(figsize=(10,10))

ax.plot(cover_hist['r'][50:80], color = 'c', label=algorithm[0])

ax.plot(jmipod_hist['r'][50:80], color = 'm', label=algorithm[1])

ax.plot(uerd_hist['r'][50:80], color = 'y', label=algorithm[2])

ax.plot(juniward_hist['r'][50:80], color = 'g', label=algorithm[3])

ax.legend()

ax.set_ylabel('# of pixels', fontsize=15) 

ax.set_xlabel('Pixel value(50-80)', fontsize=15)

ax.xaxis.set(ticklabels=np.linspace(50,80,8, dtype=np.int))

ax.set_title('R-channel Histogram Compared (zoomed in)', fontsize=20)

plt.show()
print('Cover image:\n', cover_img[10:20,10:20,0])

print('\nJMiPOD image:\n', jmipod_img[10:20,10:20,0])

print('\nUERD image:\n', uerd_img[10:20,10:20,0])

print('\nJUNIWARD image:\n', juniward_img[10:20,10:20,0])
fig, axes = plt.subplots(nrows=4, ncols=4, figsize = (11,11) )

np.random.seed(55)

def disp_diff_img(alt, ref, ax, chnl=0):

    diff = np.abs(alt.astype(np.int)-ref.astype(np.int)).astype(np.uint8)

    ax.imshow(diff[:,:,chnl], vmin=0, vmax=np.amax(diff[:,:,chnl]), cmap='hot')

for i,id in enumerate(np.random.randint(0,75001,4)):

    id = '{:05d}'.format(id)    

    cover_path = os.path.join(base_path, 'Cover', id + '.jpg')

    jmipod_path = os.path.join(base_path, 'JMiPOD', id + '.jpg')

    uerd_path = os.path.join(base_path, 'UERD', id + '.jpg')

    juniward_path = os.path.join(base_path, 'JUNIWARD', id + '.jpg')

    cover_img = plt.imread(cover_path)

    jmipod_img = plt.imread(jmipod_path)

    uerd_img = plt.imread(uerd_path)

    juniward_img = plt.imread(juniward_path)

    axes[i,0].imshow(cover_img)

    disp_diff_img(jmipod_img, cover_img, axes[i,1], 0)

    disp_diff_img(uerd_img, cover_img, axes[i,2], 0)

    disp_diff_img(juniward_img, cover_img, axes[i,3], 0)

    axes[i,0].set(ylabel=id+'.jpg')



for i,algo in enumerate(algorithm):

    axes[0,i].set(title=algo + 'diff') 

for ax in axes.flat:

    ax.set(xticks=[], yticks=[])

plt.show()