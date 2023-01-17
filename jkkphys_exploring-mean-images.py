%matplotlib inline

import matplotlib.pyplot as plt

from scipy.stats import mode

import pandas as pd

import numpy as np



plt.style.use('ggplot')

# Set base path to images:

base_path = '../input/sample/images/'
def get_mean_image(fnames, base_path):

    '''Computes mean intensity image for given filenames'''

    imgs = []

    for i, fname in enumerate(fnames):

        img = plt.imread(base_path + fname)

        if img.shape == (1024,1024):

            imgs.append(img)

    imgs = np.array(imgs)

    return np.mean(imgs, axis=0)
df = pd.read_csv('../input/sample_labels.csv')

df.head()
df['View Position'].value_counts()
fnames_m_ap = df[(df['Patient Gender']=='M') & (df['View Position']=='AP')]['Image Index'].values

img_m_ap = get_mean_image(fnames_m_ap, base_path)
fnames_m_pa = df[(df['Patient Gender']=='M') & (df['View Position']=='PA')]['Image Index'].values

img_m_pa = get_mean_image(fnames_m_pa, base_path)
fnames_f_ap = df[(df['Patient Gender']=='F') & (df['View Position']=='AP')]['Image Index'].values

img_f_ap = get_mean_image(fnames_f_ap, base_path)
fnames_m_pa = df[(df['Patient Gender']=='F') & (df['View Position']=='PA')]['Image Index'].values

img_f_pa = get_mean_image(fnames_m_pa, base_path)
img_f_pa[0]
img_f_ap[0]
fig = plt.figure(figsize=(10,10))



ax_0 = fig.add_subplot(221)

ax_0.imshow(img_m_ap)

ax_0.set_title('Male AP', fontsize=16)

ax_0.axis('off')



ax_1 = fig.add_subplot(222)

ax_1.imshow(img_m_pa)

ax_1.set_title('Male PA', fontsize=16)

ax_1.axis('off')



ax_2 = fig.add_subplot(223)

ax_2.imshow(img_f_ap)

ax_2.set_title('Female AP', fontsize=16)

ax_2.axis('off')



ax_3 = fig.add_subplot(224)

ax_3.imshow(img_f_pa)

ax_3.set_title('Female PA', fontsize=16)

ax_3.axis('off')



plt.show()