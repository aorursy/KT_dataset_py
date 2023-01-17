%matplotlib inline
%load_ext autoreload
%autoreload 2

import math
import matplotlib.pyplot as plt
from sklearn.externals import joblib

def plot_list(images=[], labels=[], n_rows=1):
    n_img = len(images)
    n_lab = len(labels)
    n_cols = math.ceil((n_lab+n_img)/n_rows)
    plt.figure(figsize=(16,10))
    for i, image in enumerate(images):
        plt.subplot(n_rows,n_cols,i+1)
        plt.imshow(image)
    for j, label in enumerate(labels):
        plt.subplot(n_rows,n_cols,n_img+j+1)
        plt.imshow(label, cmap='nipy_spectral')
    plt.show()
non_nuclei_images = joblib.load('../input/non-nuclei-images/non_nuclei_images.pkl')
plot_list(non_nuclei_images, n_rows=4)
