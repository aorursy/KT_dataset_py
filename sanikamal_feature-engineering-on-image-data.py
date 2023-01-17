from IPython.display import Image
Image(filename='../input/data/data/clear/IMG_9638.JPG') 
import skimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io

%matplotlib inline
butterfly_one = io.imread('../input/data/data/clear/IMG_9404.JPG')
butterfly_two = io.imread('../input/data/data/clear/IMG_7013.JPG')
df = pd.DataFrame(['butterfly_one', 'butterfly_two'], columns=['Image'])


print(butterfly_one.shape, butterfly_two.shape)
fig = plt.figure(figsize = (15,4))
ax1 = fig.add_subplot(1,2, 1)
ax1.imshow(butterfly_one)
ax2 = fig.add_subplot(1,2, 2)
ax2.imshow(butterfly_two)
butterfly_one_r = butterfly_one.copy() # Red Channel
butterfly_one_r[:,:,1] = butterfly_one_r[:,:,2] = 0 # set G,B pixels = 0
butterfly_one_g = butterfly_one.copy() # Green Channel
butterfly_one_g[:,:,0] = butterfly_one_r[:,:,2] = 0 # set R,B pixels = 0
butterfly_one_b = butterfly_one.copy() # Blue Channel
butterfly_one_b[:,:,0] = butterfly_one_b[:,:,1] = 0 # set R,G pixels = 0

plot_image = np.concatenate((butterfly_one_r, butterfly_one_g, butterfly_one_b), axis=1)
plt.figure(figsize = (15,4))
plt.imshow(plot_image)
butterfly_one_r
from skimage.color import rgb2gray

butterfly_one_gs = rgb2gray(butterfly_one)
butterfly_two_gs = rgb2gray(butterfly_two)

print('Image shape:', butterfly_one_gs.shape, '\n')

# 2D pixel map
print('2D image pixel map')
print(np.round(butterfly_one_gs, 2), '\n')

# flattened pixel feature vector
print('Flattened pixel map:', (np.round(butterfly_one_gs.flatten(), 2)))
fig = plt.figure(figsize = (8,4))
ax1 = fig.add_subplot(2,2, 1)
ax1.imshow(butterfly_one_gs, cmap="gray")
ax2 = fig.add_subplot(2,2, 2)
ax2.imshow(butterfly_two_gs, cmap='gray')
ax3 = fig.add_subplot(2,2, 3)
c_freq, c_bins, c_patches = ax3.hist(butterfly_one_gs.flatten(), bins=30)
ax4 = fig.add_subplot(2,2, 4)
d_freq, d_bins, d_patches = ax4.hist(butterfly_two_gs.flatten(), bins=30)
from scipy.stats import describe

butterfly_one_rgb = butterfly_one.reshape((3456*5184), 3).T
butterfly_two_rgb = butterfly_two.reshape((3456*5184), 3).T

cs = describe(butterfly_one_rgb, axis=1)
ds = describe(butterfly_two_rgb, axis=1)

butterfly_one_rgb_range = cs.minmax[1] - cs.minmax[0]
butterfly_two_rgb_range = ds.minmax[1] - ds.minmax[0]
rgb_range_df = pd.DataFrame([butterfly_one_rgb_range, butterfly_two_rgb_range], 
                            columns=['R_range', 'G_range', 'B_range'])
pd.concat([df, rgb_range_df], axis=1)
butterfly_one_stats= np.array([np.round(cs.mean, 2),np.round(cs.variance, 2),
                     np.round(cs.kurtosis, 2),np.round(cs.skewness, 2),
                     np.round(np.median(butterfly_one_rgb, axis=1), 2)]).flatten()
butterfly_two_stats= np.array([np.round(ds.mean, 2),np.round(ds.variance, 2),
                        np.round(ds.kurtosis, 2),np.round(ds.skewness, 2),
                        np.round(np.median(butterfly_two_rgb, axis=1), 2)]).flatten()

stats_df = pd.DataFrame([butterfly_one_stats, butterfly_two_stats],
                        columns=['R_mean', 'G_mean', 'B_mean', 
                                 'R_var', 'G_var', 'B_var',
                                 'R_kurt', 'G_kurt', 'B_kurt',
                                 'R_skew', 'G_skew', 'B_skew',
                                 'R_med', 'G_med', 'B_med'])
pd.concat([df, stats_df], axis=1)
from skimage.feature import canny

butterfly_one_edges = canny(butterfly_one_gs, sigma=3)
butterfly_two_edges = canny(butterfly_two_gs, sigma=3)

fig = plt.figure(figsize = (8,4))
ax1 = fig.add_subplot(1,2, 1)
ax1.imshow(butterfly_one_edges, cmap='binary')
ax2 = fig.add_subplot(1,2, 2)
ax2.imshow(butterfly_two_edges, cmap='binary')
from skimage.feature import hog
from skimage import exposure

fd_butterfly_one, butterfly_one_hog = hog(butterfly_one_gs, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(3, 3), visualize=True)
fd_butterfly_two, butterfly_two_hog = hog(butterfly_two_gs, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(3, 3), visualize=True)

# rescaling intensity to get better plots
butterfly_one_hogs = exposure.rescale_intensity(butterfly_one_hog, in_range=(0, 0.04))
butterfly_two_hogs = exposure.rescale_intensity(butterfly_two_hog, in_range=(0, 0.04))

fig = plt.figure(figsize = (15,4))
ax1 = fig.add_subplot(1,2, 1)
ax1.imshow(butterfly_one_hogs, cmap='binary')
ax2 = fig.add_subplot(1,2, 2)
ax2.imshow(butterfly_two_hogs, cmap='binary')
print(fd_butterfly_one, fd_butterfly_one.shape)
!pip install mahotas
from mahotas.features import surf
import mahotas as mh

butterfly_one_mh = mh.colors.rgb2gray(butterfly_one)
butterfly_two_mh = mh.colors.rgb2gray(butterfly_two)

butterfly_one_surf = surf.surf(butterfly_one_mh, nr_octaves=8, nr_scales=16, initial_step_size=1, threshold=0.1, max_points=50)
butterfly_two_surf = surf.surf(butterfly_two_mh, nr_octaves=8, nr_scales=16, initial_step_size=1, threshold=0.1, max_points=54)

fig = plt.figure(figsize = (15,4))
ax1 = fig.add_subplot(1,2, 1)
ax1.imshow(surf.show_surf(butterfly_one_mh, butterfly_one_surf))
ax2 = fig.add_subplot(1,2, 2)
ax2.imshow(surf.show_surf(butterfly_two_mh, butterfly_two_surf))
butterfly_one_surf_fds = surf.dense(butterfly_one_mh, spacing=10)
butterfly_two_surf_fds = surf.dense(butterfly_two_mh, spacing=10)
butterfly_one_surf_fds.shape
from sklearn.cluster import KMeans

k = 20
km = KMeans(k, n_init=100, max_iter=100)

surf_fd_features = np.array([butterfly_one_surf_fds, butterfly_two_surf_fds])
km.fit(np.concatenate(surf_fd_features))

vbow_features = []
for feature_desc in surf_fd_features:
    labels = km.predict(feature_desc)
    vbow = np.bincount(labels, minlength=k)
    vbow_features.append(vbow)

vbow_df = pd.DataFrame(vbow_features)
pd.concat([df, vbow_df], axis=1)
new_butterfly = io.imread('../input/data/data/clear/IMG_6848.JPG')
new_butterfly_mh = mh.colors.rgb2gray(new_butterfly)
new_butterfly_surf = surf.surf(new_butterfly_mh, nr_octaves=8, nr_scales=16, initial_step_size=1, threshold=0.1, max_points=50)

fig = plt.figure(figsize = (15,4))
ax1 = fig.add_subplot(1,2, 1)
ax1.imshow(surf.show_surf(new_butterfly_mh, new_butterfly_surf))
new_surf_fds = surf.dense(new_butterfly_mh, spacing=10)

labels = km.predict(new_surf_fds)
new_vbow = np.bincount(labels, minlength=k)
pd.DataFrame([new_vbow])
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

eucdis = euclidean_distances(new_vbow.reshape(1,-1) , vbow_features)
cossim = cosine_similarity(new_vbow.reshape(1,-1) , vbow_features)

result_df = pd.DataFrame({'EuclideanDistance': eucdis[0],
              'CosineSimilarity': cossim[0]})
pd.concat([df, result_df], axis=1)
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
model = Sequential()
model.add(Conv2D(4, (4, 4), input_shape=(168, 300, 3), activation='relu',kernel_initializer='glorot_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(4, (4, 4), activation='relu', kernel_initializer='glorot_uniform'))
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model, show_shapes=True, 
                 show_layer_names=True, rankdir='TB').create(prog='dot', format='svg'))
first_conv_layer = K.function([model.layers[0].input, K.learning_phase()], 
                              [model.layers[0].output])
second_conv_layer = K.function([model.layers[0].input, K.learning_phase()], 
                               [model.layers[2].output])
butterfly_one_r = butterfly_one.reshape(1, 3456,5184,3)

# extract feaures 
first_conv_features = first_conv_layer([butterfly_one_r])[0][0]
second_conv_features = second_conv_layer([butterfly_one_r])[0][0]

# view feature representations
fig = plt.figure(figsize = (14,4))
ax1 = fig.add_subplot(2,4, 1)
ax1.imshow(first_conv_features[:,:,0])
ax2 = fig.add_subplot(2,4, 2)
ax2.imshow(first_conv_features[:,:,1])
ax3 = fig.add_subplot(2,4, 3)
ax3.imshow(first_conv_features[:,:,2])
ax4 = fig.add_subplot(2,4, 4)
ax4.imshow(first_conv_features[:,:,3])

ax5 = fig.add_subplot(2,4, 5)
ax5.imshow(second_conv_features[:,:,0])
ax6 = fig.add_subplot(2,4, 6)
ax6.imshow(second_conv_features[:,:,1])
ax7 = fig.add_subplot(2,4, 7)
ax7.imshow(second_conv_features[:,:,2])
ax8 = fig.add_subplot(2,4, 8)
ax8.imshow(second_conv_features[:,:,3])
sample_features = np.round(np.array(first_conv_features[:,:,1], dtype='float'), 2)
print(sample_features)
print(sample_features.shape)