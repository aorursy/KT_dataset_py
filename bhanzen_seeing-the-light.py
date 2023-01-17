import numpy as np # linear algebra

import matplotlib.pyplot as plt # graphs

%matplotlib inline

from scipy.ndimage.filters import uniform_filter # to smooth images

import os

import copy

import skimage.feature

from sklearn.linear_model import LinearRegression

import astropy.stats as ap

from sklearn.cluster import KMeans

#import photutils as pu
wd = "../input"

print(os.listdir(wd))

datafile = os.path.join(wd, "data141110.csv")

data = np.loadtxt(datafile, delimiter=',', skiprows=1)

image_no=data[:,0].reshape(-1,1)

frame_no=data[:,1].reshape(-1,1)

time_hrs=data[:,2].reshape(-1,1)

nb_images = data.shape[0]
flo1file = os.path.join(wd, "flo_image_1.npz")

flo2file = os.path.join(wd, "flo_image_2.npz")

flo_image_1 = np.load(flo1file)

flo_image_2 = np.load(flo2file)

# group the data from the 2 files

image_ids = np.concatenate([flo_image_1['image_ids'], flo_image_2['image_ids']])

images = np.concatenate([flo_image_1['image_stack'], flo_image_2['image_stack']])

# these object are no more useful, free memory

del flo_image_1, flo_image_2
mean_values = np.mean(images, axis=(1,2))

model = LinearRegression()

model.fit(time_hrs, mean_values)

mean_linear = model.predict(time_hrs)

norm_values = mean_values - mean_linear

ffit = np.fft.fft(norm_values)

freqs = np.fft.fftfreq(ffit.shape[-1])

dfreqs = freqs*240

rffit = np.real(ffit)

cffit = np.imag(ffit)

power = np.sqrt(rffit*rffit+cffit*cffit)

plt.plot(dfreqs, power)
plt.plot(dfreqs[0:40], power[0:40])
indices = power>40000

pics = dfreqs[indices]

print(pics)

periodes = 1/pics

print(periodes)

print (power[indices])
fix, ax = plt.subplots(1,8, figsize=(13,8))

idx = [80, 200, 320, 440, 560, 740, 860, 980]

for j in range(8):

    ax[j].set_axis_off()

    ax[j].imshow(images[idx[j],:,:] - mean_linear[idx[j]], cmap='hot', vmin=-800, vmax=6400)

        
# reshape the data to be compatible with K-Means

imagesshape = images.shape

images = np.transpose(images.reshape(imagesshape[0], -1))

sets = 10 #number of regions we want to see

nimages=1685

sampling=3 #sampling factor: 3 is sample every third image. Needed to stay within the memory bounds

model = KMeans(n_clusters = sets, n_jobs=1)

clusters = model.fit(images[:, 0:nimages:sampling])

centroids = clusters.cluster_centers_
cmap = plt.get_cmap('Paired')

fig3, ax3 = plt.subplots(1,2, figsize=(13,8))

colors = [cmap(i) for i in np.linspace(0, 1, sets)]

for i in range(0, sets):

    ax3[0].plot(centroids[i], color=colors[i], label=i)

ax3[0].legend(loc=0)

zones = clusters.predict(images[:,0:nimages:sampling])

image = copy.deepcopy(zones)

image = image.reshape(512, 512)

ax3[1].imshow(image, cmap='Paired')
fig4, ax4 = plt.subplots(sets,4, figsize=(16, 4*sets))

for seq in range(0, sets):

    category = images[zones==seq,:]

    serie = np.mean(category, axis = 0)

    days = data[:,1].reshape(-1,1)/240

    model = LinearRegression()

    model.fit(days, serie)

    ax4[seq, 0].plot(days, model.predict(days), 'r')

    ax4[seq, 0].plot(days, serie, color = colors[seq])

    model_value = model.predict(days)

    detrended = serie - model_value

    ax4[seq, 1].plot(days, detrended, color = colors[seq])

    ffit = np.fft.fft(detrended)

    freqs = np.fft.fftfreq(ffit.shape[-1])*240

    rffit = np.real(ffit)

    cffit = np.imag(ffit)

    power = np.sqrt(rffit*rffit+cffit*cffit)

    ax4[seq, 2].plot(freqs, power, color = colors[seq])

    ax4[seq, 3].plot(freqs[0:40], power[0:40], color = colors[seq])