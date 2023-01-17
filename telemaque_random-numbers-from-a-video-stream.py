from math import modf, log10

import numpy as np

from scipy.stats import wasserstein_distance

import skvideo.io

from skimage.transform import resize

import custom_plots  



infile = "./data/clip_2_min.mp4"

#infile = "./data/downtown.mp4"

videodata = skvideo.io.vread(infile)

print(videodata.shape, videodata.dtype) 

N_frames, rows, cols, channels = videodata.shape

rr, cc = (rows//2, cols//2)

FIRST_FRAME = True

wass_list = []

rand_list = []

#for iframe in range(N_frames):

for iframe in range(N_frames):

    imgnumb = str(iframe).zfill(4)

    Img = videodata[iframe,:,:,:]



    # reduce the size of the image to increase compute speed

    Img = resize(Img, (rr,cc))

    

    # uncomment to display each frame

    #custom_plots.show_img(Img, imgnumb, pause=False)



    if not FIRST_FRAME:

        # Compute the first Wasserstein distance between two 1D distributions.

        u_img = Img.reshape(rr*cc*channels) # distribution 1

        v_img = Img_prev.reshape(rr*cc*channels) # distribution 2

        wd = wasserstein_distance(u_img, v_img)

        # generate a random float in the half-open interval [0.0, 1.0) based on wd

        rnd = modf(log10(1/wd))[0] if wd > 0 else 0

        if rnd > 0:

            rand_list.append(rnd)

            wass_list.append(wd)

            print(imgnumb, wd, rnd)



    Img_prev = Img

    FIRST_FRAME = False





# Histogram - are these numbers uniformly distributed? 

import matplotlib.pyplot as plt

num_bins = 100

fig, ax = plt.subplots(figsize=(6.5,5))

x = np.array(rand_list)

n, bins, patches = ax.hist(x, num_bins, color='darkgreen')

ax.set_xlabel('number')

ax.set_ylabel('# of instances')

ax.set_title('RNG distribution', fontname='Hack')

#plt.show()

plt.savefig('RNG distribution', bbox_inches='tight')





#Wasserstein Distance vs. Frame Number

x = np.arange(len(wass_list))

fig, ax = plt.subplots()

ax.plot(x, rand_list, '.', color='darkorange', label = 'number based on W.D.')

ax.plot(x, wass_list, '-', color='darkblue', label = 'Wasserstein distance')

ax.set_xlabel('frame number')

ax.set_ylabel('value')

legend = ax.legend(loc='upper right', shadow=True)

plt.title('Uniform-ish, Pseudo-Random Numbers from Video Feed')

#plt.show()
