import numpy as np # linear algebra

import matplotlib.pyplot as plt
# the goal is to take an image with a black background and make it transparent

# that way we can overlay it on other images

# for some reason i can't find a convenient library for this

# maybe it's too simple? wasn't for me...

# TL;DR see `def black_to_transparent` at the end if you just want to copy/paste



# first let's illustrate the problem:
# start with RGB uint8 array

foreground_img = np.array([

            [[0,0,0],[0,0,0],[0,1,0],[0,1,1],[0,1,1]],

            [[1,0,0],[0,0,0],[0,1,0],[0,0,0],[0,1,1]],

            [[1,0,0],[0,0,0],[0,1,0],[0,0,0],[0,1,1]],

            [[1,0,0],[1,1,0],[0,1,0],[0,0,0],[0,0,0]],

        ]) * 255 # (the uint8 part, all the 1s needs to be 255)

print(foreground_img.shape) # note the shape (height, width, depth)

plt.imshow(foreground_img)
# note that the shape is 4,5,3

# the black is ugly, let's replace it with a soothing blue



# make soothing background color

background_img = (np.ones(foreground_img.shape)*[0,0.3,0.3] * 255).astype(np.uint8)

plt.imshow(background_img)
# now let's overlay

fig = plt.figure()

bg = fig.add_subplot(111)

bg.imshow(background_img)

fg = fig.add_subplot(111)

fg.imshow(foreground_img)

plt.show(fig)
# seems like we lost our background, but if there was alpha on the foreground we'd see it

fig = plt.figure()

bg = fig.add_subplot(111)

bg.imshow(background_img)

fg = fig.add_subplot(111)

fg.imshow(foreground_img, alpha=0.5)

plt.show(fig)
# so how do we add an alpha channel and make all of our blacks transparent?



# first make the alpha channel:

alpha_channel = (np.ones(foreground_img.shape[0:2]) * 255).astype(np.uint8)



# then we can use dstack to append it to the 3rd dimension of the foreground_img

alpha_img = np.dstack((foreground_img, alpha_channel ))

print(alpha_img.shape)



# now our image with alpha has one more dimension for alpha but it should plot the same as the foreground image

plt.imshow(alpha_img, alpha=0.5) # alpha to show the white background behind it
# if we put 0s in the areas of the image without color in them, we'll be able to see the background

# so first an experiment to make sure we know how to set the alpha channel from alpha_img

# let's try cutting the 255 alpha's down a bit and make sure it plots correctly

alpha_img[:,:,foreground_img.shape[2]] = 180

alpha_img

plt.imshow(alpha_img)
# so now instead of a fixed 180, we need it to be 255 when one of the other channels is non-zero or 0 if all channels are zero

# alpha_img[:, :, 0:3] gets us the RGB without the A, so if we sum up the whole image by the RGB axis

# and clip it to valid rgb values, we've got our mask

mask = np.sum(alpha_img[:, :, 0:foreground_img.shape[2]], axis=2).clip(0,255)

print(mask.shape)

mask
# now let's set our alpha channel on foreground_img that we made above to the mask

alpha_img[:,:,foreground_img.shape[2]] = mask



# and overlay

fig = plt.figure()

bg = fig.add_subplot(111)

bg.imshow(background_img, alpha=1.0)

fg = fig.add_subplot(111)

fg.imshow(alpha_img)

plt.show(fig)
# copy and paste here :D

# let's distill this down into a reusable function

# this turns a (uint8) image into (uint8 + alpha channel) image where black pixels are alpha=0

def black_to_transparent(img):

    num_channels = img.shape[2]

    dimensions = img.shape[0:2]

    alpha_channel = (np.ones(dimensions) * 255).astype(np.uint8)

    alpha_img = np.dstack((img, alpha_channel ))

    mask = np.sum(alpha_img[:, :, 0:img.shape[2]], axis=2).clip(0,255)

    alpha_img[:,:,num_channels] = mask

    return alpha_img
# done :D

fig = plt.figure()

bg = fig.add_subplot(111)

bg.imshow(background_img)

fg = fig.add_subplot(111)

fg.imshow(black_to_transparent(foreground_img))

plt.show(fig)