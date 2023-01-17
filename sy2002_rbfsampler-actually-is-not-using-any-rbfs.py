%matplotlib inline

import matplotlib.pyplot as plt



import numpy as np

from sklearn.kernel_approximation import RBFSampler
# let's use len_x * len_y samples to draw the visualization.

len_x = 200

len_y = 200



# let's have the value space in our 2D example run from min_val to max_val

min_val = -2

max_val = +2



# * the image will be drawn by using matplotlib's imshow, which expects a 3d array/list of pixels

#   in the following form: [rows, columns, rgb-values]

# * so we generate a mesh grid that we can use to map the x, y values to colors in the

#   same order, as imshow needs it

# * since imshow draws from top/left to bottom/right, we need to make sure, that the y-axis

#   runs from max_val to min_val and in contrast the x-axis from min_val to max_val

# * the complex number syntax is just an oddity of numpy.mgrid - refer to the documentation

grid_y, grid_x = np.mgrid[max_val:min_val:complex(0, len_y), min_val:max_val:complex(0, len_x)]

grid_x = grid_x.ravel()

grid_y = grid_y.ravel()
# Gaussian Radial Basis Function

# for example as defined here https://en.wikipedia.org/wiki/Radial_basis_function

def rbf(x, l, g):

    x = np.array(x)

    l = np.array(l)

    return np.exp(-(g * np.linalg.norm(x - l)**2))
# apply the RBF to our value spectrum ranging from min_val to max_val (e.g. from -2 to +2)

# and by doing so, generate the image that we want to show

image = []



for y, x in zip(grid_y, grid_x):

    X = [x, y]       # 2D vector for RBF

    C = [0.7, 0.7]   # center of the RBF

    image.append(3 * [rbf(X, C, 1.0)]) # RBF with gamma = 1.0



# hint: the above 3 * [] statement makes sure, that R, G and B color values are equal,

# so that the result will be a grayscale from black to white
# transform the flat image data into the [rows, columns, RGB] form that the imshow routine expects

image = np.array(image).reshape(len_y, len_x, 3)
# draw the values of the feature transformation of the x, y coordinates to the RBF

plt.figure(figsize=(8, 8))

plt.imshow(image, extent=[min_val, max_val, min_val, max_val], interpolation="none")

plt.show()
# we are using just one exemplar / center by setting n_components to 1

# also note that we use the same gamma value as abve

rbfs = RBFSampler(gamma=1.0, n_components=1)
# reading the documentation shows, that the fit function does not consider the data at all, but only

# the dimensionality of the data, so we can pass some dummy numbers

rbfs.fit([[0, 0]])
# transformation function that takes into consideration, that rbf.transform returns an array with one

# element, so using "min" (or "max") extracts that; additionally and strangely enough, RBFSampler's transformation

# can also yield negative numbers

def rbf_distance(x, y):

    return np.min(rbfs.transform([[x, y]]))
# same visualization technique as above ...

image2 = []



for y, x in zip(grid_y, grid_x):

    image2.append(3 * [rbf_distance(x, y)])



image2 = np.array(image2).reshape(len_y, len_x, 3)



# ... but this time we need to make sure that the output is normalized to be between 0 and 1

# (something we would not have to do if RBFSampler actually behaved like RBFs)

image2 -= np.min(image2)  # make sure that all values are > 0

image2 /= np.max(image2)  # normalize between 0 .. 1
# draw it

plt.figure(figsize=(8, 8))

plt.imshow(image2, extent=[min_val, max_val, min_val, max_val], interpolation="none")

plt.show()
EXEMPLARS = 20

Cs_GAMMA = 3.0



# create EXEMPLARS amount of centers that fit into min_val .. max_val

Cs = []

Cs_width = max_val - min_val

for i in range(EXEMPLARS):

    Cs.append([np.random.rand()*Cs_width - Cs_width/2.0, np.random.rand()*Cs_width - Cs_width/2.0, Cs_GAMMA])
# to visualize multiple RBFs, we are adding up all distances to all centers for the current pixel

def multi_rbf(x):

    ret_val = 0.0

    x = np.array(x)

    for c in Cs:

        l = [c[0], c[1]]

        ret_val += rbf(x, l, c[2])

    return ret_val
# plot using the technique described above in detail

image3 = []



for y, x in zip(grid_y, grid_x):

    image3.append(3 * [multi_rbf([x, y])])   

    

image3 = np.array(image3).reshape(len_y, len_x, 3)

image3 /= np.max(image3) # as we are adding up all RBFs distances, we need to normalize to 0 .. 1



plt.figure(figsize=(8, 8))

plt.imshow(image3, extent=[min_val, max_val, min_val, max_val], interpolation="none")

plt.show()
rbfs2 = RBFSampler(gamma=Cs_GAMMA, n_components=EXEMPLARS)

rbfs2.fit([[0, 0]])
# similar mechanism as above: to visualize multiple RBFs, we are adding up all distances to all centers

# for the current pixel with the speciality, that we also have negative values here

def rbf_distance(x, y):

    return np.sum(rbfs2.transform([[x, y]]))
image4 = []



for y, x in zip(grid_y, grid_x):

    image4.append(3 * [rbf_distance(x, y)])
image4 = np.array(image4).reshape(len_y, len_x, 3)



# special post processing needed:

# we have potentially large negative values; by clipping them, the resulting image is "a bit less crowded"

image4 = np.clip(image4, np.min(image4)*0.4, 100)



# the usual "make all values positive and scale to 0 .. 1"

image4 -= np.min(image4)

image4 /= np.max(image4)
plt.figure(figsize=(8, 8))

plt.imshow(image4, extent=[min_val, max_val, min_val, max_val], interpolation="none")

plt.show()