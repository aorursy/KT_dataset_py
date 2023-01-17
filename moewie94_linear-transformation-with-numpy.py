# numpy for numerical representations

import numpy as np

# pyplot for plotting and image showing

from matplotlib import pyplot as plt
# Now we define a function to perform linear transformation.

def linear_transformation(src, a):

    '''

    linear transformation of a matrix is simply a mutation of PLACES in that matrix.

    

    '''

    # get shape of input images

    M, N = src.shape

    # get points as matrix of LOCATION in the images

    points = np.mgrid[0:N, 0:M].reshape((2, M*N))

    # use the dot product to transform the LOCATION of values in image

    new_points = np.linalg.inv(a).dot(points).round().astype(int)



    # get x,y as the new location on x axis and y axis

    x, y = new_points.reshape((2, M, N), order='F')

    # transform x,y to 1-D indices

    indices = x + N*y

    # put each value to its new place

    return np.take(src, indices, mode='wrap')
# create a image

# start with size 200x200

images = np.ndarray(shape=(200,200))

# for easy illustration, this image will be images with a different color in each quarter

# so we assign a different value for each quarter of the images

# first quarter

images[:100, :100] = 1

# second quarter

images[:100, 101:] = 2

# third quarter

images[101:, :100] = 3

# forth quarter

images[101:, 101:] = 4



# convert image to range 0-255 (natural representation of single channel images)

images = ((images / np.max(images)) * 255).astype(np.uint8)
# use plt.imshow() function to show image

# i choose summer colormap because i like it

plt.imshow(images, cmap='summer')
# define a transformation matrix

# this matrix will dilate the images by 1.5 in x axis

transformation_matrix = np.array([[1.5, 0],

                                  [0, 1]])
# call function to perform transformation

transformed_image = linear_transformation(images, transformation_matrix)
# show the transformed image

plt.imshow(transformed_image, cmap='summer')
# call function to perform transformation

transformed_image = linear_transformation(transformed_image, np.linalg.inv(transformation_matrix))
# show the transformed image

plt.imshow(transformed_image, cmap='summer')
transformation_matrix = np.array([[1, 0],

                                  [0, 1.5]])
transformation_matrix = np.array([[1.5, 0],

                                  [0, 1.5]])
transformation_matrix = np.array([[0.5, 0],

                                  [0, 0.5]])
transformation_matrix = np.array([[1, 1],

                                  [0, 1]])
transformation_matrix = np.array([[1, 0],

                                  [1, 1]])
transformation_matrix = np.array([[-1, 0],

                                  [0, 1]])
transformation_matrix = np.array([[1, 0],

                                  [0, -1]])
transformation_matrix = np.array([[-1, 0],

                                  [0, -1]])