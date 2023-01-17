""" Load an image """

from PIL import Image

image = Image.open('../input/images/beer.jpg')
""" Extract pixel data to array """

import numpy as np

image_data = np.array(image)
width = image.width
height = image.height
size = width*height

pixels = image_data.flatten().reshape([size, 3])
""" Helper functions """

def are_considerably_different(centroids_old, centroids_new, epsilon):
    delta = np.linalg.norm(centroids_new-centroids_old, axis=1)
    print(f'Centroid deltas: {delta}')
    print(f'Centroid max delta: {np.max(delta)}')
    return np.max(delta)>epsilon
    
""" Perform 'vectorized' k-means clustering """

from random import randint, seed
#seed(7)

# initialization
K = 5
epsilon = 0.05

centroids_old = np.zeros([K,3])
centroids = np.zeros([K,3])
for k in range(K):
    centroids[k] = pixels[randint(0,size-1)]

iteration_count = 0
max_iteration_count = 20

# iteration
while are_considerably_different(centroids_old, centroids, epsilon):

    print(f'Iteration number: {iteration_count}')
    iteration_count += 1
    if(iteration_count > max_iteration_count):
        print('Max iteration number reached')
        break

    distances = np.empty([size, K])

    for k in range(K):
        distances[:,k] = np.linalg.norm(pixels - centroids[k], axis=1)

    cluster_indices = np.argmin(distances, axis=1)

    # Update the centroids
    centroids_old = np.copy(centroids)

    for k in range(K):
        indicator = np.array([int(ci == k) for ci in cluster_indices])
        pixels_sum = np.matmul(indicator, pixels)
        pixels_count = np.sum(indicator)
        centroids[k] = pixels_sum / pixels_count

print('done')
""" Process the image """

processed_pixels = np.copy(pixels)

for i in range(size):
    for k in range(K):
        if(cluster_indices[i] == k):
            processed_pixels[i] = centroids[k]
            break

""" Show the result """

processed_image = Image.fromarray(processed_pixels.reshape([height, width, 3]))
processed_image.show()
