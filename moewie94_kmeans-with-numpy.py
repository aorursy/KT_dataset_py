import matplotlib.pyplot as plt

import numpy as np
np.random.seed(24)
points = np.vstack(((np.random.randn(100, 2) * 0.5 + np.array([1, 1])),

                  (np.random.randn(100, 2) * 0.5 + np.array([3, 2])),

                  (np.random.randn(100, 2) * 0.5 + np.array([1, 3]))))
plt.scatter(points[:, 0], points[:, 1]);
def initialize_centroids(points, k):

    """returns k centroids from the initial points"""

    centroids = points.copy()

    np.random.shuffle(centroids)

    return centroids[:k]
plt.scatter(points[:, 0], points[:, 1]);

centroids = initialize_centroids(points, 3)

plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100, marker='*');
def closest_centroid(points, centroids):

    """returns an array containing the index to the nearest centroid for each point"""

    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))

    return np.argmin(distances, axis=0)
point_c = closest_centroid(points, centroids)
plt.scatter(points[:, 0], points[:, 1], c=point_c);

plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100, marker='*');
def move_centroids(points, closest, centroids):

    """returns the new centroids assigned from the points closest to them"""

    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])
print('Old centroids:\n', centroids)

centroids = move_centroids(points, point_c, centroids)

print('New centroids:\n', centroids)
point_c = closest_centroid(points, centroids)

plt.scatter(points[:, 0], points[:, 1], c=point_c);

plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100, marker='*');
MOVES = 5

K = 3

centroids = initialize_centroids(points, K)

for m in range(MOVES):

    closest = closest_centroid(points, centroids)

    centroids = move_centroids(points, closest, centroids)

    plt.scatter(points[:, 0], points[:, 1], c=closest)

    plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100, marker='*')

    plt.show()

    print(centroids)