!pip install imagededup
import matplotlib.pylab as plt

import imagededup

from imagededup.methods import PHash

from imagededup.utils import plot_duplicates
image_dir='../input/pku-autonomous-driving/test_images/'
phasher = PHash()

duplicates = phasher.find_duplicates(image_dir=image_dir, scores=True, max_distance_threshold=3)
{y: duplicates[y] for y in [x for x in duplicates if duplicates[x] != []][:15]}
print('There are', len([x for x in duplicates if duplicates[x] != []]), 'images with similar images over', len(duplicates), 'images.')
plt.figure(figsize=(20,20))

plot_duplicates(image_dir=image_dir, duplicate_map=duplicates, filename='ID_5bf531cf3.jpg')
plt.figure(figsize=(20,20))

plot_duplicates(image_dir=image_dir, duplicate_map=duplicates, filename='ID_ca20646c5.jpg')