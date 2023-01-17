import json

import numpy as np

from matplotlib import pyplot as plt

from PIL import Image
f = open(r'../input/shipsnet.json')

dataset = json.load(f)

f.close()

print(dataset.keys())
print(len(dataset['data']))

print(len(dataset['data'][0]))
index = 50 # Image to be reformed

pixel_vals = dataset['data'][index]

arr = np.array(pixel_vals).astype('uint8')

im = arr.reshape((3, 6400)).T.reshape((80,80,3))

print(im.shape)
im = Image.fromarray(im)

im.save('80x80.png')
plt.imshow(im)

plt.show()