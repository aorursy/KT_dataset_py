import json

import numpy as np

from PIL import Image

from matplotlib import pyplot as plt
f = open(r'../input/planesnet.json')

planesnet = json.load(f)

f.close()

print(planesnet.keys())
index = 100 # Row to be saved

im = np.array(planesnet['data'][index]).astype('uint8')

im = im.reshape((3, 400)).T.reshape((20,20,3))

print(im.shape)
plt.imshow(im)

plt.show()
out_im = Image.fromarray(im)

out_im.save('test.png')