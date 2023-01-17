# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
% cd ../input
from IPython.display import Image as disp_img

disp_img('datascience.png', width=100, height=100)
import numpy as np

from PIL import Image

import scipy.misc

import scipy.cluster
img = Image.open('datascience.png')

img.resize((50,50))

data = np.array(img)

data = data.reshape(np.prod(data.shape[:2]), 3)
import matplotlib.pyplot as plt

hist = np.histogram(data)

plt.hist(hist, )

plt.title('Color histogram of the default image')

plt.show()
codes, dist = scipy.cluster.vq.kmeans(data.astype('float'), 4)
print('cluster centers:\n', codes)
vecs, dist = scipy.cluster.vq.vq(data, codes) 

counts, bins = np.histogram(vecs, len(codes))

indexes = np.argsort(counts)
frequent_colors = [codes[idx] for idx in indexes]
import matplotlib.pyplot as plt

import matplotlib.patches as patches



fig = plt.figure()

ax = fig.add_subplot(111)

for idx, rgb in enumerate(frequent_colors):

    hexname = '#%02x%02x%02x' % tuple(rgb.astype('int'))

    x = 0.03 + 0.23*idx

    p = patches.Rectangle(

        (x, 0.1), 0.2, 0.3,

        facecolor=hexname

    )

    ax.add_patch(p)

fig.show()