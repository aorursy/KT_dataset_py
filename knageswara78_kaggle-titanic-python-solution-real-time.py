import numpy as np # Linear Algebra

import pandas as pd # Reading csv file

import os           # List directory files



print(os.listdir("../input"))
import warnings

warnings.filterwarnings('ignore') # to suppress some matplotlib deprecation warnings



import ast

import math



# Install your own package in Kernels. 

#from simplification.cutil import simplify_coords



import matplotlib.pyplot as plt

import matplotlib.style as style



%matplotlib inline

%config InlineBackend.figure_format = 'retina'
train = pd.read_csv('../input/train_simplified/roller coaster.csv',

                   index_col='key_id',

                   nrows=100)

train.info() # 100 rows with 5 columns

train.head()
train['word'] = train['word'].replace(' ', '_', regex=True) # See word column updated.

train.head()
test_raw = pd.read_csv('../input/test_raw.csv', index_col='key_id')

test_raw.head() # 112199 rows, 2 columns

test_raw.info()



first_ten_ids = test_raw.iloc[:10].index

print(first_ten_ids)

raw_images = [ast.literal_eval(lst) for lst in test_raw.loc[first_ten_ids, 'drawing'].values]

def resample(x, y, spacing=1.0):

    output = []

    n = len(x)

    px = x[0]

    py = y[0]

    cumlen = 0

    pcumlen = 0

    offset = 0

    for i in range(1, n):

        cx = x[i]

        cy = y[i]

        dx = cx - px

        dy = cy - py

        curlen = math.sqrt(dx*dx + dy*dy)

        cumlen += curlen

        while offset < cumlen:

            t = (offset - pcumlen) / curlen

            invt = 1 - t

            tx = px * invt + cx * t

            ty = py * invt + cy * t

            output.append((tx, ty))

            offset += spacing

            pcumlen = cumlen

        px = cx

        py = cy

    output.append((x[-1], y[-1]))

    return output

  

def normalize_resample_simplify(strokes, epsilon=1.0, resample_spacing=1.0):

    if len(strokes) == 0:

        raise ValueError('empty image')



    # find min and max

    amin = None

    amax = None

    for x, y, _ in strokes:

        cur_min = [np.min(x), np.min(y)]

        cur_max = [np.max(x), np.max(y)]

        amin = cur_min if amin is None else np.min([amin, cur_min], axis=0)

        amax = cur_max if amax is None else np.max([amax, cur_max], axis=0)



    # drop any drawings that are linear along one axis

    arange = np.array(amax) - np.array(amin)

    if np.min(arange) == 0:

        raise ValueError('bad range of values')



    arange = np.max(arange)

    output = []

    for x, y, _ in strokes:

        xy = np.array([x, y], dtype=float).T

        xy -= amin

        xy *= 255.

        xy /= arange

        resampled = resample(xy[:, 0], xy[:, 1], resample_spacing)

        simplified = simplify_coords(resampled, epsilon)

        xy = np.around(simplified).astype(np.uint8)

        output.append(xy.T.tolist())



    return output
# Below package need to add with your login and un comment below lines 

# by removing # at start of each below line.



#simplified_drawings = []

#for drawing in raw_images:

    #simplified_drawing = normalize_resample_simplify(drawing)

    #simplified_drawings.append(simplified_drawing)
for index, raw_drawing in enumerate(raw_images, 0):

    

    plt.figure(figsize=(6,3))

    

    for x,y,t in raw_drawing:

        plt.subplot(1,2,1)

        plt.plot(x, y, marker='.')

        plt.axis('off')



    plt.gca().invert_yaxis()

    plt.axis('equal')



    #for x,y in simplified_drawings[index]:

        #plt.subplot(1,2,2)

        #plt.plot(x, y, marker='.')

        #plt.axis('off')



    plt.gca().invert_yaxis()

    plt.axis('equal')

    plt.show()  
submission = pd.read_csv('../input/sample_submission.csv', index_col='key_id')

# Don't forget, your multi-word labels need underscores instead of spaces!

my_favorite_words = ['donut', 'roller_coaster', 'smiley_face']  

submission['word'] = " ".join(my_favorite_words)

submission.to_csv('my_favorite_words.csv')





submission.head()