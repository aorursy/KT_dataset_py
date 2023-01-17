import pandas as pd

import numpy as np

import tifffile as tiff



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline
df = pd.read_csv("../input/train.csv")

categories = list(np.unique(' '.join(df.tags.values).split(' ')))

for category in categories:

    df[category] = df['tags'].apply(lambda x: 1 if category in x.split(' ') else 0)

#df.drop('tags', axis = 1, inplace = True)



print('Of {} rows in the dataframe, there are {} unique combinations of the {} tags'

      .format(len(df), len(df.groupby(categories).count()), len(categories)))
tagfreq = pd.DataFrame(df.sum(axis = 0)[2:]).sort_values(by = 0, ascending = False)



plt.figure(figsize=(12, 5))

plt.bar(range(len(tagfreq)), tagfreq.values)

plt.xticks(range(len(tagfreq)), tagfreq.index, rotation='80')

plt.show()
np.random.seed(143)

subset = df.sample(9).reset_index(drop = True)

plt.figure(figsize=(12,12))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.imshow(mpimg.imread('../input/train-jpg/{}.jpg'.format(subset['image_name'][i])))

    plt.annotate('\n'.join(str(subset['tags'][i]).split(' ')), xy=(5,240), color = 'white')

plt.show()
plt.figure(figsize=(12,27))

for i in range(9):

    im = tiff.imread('../input/train-tif/{}.tif'.format(subset['image_name'][i]))

    for j in range(4):

        plt.subplot(9, 4, (i*4)+j+1)

        plt.imshow(im[:,:,j])

        if j == 0:

            plt.annotate('\n'.join(str(subset['tags'][i]).split(' ')), xy=(5,240), color = 'white')

            plt.title(subset['image_name'][i])

plt.show()