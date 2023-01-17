# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

#from skimage.io import imread, imshow

from scipy import misc

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/train"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from glob import glob

total_cervix_images = []



for path in glob("../input/train/*"):

    cervix_images = sorted(glob(path+"/*"))

    total_cervix_images += cervix_images

    

total_cervix_images = pd.DataFrame({'imagepath':total_cervix_images})

total_cervix_images['filetype'] = total_cervix_images.apply(lambda row: row.imagepath.split(".")[-1], axis=1)

total_cervix_images['type'] = total_cervix_images.apply(lambda row: row.imagepath.split("/")[-2], axis=1)



total_cervix_images.head()
type_aggregation = total_cervix_images.groupby(['type','filetype']).agg('count')

type_aggregation_p = type_aggregation.apply(lambda row: 1.0*row['imagepath']/total_cervix_images.shape[0], axis=1)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,8))

type_aggregation.plot.barh(ax=axes[0])

axes[0].set_xlabel("image count")

type_aggregation_p.plot.barh(ax=axes[1])

axes[1].set_xlabel("training size fraction")
fig = plt.figure(figsize= (12,8))

i = 1



for t in sorted(total_cervix_images['type'].unique()):

    ax = fig.add_subplot(1,3,i)    

    i += 1

    f = total_cervix_images[total_cervix_images['type']==t]['imagepath'].values[0]

    plt.imshow(plt.imread(f))

    plt.title("Sample cervix {}".format(t))
from collections import defaultdict

images = defaultdict(list)



for t in total_cervix_images['type'].unique():

    sample_counter = 0

    for _,row in total_cervix_images[total_cervix_images['type']==t].iterrows():

        try:

            print(row.imagepath.values[0])

            image = misc.imread(row.imagepath)

            images[t].append(img)

            sample_counter += 1

        except:

            print("imread failed for image {}".format(row.imagepath))

            print("sample counter = {}".format(sample_counter))

        if sample_counter > 35:

            break
dfs = []

for t in all_cervix_images['type'].unique():

    t_ = pd.DataFrame(

        {

            'nrows': list(map(lambda i: i.shape[0], images[t])),

            'ncols': list(map(lambda i: i.shape[1], images[t])),

            'nchans': list(map(lambda i: i.shape[2], images[t])),

            'type': t

        }

    )

    dfs.append(t_)



shapes_df = pd.concat(dfs, axis=0)

shapes_df_grouped = shapes_df.groupby(by=['nchans', 'ncols', 'nrows', 'type']).size().reset_index().sort_values(['type', 0], ascending=False)

shapes_df_grouped