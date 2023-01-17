# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



#get the data from csv

sightings = pd.read_csv("../input/scrubbed.csv")



#remove NaN's

new_sightings = sightings[pd.isnull(sightings["shape"]) == False]



#bucket names

objects = new_sightings["shape"].unique()



#x-axis - count buckets

x_pos = np.arange(len(objects))



#plot the summations of each bucket

plt.bar(x_pos, new_sightings["shape"].value_counts(), align='center', alpha=0.5)



#labels and title

plt.xticks(x_pos, objects)

plt.ylabel('Sightings')

plt.xlabel('Shapes')

plt.title('UFO Sightings by Shape')

plt.xticks(rotation=90)

plt.show()




