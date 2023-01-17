# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plotting stuff
import seaborn as sns # for maps

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train = pd.DataFrame.from_csv("../input/train.csv")

columns = train.columns

Category = train['Category']

fig,ax = plt.subplots()
ax = Category.value_counts()[0:10].plot(kind = 'bar',title = 'top ten most popular crimes')
ax.set_xticklabels(Category.value_counts()[0:10].index, rotation=15)

plt.savefig('bar plot')

#I start by looking at where Larceny/Theft occured, because apparently its the most prevalent crime in SF

mapdata = np.loadtxt("../input/sf_map_copyright_openstreetmap_contributors.txt")
asp = mapdata.shape[0] * 1.0 / mapdata.shape[1]
plt.figure(figsize=(20,20*asp))
ax2 = sns.kdeplot(train.Xok, train.Yok, clip=clipsize, aspect=1/asp)
ax2.imshow(mapdata, cmap=plt.get_cmap('gray'), extent=(-122.5247, -122.3366, 37.699, 37.8299), 
              aspect=asp)
plt.savefig('map')
