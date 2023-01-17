# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

#import scipy

import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
filenames=['uber-raw-data-apr14.csv',

#'uber-raw-data-jun14.csv',

#'uber-raw-data-jul14.csv',

'uber-raw-data-aug14.csv']



#uber-raw-data-janjune-15.csv

#uber-raw-data-may14.csv

#uber-raw-data-sep14.csv
consoliData = None

for filename in filenames:

    df=pd.read_csv(r'../input/'+filename)

    df.drop('Base',axis=1,inplace=True)

    if consoliData is None:

        consoliData=df

    else:

        consoliData=pd.merge(consoliData,df,how='outer')

        

print(consoliData.head())





#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
consoliData.head()
# from http://matplotlib.org/examples/pylab_examples/scatter_hist.html

x = consoliData.Lat.values

y = consoliData.Lon.values



nullfmt = NullFormatter()         # no labels



# definitions for the axes

left, width = 0.1, 0.65

bottom, height = 0.1, 0.65

bottom_h = left_h = left + width + 0.02



rect_scatter = [left, bottom, width, height]

rect_histx = [left, bottom_h, width, 0.2]

rect_histy = [left_h, bottom, 0.2, height]



# start with a rectangular Figure

plt.figure(1, figsize=(8, 8))



axScatter = plt.axes(rect_scatter)

axHistx = plt.axes(rect_histx)

axHisty = plt.axes(rect_histy)



# no labels

axHistx.xaxis.set_major_formatter(nullfmt)

axHisty.yaxis.set_major_formatter(nullfmt)



# the scatter plot:

axScatter.scatter(x, y)
# now determine nice limits by hand:

binwidth = 1

xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])

lim = (int(xymax/binwidth) + 1) * binwidth



axScatter.set_xlim((-lim, lim))

axScatter.set_ylim((-lim, lim))



bins = np.arange(-lim, lim + binwidth, binwidth)

axHistx.hist(x, bins=bins)

axHisty.hist(y, bins=bins, orientation='horizontal')



axHistx.set_xlim(axScatter.get_xlim())

axHisty.set_ylim(axScatter.get_ylim())



plt.show()