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
import kagglegym as kagglegym

%matplotlib inline

from matplotlib import pyplot



with pd.HDFStore("../input/train.h5", "r") as train:

    # Note that the "train" dataframe is the only dataframe in the file

    df = train.get("train")



df.info()
# brand count

len(df["id"].unique())
# timestamp count

len(df["timestamp"].unique())
# Create environment

env = kagglegym.make()



# Get first observation

observation = env.reset()



# Look at train dataframe info

observation.train.info()
# normalize

def normalize(arr):

    arr_copy = np.copy(arr)

    arr_copy = (arr_copy - arr_copy.mean()) / arr_copy.std()



    return arr_copy
# plot specified brandId`s feature`s

# @note feature: 2 ~ 109

def plotFeature(brandId, featureFrom, featureTo, ax):

    brand = observation.train.query("id == " + brandId)

    brand.sort_values("timestamp")

    x = brand["timestamp"]

    for i in range(featureFrom, featureTo):

        y_normalized = normalize(brand.iloc[:,i])

        ax.plot(x, y_normalized)



fig = pyplot.figure()



ax1 = fig.add_subplot(2,1,1)

ax2 = fig.add_subplot(2,1,2)



fig2 = pyplot.figure()

ax3 = fig2.add_subplot(2,1,1)

ax4 = fig2.add_subplot(2,1,2)



# plot feature

plotFeature("11", 2, 5, ax1)

plotFeature("11", 5, 10, ax2)

plotFeature("11", 10, 15, ax3)

plotFeature("11", 15, 20, ax4)
# overview some brand 

brand = observation.train.query("id == 11")

brand.sort_values("timestamp")



# plot y.  we predict target

x = brand["timestamp"]

y = brand["y"]

pyplot.plot(x, y)

pyplot.plot(x, y.cumsum())
target_brand = observation.train.query("id == 16")

target_brand.sort_values("timestamp")



target_brand[["timestamp","y"]]



# this brand timestamp start by 68, end by 905

# some feature short
y = target_brand[["timestamp","technical_40", "technical_44"]]



y.iloc[160:190,:]



# some feature short