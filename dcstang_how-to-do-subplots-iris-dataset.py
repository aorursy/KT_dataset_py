# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt # main visualization library

import seaborn as sns # sits ontop of matplotlib

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
iris = pd.read_csv('../input/iris-data/Iris.csv') # load in the data
iris.head(10) # show first 10 rows of the data
iris.columns
#create a figure "fig" with axis "ax1" with 3x2 configuration

fig, ax1 = plt.subplots(3,2, sharex='col', figsize=(22,18), gridspec_kw={'hspace': 0, 'wspace': 0.1}) 





# 1st plot

sns.set_style("whitegrid");

sns.scatterplot(data=iris, x="SepalLengthCm", y="SepalWidthCm", hue="Species", ax=ax1[0, 0], legend='brief') 



# 2nd plot

sns.scatterplot(data=iris, x="SepalWidthCm", y="SepalLengthCm", hue="Species", ax=ax1[0, 1], legend='brief') 



# 3rd plot

sns.scatterplot(data=iris, x="SepalLengthCm", y="PetalLengthCm", hue="Species", ax=ax1[1, 0], legend='brief') 



# 4th plot

sns.scatterplot(data=iris, x="SepalWidthCm", y="PetalLengthCm", hue="Species", ax=ax1[1, 1], legend='brief') 



# 5th

sns.scatterplot(data=iris, x="SepalLengthCm", y="PetalWidthCm", hue="Species", ax=ax1[2, 0], legend='brief') 



# 6th

sns.scatterplot(data=iris, x="SepalWidthCm", y="PetalWidthCm", hue="Species", ax=ax1[2, 1], legend='brief') 



fig.savefig("/kaggle/working/output.png")

sns.set_style("whitegrid");

sns.pairplot(iris, hue="Species", size=3);

plt.show()