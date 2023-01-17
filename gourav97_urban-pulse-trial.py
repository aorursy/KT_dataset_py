# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
lat = pd.read_csv("../input/sanFran.csv",usecols=["latitude"])["latitude"]
lon = pd.read_csv("../input/sanFran.csv",usecols=["longitude","latitude"])
print(lat.head())
print(lon.head())

#Histogram of latitude and its occurence
g=sns.distplot(lat,kde=False)

#checkin locations in a scatter plot
sns.jointplot(x="latitude", y="longitude", data=lon);

#checkin in a geospatial way
sns.jointplot("latitude", "longitude", data=lon,kind="kde", space=0, color="g")
plt.show()


# Any results you write to the current directory are saved as output.
