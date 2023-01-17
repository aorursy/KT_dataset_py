# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install datatable
import datatable as dt
path = "/kaggle/input/amazon-cell-phones-reviews/"
help(dt.fread)
reviews = dt.fread(path+"20190928-reviews.csv")

reviews.head(2)
type(reviews)
items = dt.fread(path+"20190928-items.csv")

items.head(2)
reviews.names
reviews.nrows
reviews.ncols
reviews.ndims
reviews.stypes
reviews[0,0]
# The column label can be provided also 

reviews[0,"asin"]
reviews[0:10,0:4]
reviews[0:10,'asin':'verified']
from datatable import *
rating3 = reviews[f.rating == 3,:]

rating3.head(5)
avgRating = reviews[:,dt.mean(f.rating),dt.by(f.verified)]

avgRating
avgRating1 = reviews[:,dt.mean(f.rating),dt.by(f.verified,f.asin)]

avgRating1
avgRating2 = reviews[:,[dt.mean(f.rating),dt.sum(f.rating)],dt.by(f.verified,f.asin)]

avgRating2
items.key="asin"
data12 =  reviews[:,:,dt.join(items)]
data12.head(3)
sortedData = reviews[:,:,dt.sort(f.rating)]

sortedData.head(3)
sortedData = reviews[:,:,dt.sort(-f.rating)]

sortedData.head(3)
sortedData = reviews[:,:,dt.sort(-f.rating),dt.sort(f.verified)]

sortedData.head(3)