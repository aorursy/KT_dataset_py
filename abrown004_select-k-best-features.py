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
# Import Data

features = pd.read_csv("../input/properties_2016.csv", nrows=100000)

labels = pd.read_csv("../input/train_2016.csv", nrows=100000)

labels.head()



# Join Data

df = pd.DataFrame.merge(features,labels,on="parcelid")

df.head()
# Convert to Numpy Array

features = df.as_matrix(columns=["bathroomcnt","bedroomcnt",

                                 "calculatedfinishedsquarefeet",

                                 "fireplacecnt",

                                 "garagecarcnt",

                                 "poolcnt",

                                 "unitcnt",

                                 "numberofstories",

                                 #"hashottuborspa",

                                 "structuretaxvaluedollarcnt",

                                 "landtaxvaluedollarcnt"])

features = np.nan_to_num(features)

labels = df.as_matrix(columns=["logerror"])
# Find K  best features

from sklearn.feature_selection import SelectKBest

kbest = SelectKBest(k=3)

kbest = kbest.fit_transform(features,labels)
print(kbest)