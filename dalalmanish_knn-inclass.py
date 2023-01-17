# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

X=train.drop('Id',axis=1)

Y=test.drop('Id',axis=1)

from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X)

distances, indices = nbrs.kneighbors(Y)
indices=indices+1

sub=pd.read_csv("../input/sample_submission.csv")

sub.head()
sub['train_image_id']=indices
sub.head()
sub.to_csv("knn_inClass.csv",index=False)