# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



i=1

for dirname, _, filenames in os.walk('/kaggle/input/cicids2017/MachineLearningCSV'):

    for filename in filenames:

        df=pd.read_csv(os.path.join(dirname, filename))

        print(df.shape)

        if (i==1):

            cicids=df

            i=i+1

        else:

            cicids=cicids.append(df,sort=False)

    

print(cicids)



# Any results you write to the current directory are saved as output.
cicids=cicids.drop_duplicates()

print(cicids.shape)

cicids=cicids.dropna()

print(cicids.shape)
le=preprocessing.LabelEncoder()

for cols in cicids.columns:

    if(cicids[cols].dtype==object):

        cicids[cols]=le.fit_transform(cicids[cols].astype(str))
for cols in cicids.columns:

    print (cicids[cols].dtype)
cicids.to_csv("cicids_numeric.csv")