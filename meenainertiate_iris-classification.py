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
from sklearn.naive_bayes import GaussianNB
iris_df = pd.read_csv('../input/Iris.csv')
iris_df.head(10)
iris_df.count()
from sklearn.model_selection import train_test_split



train_df, test_df = train_test_split(iris_df, test_size = 0.1)
train_df.count()
test_df.count()
a = train_df.SepalLengthCm

b = train_df.SepalWidthCm

c = train_df.PetalLengthCm

d = train_df.PetalWidthCm

y = train_df.Species
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier

model = GaussianNB()

train_df.Species.unique()
test_df.Species.unique()