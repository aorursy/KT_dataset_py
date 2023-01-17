# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt

%matplotlib inline

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train["id_shift"] = train['ID'].shift(1) - train['ID']

test["id_shift"] = test['ID'].shift(1) - test['ID']
# It seems mercedes use stratified sampling to split train and test

# I add this variable to my model and it has slight improvement to my local cv, but a decrese to PLB

# I think "id" have the most important information to distinguish the big "Y" and small "Y", but there need more efforts to find how to use it correctly

f, ax = plt.subplots(nrows=1, ncols=2, figsize=[10,4])

ax[0].plot(range(train.shape[0]), train.id_shift, '.')

ax[1].plot(range(test.shape[0]), test.id_shift, '.')