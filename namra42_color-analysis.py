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
train = pd.read_csv("../input/train.csv")
print(train.head())
is_dog = train.AnimalType == 'Dog'
print("Distinct Colors in Dogs : " , (train[is_dog].Color.value_counts() > 0).sum())
print((train[is_dog].Color.value_counts() > 100).sum())
print(train[is_dog].Color.value_counts()[train[is_dog].Color.value_counts() > 100])
colors = train[~is_dog].Color.unique()
print(colors)
color_set = set()
for color in colors:
    color_set.update(color.split("/"))
print(color_set)
print(len(color_set))
