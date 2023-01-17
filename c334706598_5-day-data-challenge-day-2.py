# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/cereal.csv")

print(data.columns)

sodium = data["sodium"]

plt.hist(sodium, bins=5,edgecolor="black" )

plt.title("Sodium in cereals")

plt.xlabel("Sodium in mg")

plt.ylabel("Count")
sodium.hist(column= "Sodium in mg", figsize = (12,12))