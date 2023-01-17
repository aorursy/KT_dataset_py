# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/camera_dataset.csv")

df.describe()
Max_resolution = df['Max resolution']

plt.hist(Max_resolution, bins=20, edgecolor="black")

plt.title("Cameras Maximum Resolutions")

plt.xlabel("Resolution")

plt.ylabel("Count")

plt.grid("true")
Max_resolution.plot.hist(bins=50, figsize=(12,10) )
sns.distplot(Max_resolution, bins=50).set_title("Cameras Maximum Resolutions")
