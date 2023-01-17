# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

camera_df = pd.read_csv('../input/camera_dataset.csv')

# Any results you write to the current directory are saved as output.
camera_df.describe()
# Import Mat Plot lib
from matplotlib import pyplot as plt

# make an histogram with pyplot
plt.hist(camera_df["Max resolution"])
plt.title("Max Resolution distribution")
plt.xlabel("Max resolution(pixels)")
plt.ylabel("count")
# Pandas has an histogram function too 
camera_df.hist(column="Max resolution")