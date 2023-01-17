# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# ny_dataframe = pd.read_csv("../input/MN.csv")



mn_dataframe = pd.read_csv("../input/MN.csv")

bk_dataframe = pd.read_csv("../input/BK.csv")

bx_dataframe = pd.read_csv("../input/BX.csv")

qn_dataframe = pd.read_csv("../input/QN.csv")

si_dataframe = pd.read_csv("../input/SI.csv")



ny_dataframe = mn_dataframe.append(bk_dataframe)

ny_dataframe = ny_dataframe.append(bx_dataframe)

ny_dataframe = ny_dataframe.append(qn_dataframe)

ny_dataframe = ny_dataframe.append(si_dataframe)



ny_dataframe = ny_dataframe.dropna(subset=["YearBuilt"])

print(ny_dataframe['YearBuilt'].head())



valid_XCoord = ny_dataframe['XCoord'] > 0

valid_YCoord = ny_dataframe['YCoord'] > 0

valid_YearBuiltgt0 = ny_dataframe['YearBuilt'] != 0

# valid_YearBuiltValue = ny_dataframe['YearBuilt'] != 'NaN'



ny_dataframe = ny_dataframe.dropna(subset=["YearBuilt"])

ny_dataframe = ny_dataframe[np.isfinite(ny_dataframe['YearBuilt'])]

ny_dataframe = ny_dataframe.where(valid_XCoord & valid_YCoord & valid_YearBuiltgt0)

# ny_dataframe['YearBuiltSegment'] = np.floor(ny_dataframe["YearBuilt"]/10) - 170

print(ny_dataframe['YearBuilt'].head())

ny_dataframe['YearBuiltSegment'] = (2017 - ny_dataframe["YearBuilt"]) / 10

print(ny_dataframe['YearBuiltSegment'].head())



ny_dataframe.plot(kind="scatter", figsize=(25,25), alpha=0.6,

                                                     x="XCoord", y="YCoord", c="ZipCode", colorbar=True,

                  cmap=plt.get_cmap("jet"))

plt.show()