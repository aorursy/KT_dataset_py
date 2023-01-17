# Muhammad Glend Angga Dwitama

# 1806191396



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/500-person-gender-height-weight-bodymassindex/500_Person_Gender_Height_Weight_Index.csv")

print(data)

berat = data.Weight

berat.mean()

berat.median()

#modus

berat.mode()
range = berat.max() - berat.min() + 1

print(range)
#variance

berat.var()
#standard deviation

berat.std()
#percentile-15

berat.quantile(0.15)
#percentile-90

berat.quantile(0.9)
#interquantile range

iqr = berat.quantile(0.75) - berat.quantile(0.25)

print(iqr)
#reasonable lower boundary

rlb = berat.quantile(0.25) - 1.5*iqr

print(rlb)
#reasonable upper boundary

rub = berat.quantile(0.75) + 1.5*iqr

print(rub)
#outlier

data.loc[berat<rlb]
#plot

plt.boxplot(berat)

plt.show()
plt.hist(berat)
y_pos = np.arange(len(berat))



plt.bar(y_pos, berat)