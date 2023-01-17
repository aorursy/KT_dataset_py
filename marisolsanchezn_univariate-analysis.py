# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
datos = pd.read_csv('/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv')
datos.head()
datos.dtypes
datos.describe()
datos.count()
x = datos.price

plt.subplot(221)
plt.hist(x)
plt.title('Absoluta')

plt.subplot(222)
plt.hist(x, density = True)
plt.title('Relativa')

plt.subplot(223)
plt.hist(x,cumulative = True)
plt.title('Absoluta acumulada')

plt.subplot(224)
plt.hist(x, density = True, cumulative = True)
plt.title('Relativa acumulada')