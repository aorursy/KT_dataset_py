# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


data = np.random.multivariate_normal([0,0],[[5,2],[2,2]],size=2000)

data = pd.DataFrame(data, columns=['x', 'y'])



sns.kdeplot(data)


data = np.random.multivariate_normal([0,0],[[5,2],[2,2]],size=2000)

data = pd.DataFrame(data, columns=['x', 'y'])



for col_char in 'xy':

    sns.kdeplot(data[col_char], shade = True)


data = np.random.multivariate_normal([0,0],[[5,2],[2,2]],size=2000)

data = pd.DataFrame(data, columns=['x', 'y'])



for col_char in 'xy':

    sns.kdeplot(data[col_char], shade = False)



data = np.random.multivariate_normal([0,0],[[5,2],[2,2]],size=2000)

data = pd.DataFrame(data, columns=['x', 'y'])



for col_char in 'xy':

    sns.kdeplot(data[col_char], shade = True)

    sns.distplot(data[col_char])




data =np.random.multivariate_normal([0,0],[[5,2],[2,2]],size=2000)

data = pd.DataFrame(data, columns=['x', 'y'])

with sns.axes_style('white'):

    sns.jointplot("x", "y", data, kind='kde');


data = np.random.multivariate_normal([0,0],[[5,2],[2,2]],size=2000)

data = pd.DataFrame(data, columns=['x', 'y'])

with sns.axes_style('white'):

    sns.jointplot("x", "y", data, kind='hex');