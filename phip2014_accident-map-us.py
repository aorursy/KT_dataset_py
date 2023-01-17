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
# Import numpy, pandas, matpltlib.pyplot, sklearn modules and seaborn



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



plt.style.use('ggplot')
# read the survey results, slice the states of interests, and display the first 5 rows

df = pd.read_csv('/kaggle/input/us-accidents/US_Accidents_May19.csv')

# Map of accidents

sns.scatterplot(x='Start_Lng', y='Start_Lat', data=df, hue='State',s=20, legend=False)

plt.xlabel('Longitude')

plt.ylabel('Latitude)')

plt.show()