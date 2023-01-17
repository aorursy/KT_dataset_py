# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns
nyairbnb_df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
nyairbnb_df.head()
nyairbnb_df.tail()
nyairbnb_df.count()
nyairbnb_df.shape
nyairbnb_df.info()
nyairbnb_df.describe()
#Print unique values of neighbourhood_group column

nyairbnb_df.neighbourhood_group.unique()
#Print unique values of neighbourhood column

nyairbnb_df.neighbourhood.unique()
nyairbnb_df.room_type.unique()
sns.set(style="darkgrid")

sns.catplot(x="neighbourhood_group", kind="count", data=nyairbnb_df)