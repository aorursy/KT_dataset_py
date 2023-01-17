# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
airbnb_filepath = "../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv"

airbnb_data = pd.read_csv(airbnb_filepath)

airbnb_data.head()
plt.figure(figsize=(16,6))

sns.barplot(x=airbnb_data["price"], y=airbnb_data["room_type"])
plt.figure(figsize=(16,6))

sns.barplot(x=airbnb_data["price"], y=airbnb_data["neighbourhood_group"])