# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plotting

import seaborn as sns #for beatiful visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
fire_nrt_m6 = pd.read_csv("../input/australian-bush-fire-satellite-data-nasa/fire_nrt_M6_101673.csv")

fire_archive_m6 = pd.read_csv("../input/australian-bush-fire-satellite-data-nasa/fire_archive_M6_101673.csv")

fire_nrt_v1 = pd.read_csv("../input/australian-bush-fire-satellite-data-nasa/fire_nrt_V1_101674.csv")

fire_archive_v1 = pd.read_csv("../input/australian-bush-fire-satellite-data-nasa/fire_archive_V1_101674.csv")



type(fire_nrt_v1)
fire_nrt_v1.head()
df_concat = pd.concat([fire_archive_v1,fire_nrt_v1],sort=True)

df_concat.head()
