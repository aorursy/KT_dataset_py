# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/nypd-motor-vehicle-collisions.csv")
data.head(3)
data['BOROUGH'].value_counts().plot.bar()
data['ON STREET NAME'].value_counts().head(10).plot.bar()
killdata_person = data['NUMBER OF PERSONS KILLED'] > 0
killdata_pedestrian = data['NUMBER OF PEDESTRIANS KILLED'] > 0

killdata_cyclist = data['NUMBER OF CYCLIST KILLED'] > 0
killdata_motorist = data['NUMBER OF MOTORIST KILLED'] > 0


data[killdata_pedestrian]['ON STREET NAME'].value_counts().head(10).plot.bar()
