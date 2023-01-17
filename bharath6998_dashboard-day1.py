# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
%matplotlib inline
# Any results you write to the current directory are saved as output.
collisions_data = pd.read_csv('../input/nypd-motor-vehicle-collisions.csv')
collisions_data.head()
collisions_data.info()
#percentage of accidents in ny for each borough
collisions_data.BOROUGH.value_counts().plot(kind = 'pie',autopct = '%.2f')
#bar graph of accidents per borough
collisions_data.BOROUGH.value_counts().plot(kind = 'bar')
pedestrians_data = collisions_data[collisions_data['NUMBER OF PEDESTRIANS KILLED']>0]
pedestrians_data['BOROUGH'].value_counts().plot(kind = 'bar')
