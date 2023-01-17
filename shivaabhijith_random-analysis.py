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
taxi=pd.read_csv("/kaggle/input/taxi-trajectory/train.csv")

taxi.head()
taxi_central_based=taxi.loc[taxi.CALL_TYPE=='A']

stand_based=taxi.loc[taxi.CALL_TYPE=='B']

random_street=taxi.loc[taxi.CALL_TYPE=='C']
Locations_complete=taxi.loc[taxi.MISSING_DATA == False]

Locations_missing=taxi.loc[taxi.MISSING_DATA == True]
Locations_complete['Travel_time']=(Locations_complete['POLYLINE'].str.len()-1)*15

Locations_complete
A_count=taxi_central_based.shape[0]

B_count=stand_based.shape[0]

C_count=random_street.shape[0]
import matplotlib.pyplot as plt



labels=['taxi central based','taxi stand based','taxi random street']

sizes=[A_count,B_count,C_count]

colors = ['gold', 'yellowgreen', 'lightcoral']

explode = (0.1, 0, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=120)



plt.axis('equal')

plt.show()