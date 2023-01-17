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
crisis_data = pd.read_csv('../input/seattle-crisis-data/crisis-data.csv')
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
crisis_data.head()
# Set Date Format 
crisis_data['Reported Date'] = pd.to_datetime(crisis_data['Reported Date'], format='%Y-%m-%d')
# Plotting Year and Month Reported Date 
crisis_data[(crisis_data['Reported Date'].dt.year == pd.datetime.now().year) & 
           (crisis_data['Reported Date'].dt.month == pd.datetime.now().month)]['Reported Date'].value_counts().plot.line()
plt.show()
# BarPlot
crisis_data[(crisis_data['Reported Date'].dt.year == pd.datetime.now().year) & 
           (crisis_data['Reported Date'].dt.month == pd.datetime.now().month)]['Reported Date'].value_counts().plot.bar()
plt.show()