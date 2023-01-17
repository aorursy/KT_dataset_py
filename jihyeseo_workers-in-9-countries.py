# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/HR_Engagement_Sat_Sales_UpdatedV4.0.csv')
df.head()
df.dtypes
df.Role.value_counts().sort_index().plot.bar()
df.GEO.value_counts().plot.bar()
df.average_montly_hours.hist()
df.LinkedIn_Hits.hist() 
df['Sensor_Heartbeat(Average/Min)'].hist()      
