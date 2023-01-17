import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('../input/3-Airplane_Crashes_Since_1908.txt')
df.columns.values
deaths = df['Fatalities']
df['Operator']
date_list= df['Date']
yr_list = pd.np.zeros(len(date_list))
for index,time in enumerate(date_list):
    date_object = datetime.strptime(time, '%M/%d/%Y')
    yr_list[index]=(date_object.year)
plt.plot(yr_list)
plt.plot(yr_list[:50])
plt.plot(yr_list[50:100])
plt.plot(yr_list[:100])
plt.plot(yr_list, deaths)
