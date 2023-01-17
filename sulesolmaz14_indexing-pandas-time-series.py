# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
print(os.listdir('../input'))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/uncover/UNCOVER/harvard_global_health_institute/hospital-capacity-by-state-20-population-contracted.csv')

data.head()
time_list=["1992-03-08", "1992-04-12"]
print(type(time_list[1]))#As you can see date is string, however we want it to be datatime object.
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
data.head()
import warnings 
warnings.filterwarnings("ignore")
#In order to practice lets take head of this data and add it a time list.
date2 = data.head()
date_list = ["1992-01-10","1992-02-10", "1992-03-10", "1993-03-15", "1993-03-16"]
datetime_object=pd.to_datetime(date_list)
date2["date"] = datetime_object
#make date as index
date2 = date2.set_index("date")
date2
#Now we can select according to our date index
print(date2.loc["1993-03-16"])
print(date2.loc["1992-03-10":"1993-03-16"])
#We will use data2 that we create at previous part.
date2.resample("A").mean()
#Resample with month
date2.resample("M").mean() #There are a lot of non values because date2 does not include all months.
#We can interpolate from first value.
date2.resample("M").first().interpolate("linear")
#We can interpolate with mean()
date2.resample("M").mean().interpolate("linear")