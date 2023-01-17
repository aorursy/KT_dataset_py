# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        df = pd.read_csv(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df.head()
df.info()
# For the temperature we do not need to check the temperature at 9am or 3pm as the minmum and maximum of the day would be stored under MinTemp or MaxTemp

min_temp = min(df["MinTemp"])

max_temp = max(df["MaxTemp"])

max_rain = max(df["Rainfall"])



print(min_temp, max_temp, max_rain)
max(df["MinTemp"])

# Still a smaller value than the total max
min(df["MaxTemp"])

# Still a smaller value that the total min
print("Minimum temperature ever recorded:",min_temp)

print("Maximum temperature ever recorded:",max_temp)

print("Maximum rainfall ever recorded:",max_rain)
df.loc[df["MaxTemp"] == max_temp]
df.loc[df["MinTemp"] == min_temp]
df.loc[df["Rainfall"] == max_rain]

# In this instance you would also have record with id 9235 that would have the same amount of rainfall that would have been stored under 

# "RISK_MM" as that feature stores the rainfal for the next day.