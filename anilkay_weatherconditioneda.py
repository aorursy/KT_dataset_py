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

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/us-accidents/US_Accidents_Dec19.csv")

data.head()
set(data["Severity"])
data.columns
set(data["Weather_Condition"])
volcanic_ash=data[data["Weather_Condition"]=='Volcanic Ash']

volcanic_ash["Severity"].value_counts()
tornado=data[data["Weather_Condition"]=='Tornado']

tornado["Severity"].value_counts()
severity4=data[data["Severity"]==4]

set(severity4["Weather_Condition"])
set(data["Weather_Condition"])-set(severity4["Weather_Condition"])
median_severity_of_weathers=data.groupby("Weather_Condition")["Severity"].median()

sorted_median_severity_of_weathers=median_severity_of_weathers.sort_values(ascending=False)

sorted_median_severity_of_weathers[0:10]
median_severity_of_weathers=data.groupby("Weather_Condition")["Severity"].mean()

sorted_median_severity_of_weathers=median_severity_of_weathers.sort_values(ascending=False)

sorted_median_severity_of_weathers[0:10]
count_of_weathers=data.groupby("Weather_Condition")["Severity"].count()

sorted_count_of_weathers=count_of_weathers.sort_values(ascending=True)

sorted_count_of_weathers[0:10]
rare_conditions=sorted_count_of_weathers[sorted_count_of_weathers<1000]

rare_list=rare_conditions.index.values.tolist()
rare_list[0:10]
data_without_rare=data[(~data["Weather_Condition"].isin(rare_list))]
median_severity_of_weathers=data_without_rare.groupby("Weather_Condition")["Severity"].median()

sorted_median_severity_of_weathers=median_severity_of_weathers.sort_values(ascending=False)

sorted_median_severity_of_weathers[0:10]
mean_severity_of_weathers=data_without_rare.groupby("Weather_Condition")["Severity"].mean()

sorted_mean_severity_of_weathers=mean_severity_of_weathers.sort_values(ascending=False)

sorted_mean_severity_of_weathers[0:10]
sorted_mean_severity_of_weathers[10:20]
sorted_mean_severity_of_weathers[30:40]
only4severity=data_without_rare[data_without_rare["Severity"]==4]

only4severity_count=only4severity.groupby("Weather_Condition")["Severity"].count()

sorted_temp=only4severity_count.sort_values(ascending=False)

sorted_temp[0:10]