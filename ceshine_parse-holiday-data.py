# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
events = pd.read_csv("../input/holidays_events.csv")

print("# events:", events.shape[0])

events.sample(10)
events[events.type=="Work Day"]
events[events.type=="Bridge"]
print("# additional holidays:", events[events.type=="Additional"].shape[0])

events[events.type=="Additional"].tail(10)
print("# events:", events[events.type=="Event"].shape[0])

events[events.type=="Event"].tail(10)
national_holidays = events[

    (events.locale=="National") & (

        (events.type=="Additional") |  

        (events.type=="Bridge") | 

        (events.type=="Holiday")

    ) & ~events.transferred]

print("# National Holiday:", national_holidays.shape[0])

national_holidays.tail(10)
national_holidays[national_holidays["date"].duplicated(False)]
regional_holidays = events[

    (events.locale=="Regional") & (

        (events.type=="Additional") |  

        (events.type=="Bridge") | 

        (events.type=="Holiday")

    ) & ~events.transferred]

print("# Regional Holiday:", regional_holidays.shape[0])

regional_holidays.tail(10)
local_holidays = events[

    (events.locale=="Local") & (

        (events.type=="Additional") |  

        (events.type=="Bridge") | 

        (events.type=="Holiday")

    ) & ~events.transferred]

print("# Local Holiday:", local_holidays.shape[0])

local_holidays.tail(10)