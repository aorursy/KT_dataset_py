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
df = pd.read_csv("/kaggle/input/us-accidents/US_Accidents_Dec19.csv",parse_dates=["Start_Time","End_Time","Weather_Timestamp"])
highest_accidental_city = df["State"].value_counts().index[0]

us_states = df.groupby("State")

us_states.get_group(highest_accidental_city)[["State","Description"]] # the us state with max accident and description