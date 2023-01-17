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
import matplotlib.pyplot as plt
import datetime as dt
# Loading the data into a pandas dataframe
df = pd.read_csv ("../input/be_our_guest_cleaned.csv")
                  
# Want to look at first 5 rows to make sure dataframe was loaded properly 
df.head()
# Drop columns that we will not be using. Since my main focus is on the satisfaction response, I will drop "Hashed IP Address" and "Party Members"
df=df.drop(["Hashed IP Address", "Party Members"], axis=1)
#Check to make sure the column was deleted
df.head()
