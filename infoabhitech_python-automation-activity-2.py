# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings("ignore")



# Any results you write to the current directory are saved as output.
import re
df = pd.read_csv("../input/datafile.csv")
# Dataframe with various format of email and phone no.

df
df['phone#'] = df['phone#'].astype(str) 

df['email'] = df['email'].astype(str)
# Function to validate phone

def phone_fun(x):

    pattern = "(\d{3})[/-](\d{3})[/-](\d{4})"

    z = re.match(pattern, x)

    if z:

        return 1

    else:

        return 0
df['phone#_valid'] = df['phone#'].apply(phone_fun)
df
# Function to validate email

def email_fun(x):

    pattern = "(\w+)[/@](\w+)[/.](\w+)"

    z = re.match(pattern, x)

    if z:

        return 1

    else:

        return 0
df['email_valid'] = df['email'].apply(email_fun)
df
df = df[(df['phone#_valid']==1) & (df['email_valid']==1)]
# Valid email amd phone are flagged with value 1

df
import time

timestr = time.strftime("%Y%m%d_%H%M%S")

file = "validateduser_" + timestr + ".txt"
df.to_csv(file, columns = df.columns, index=False)