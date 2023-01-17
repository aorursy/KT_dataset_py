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
#highest number of accidents by state and description

filename = '/kaggle/input/us-accidents/US_Accidents_May19.csv'

df=pd.read_csv(filename,encoding='ISO-8859-1')

df[['ID','Description','State']]

df.sort_values(by="ID",ascending=False)

df[['ID','Description','State']]




