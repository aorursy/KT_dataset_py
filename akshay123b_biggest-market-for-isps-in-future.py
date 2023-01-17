# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import locale
from locale import atof
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data= pd.read_csv("../input/list-of-countries-by-number-of-internet-users/List of Countries by number of Internet Users - Sheet1.csv")
data.describe()


data.info()
data.head(10)
data['Population']=data['Population'].str.replace(',', '').astype(int)
data['Internet Users']=data['Internet Users'].str.replace(',', '').astype(int)
data.info()
data['Users Without Internet']= data['Population'] - data['Internet Users']
data.sort_values("Percentage", axis = 0, ascending = True, 
                 inplace = True, na_position ='last') 

data.head(10)

data.sort_values("Users Without Internet", axis = 0, ascending = False, 
                 inplace = True, na_position ='last') 

data.head(10)
bar_chart= data.head(10).plot.bar(x='Country or Area',y='Users Without Internet',subplots=True)

