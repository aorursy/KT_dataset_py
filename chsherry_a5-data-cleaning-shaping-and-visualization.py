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
data = pd.read_csv("/kaggle/input/unemployment/unemploymentdata.csv")
data.head(5)

#I'm interested in researching past unemployment in the US. This is the data that I took from the US Bureau of Labour Statistics. (Orginally I wanted to look at educational attainment,however, that data was not numeric and I did not know how to clean it.)
lines = data.plot.line(x='year', y='employed_percent')

#Here is a line graph of employment percentage throughout the year, we can see that in during the dot com crash and great recession, employment dipped. 
lines = data.plot.scatter(x='year', y='unemployed_percent')

#This is a scatterplot of unemployment rates. On average, the US unemployment rate has stayed around 6%. 

