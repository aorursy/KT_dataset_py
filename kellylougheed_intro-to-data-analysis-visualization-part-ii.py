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
# DataFrame of names nationally and by state

NationalNames = pd.read_csv("../input/us-baby-names/NationalNames.csv")
# See the top of the names DataFrame

# Great way to see all your column headers and know what data you're working with



NationalNames.head()
# Grab the "Name" COLUMN

NationalNames["Name"]
# Grab all the ROWS where there is a certain name and a certain gender

name = NationalNames.loc[NationalNames['Name'] == "Lena"][NationalNames['Gender'] == 'F']

print(name)
# Create a data visualization with code from Python Graph Gallery

# Original code: https://python-graph-gallery.com/240-basic-area-chart/



# library

import matplotlib.pyplot as plt

 

# Create data

x=name["Year"]

y=name["Count"]

 

# Area plot

plt.fill_between(x, y)

plt.show()