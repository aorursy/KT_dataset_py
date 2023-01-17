# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
school_explorer = pd.read_csv('../input/2016 School Explorer.csv')
school_explorer
pd.pivot_table(school_explorer,index=["Community School?"],values=["Economic Need Index"])
# Replacing na values with 0 
school_explorer['School Income Estimate'] = school_explorer['School Income Estimate'].str.replace(',', '')
school_explorer['School Income Estimate'] = school_explorer['School Income Estimate'].str.replace('$', '')
school_explorer['School Income Estimate'] = school_explorer['School Income Estimate'].str.replace(' ', '')
school_explorer['School Income Estimate'] = school_explorer['School Income Estimate'].astype(float)


school_explorer['School Income Estimate'] = school_explorer['School Income Estimate'].fillna(school_explorer['School Income Estimate'].mean())
school_explorer['Economic Need Index'] = school_explorer['Economic Need Index'].fillna(school_explorer['Economic Need Index'].mean())
# pd.pivot_table(school_explorer,index=["Community School?"])
school_explorer.head()
import matplotlib.pyplot as plt
import matplotlib
fig = plt.figure(figsize=(15,15))
x= school_explorer["Student Attendance Rate"]
y= school_explorer["Collaborative Teachers %"]
matplotlib.pyplot.scatter(x,y)
plt.xlabel('Student Attendance Rate')
plt.ylabel('Collaborative Teachers %')
matplotlib.pyplot.show()
city_economic_need = pd.pivot_table(school_explorer,index=["City"],values=["Economic Need Index"])
city_economic_need.sort_values(by=['Economic Need Index'],ascending=False)
