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
import pandas as pd

from bokeh.plotting import figure, output_file, show,output_notebook

output_notebook()
def make_dashboard(x, gdp_change, unemployment, title, file_name):

    output_file(file_name)

    p = figure(title=title, x_axis_label='year', y_axis_label='%')

    p.line(x.squeeze(), gdp_change.squeeze(), color="firebrick", line_width=4, legend="% GDP change")

    p.line(x.squeeze(), unemployment.squeeze(), line_width=4, legend="% unemployed")

    show(p)
links={'GDP':'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/projects/coursera_project/clean_gdp.csv',\

       'unemployment':'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/projects/coursera_project/clean_unemployment.csv'}
import pandas as pd

csv_path= links['GDP']

df=pd.read_csv(csv_path)

df.head()
csv_path = links['unemployment']

df1=pd.read_csv(csv_path)

df1.head()
a=df1[df1['unemployment']>8.5]

a
x = pd.DataFrame(df, columns=['date'])

x.head()
gdp_change = pd.DataFrame(df, columns=['change-current'])

gdp_change.head()
title = 'Unemployment according to the GDP data'
file_name = "index.html"
unemployment = pd.DataFrame(df1, columns=['unemployment'])

unemployment.head()
make_dashboard(x=x, gdp_change=gdp_change, unemployment=unemployment, title=title, file_name=file_name)