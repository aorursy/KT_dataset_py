#Initializing the libraries

import pandas as pd
import numpy as np
import pandas_profiling as pp
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Uncomment for installation in jupyter notebook, python version should be >=3.6
#pip install -U pandas-profiling[notebook]
#jupyter nbextension enable --py widgetsnbextension

#In case you are using kaggle use below to install pandas profiling
#import sys
#!{sys.executable} -m pip install -U pandas-profiling[notebook]
#!jupyter nbextension enable --py widgetsnbextension

#Start by loading in your pandas DataFrame, e.g. by using

data = pd.read_csv("../input/nutrition-facts/menu.csv")
data.columns
# Just taking 7 columns to keep it quick and efficient
data1 = data[['Category','Item','Calories','Total Fat','Saturated Fat','Trans Fat','Cholesterol']]

from pandas_profiling import ProfileReport

#To generate the report, run:
profile = ProfileReport(data1, title="Pandas Profiling Report")
##interface type 1 Widgets - report inline (a little slow,please be patient)
profile.to_widgets()
##interface type 2  through a HTML report - gets saved in your folder
profile.to_file("mcdonalds.html")