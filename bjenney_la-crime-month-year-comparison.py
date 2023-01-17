# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import csv

import matplotlib.pyplot as plt

import numpy as np

from sklearn import tree

%matplotlib inline



mycsv = pd.read_csv("../input/Crimes_2012-2016.csv")



df = pd.DataFrame(mycsv)



df = df[df['CrmCd.Desc'].str.contains('TRAFFIC DR') == False]



df['Date.Rptd'] = pd.to_datetime(df['Date.Rptd'])



df = df.groupby(['Date.Rptd']).size().reset_index(name='Crime Count')



df.index = df['Date.Rptd']



df = df.resample('M').sum()



df['Month'] = map(lambda x: x.month, df.index)

df['Year'] = map(lambda x: x.year, df.index) 



df.groupby(['Month','Year']).sum().unstack().plot()