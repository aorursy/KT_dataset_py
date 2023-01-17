# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import pandas as pd

import csv

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline



#read file

data = pd.read_csv("../input/2017-F500-diversity-data.csv" ,error_bad_lines=False, encoding='utf-8', engine='c')



cols = [col for col in data.columns if 'TOTAL' not in col]



totalCols = [col for col in data.columns if '11' in col]



data[totalCols] = data[totalCols].apply(pd.to_numeric, errors="coerce")



data[totalCols] = data[totalCols].dropna(how='all', axis=0)



data = data.dropna(how='all', axis=0)



for col in totalCols:

    data.plot(x='name', y=col, kind="bar", width = .8,)
