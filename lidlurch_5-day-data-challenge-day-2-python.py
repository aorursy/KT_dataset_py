#import libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

import seaborn as sns #data visualization library

print(check_output(["ls", "../input"]).decode("utf8"))

#Read in your data

df = pd.read_csv(r'../input/data.csv')
#Summarize your data

df.head() #Shows us the first 5 rows
#Further summarize your data

df.describe() #gives us the standard statistical points for each column
#visualize one of the columns

x = df['radius_mean'] #set the x axis

ax =sns.distplot(x, axlabel = 'Radius Mean', kde =False).set_title("Radius Mean Histogram")