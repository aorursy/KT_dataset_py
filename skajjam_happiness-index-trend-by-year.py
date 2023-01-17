# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



data2015 = pd.read_csv('../input/2015.csv')

data2016 = pd.read_csv('../input/2016.csv')

data2017 = pd.read_csv('../input/2017.csv')



test15=data2015[['Region','Happiness Score']].groupby(['Region'], as_index=False).mean()

test16=data2016[['Region','Happiness Score']].groupby(['Region'], as_index=False).mean()



matchregion17=pd.merge(data2017,data2016,left_on='Country', right_on='Country')

test17=matchregion17[['Region','Happiness.Score']].groupby(['Region'], as_index=False).mean()

test17.columns=['Region', 'Happiness Score']



test15['Year']=2015

test16['Year']=2016

test17['Year']=2017



final = pd.concat([test15, test16, test17])

#print(final)



#final.info()



#grid = sns.FacetGrid(final, row='Region', size=2.2,aspect=1.6)

sns.pointplot(x='Year', y='Happiness Score', hue ='Region', data=final)

#grid.add_legend()
