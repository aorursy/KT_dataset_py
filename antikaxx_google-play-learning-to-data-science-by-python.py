# Origin by ->  It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# This Python 3 environment comes with many helpful analytics libraries installed

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.graph_objs as go



import plotly

# connected=True means it will download the latest version of plotly javascript library.

plotly.offline.init_notebook_mode(connected=True)

import plotly.graph_objs as go



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/googleplaystore.csv')

data.info()

data.head(10)
# Kac tane category urunu oldugunu bakiliyor.

number_of_apps_in_category = data['Category'].value_counts().sort_values(ascending=True)

print(number_of_apps_in_category)







df = [go.Pie(

        labels = number_of_apps_in_category.index,

        values = number_of_apps_in_category.values,

        hoverinfo = 'label+value'

    

)]



plotly.offline.iplot(df, filename='active_category')
# Histogram

# bins = number of bar in figure

number_of_apps_in_category.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()



data.corr()

#data copy to data_new datafreams 

data_new = data.copy()

data_new.head()



#and I need to Installs cloum return to integer.

# burayi çalistiramadim. I'm fail 

#data_new['Installs']= data_new['Installs'].astype('float')



# print(string.strip(' xoxoe')) bu methodu kullanmam lazim. istemdigim karakteri icine bir bosluk biraki yazdigim da onu kaldiriyor.

# burada 10.000+ sonundaki "+" karakterini kaldiriyoruz sonra object ten sayisal veriye cevirebiliyoruz :) 

data_new['Installs'] = data_new['Installs'].apply(lambda x : x.strip(' ,').capitalize()) 

data_new.head()



#Installs birimini float yapalim.

#data_new['Installs']= data_new['Installs'].astype('string')



#Installs birimini float yapalim.

# data_new['Installs']= data_new['Installs'].astype('float')
#

data.Installs.plot(kind = 'line', color = 'g',label = 'Installs',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

# plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()