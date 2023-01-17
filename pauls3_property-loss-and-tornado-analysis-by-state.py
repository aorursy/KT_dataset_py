# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Tornadoes_SPC_1950to2015.csv')

tonadoes_1996 = df[df['yr'] >= 1996][['st','loss']]
tonadoes_1996.head(10)
tonadoes_1996_damage = pd.DataFrame({'count': tonadoes_1996.groupby(['st'])['loss'].count(), 'total_loss': tonadoes_1996.groupby(['st'])['loss'].sum()})
tonadoes_1996_damage
#Separate the dataframe into series

count = tonadoes_1996_damage['count'] 

loss = tonadoes_1996_damage['total_loss']



#Find the state with maximum number of tornadoes and then get the index label and plot points

max_tornado_count = tonadoes_1996_damage['count'].max()

max_tornado_cnt_label = tonadoes_1996_damage[tonadoes_1996_damage['count'] == max_tornado_count].index.tolist()[0]

max_tornado_cnt_x = tonadoes_1996_damage[tonadoes_1996_damage['count'] == max_tornado_count]['count']

max_tornado_cnt_y = tonadoes_1996_damage[tonadoes_1996_damage['count'] == max_tornado_count]['total_loss']



#Find the state with maximum amount of property damage and then get the index label and plot points

max_tornado_loss = tonadoes_1996_damage['total_loss'].max()

max_tornado_loss_label = tonadoes_1996_damage[tonadoes_1996_damage['total_loss'] == max_tornado_loss].index.tolist()[0]

max_tornado_loss_x = tonadoes_1996_damage[tonadoes_1996_damage['total_loss'] == max_tornado_loss]['count']

max_tornado_loss_y = tonadoes_1996_damage[tonadoes_1996_damage['total_loss'] == max_tornado_loss]['total_loss']



#Prepare our plot

colors = np.random.rand(51)

area = count

plt.scatter(count, loss,s=area,c=colors,alpha=.5)



#Provide axis labels and a title

xlab = "Number of Tornadoes [in thousands]"

ylab = "Total Loss [in million USD]"

title = "Total Property Loss Since 1996"



plt.xlabel(xlab)

plt.ylabel(ylab)

plt.title(title)



#set the axis limits

plt.xlim(0, 3500)

plt.ylim(0, 6000)



#Apply grid lines for good measure

plt.grid(True)



#Plot the max values for count and loss

plt.text(max_tornado_cnt_x, max_tornado_cnt_y, max_tornado_cnt_label)

plt.text(max_tornado_loss_x, max_tornado_loss_y, max_tornado_loss_label)



plt.show()