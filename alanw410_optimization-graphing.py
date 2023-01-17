# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pylab import figure, axes, pie, title, show
from mpl_toolkits.mplot3d import Axes3D
from itertools import groupby
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

%matplotlib inline
plt.rcParams['figure.figsize'] = (20, 10)
saved_style_state = matplotlib.rcParams.copy()

filepath = "../input/dataset-5/Analysis results 5.csv"
print("Loading analysis data")
db = pd.read_csv("../input/dataset-5/Analysis results 5.csv")
print("done")
db.head(10)
len(db)
maxrange = db['Generation'].max()
iterations = list(range(0,maxrange + 1))
print(iterations)
totalCO2 = []
totalComfort = []
CO2Comfort = []
CO2vbI = []
ComfortvbI = []
for x in range(0, len(iterations)):
    temp = db.loc[db['Generation'] == x]
    CO2vbI.append(temp['CO2 (kg)'])
    ComfortvbI.append(temp['Discomfort (All Clothing) (hr)'])
    totalCO2.append(temp['CO2 (kg)'].sum() / len(temp['CO2 (kg)']))
    totalComfort.append(temp['Discomfort (All Clothing) (hr)'].sum() / len(temp['Discomfort (All Clothing) (hr)']))
print(totalCO2)
print(totalComfort)
totalCC = list(map(lambda x, y: x+y, totalCO2, totalComfort))
print(totalCC)
print(list(CO2vbI[0]))
print(len(list(CO2vbI[0])))
xaxis = []
for x in range(0, len(CO2vbI)):
    for x2 in range(0, len(CO2vbI[x])):
        xaxis.append(x)

yaxis = []
for x in range(0, len(CO2vbI)):
    yaxis = yaxis + list(CO2vbI[x])


y2axis = []
for x in range(0, len(ComfortvbI)):
    y2axis = y2axis + list(ComfortvbI[x])

f = plt.figure(1)
plt.scatter(xaxis,yaxis, color = 'g')
gpatch = mpatches.Patch(color = 'g', label = 'Total CO2 (Kg) Scatter')
plt.xlabel('Generations')
plt.ylabel('Units')
plt.title('Trend of Objective iterations grouped by generation.')
plt.legend(handles = [gpatch])
f.show()

g = plt.figure(2)
plt.scatter(xaxis,y2axis, color = 'c')
kpatch = mpatches.Patch(color = 'c', label = 'Discomfort Hrs Scatter')
plt.xlabel('Generations')
plt.ylabel('Units')
plt.title('Trend of Objective iterations grouped by generation.')
plt.legend(handles = [kpatch])
g.show()
x = iterations
y = totalCO2
y2 = totalComfort
y3 = totalCC


plt.plot(x,y, color = 'g')
plt.plot(x,y2, color = 'k')
plt.plot(x,y3, color = 'c')
gpatch = mpatches.Patch(color = 'g', label = 'Total CO2 (Kg) Average')
kpatch = mpatches.Patch(color = 'k', label = 'Total Discomfort Average')
cpatch = mpatches.Patch(color = 'c', label = 'Total CO2 (Kg) + Discomfort Average')
plt.xlabel('Generations')
plt.ylabel('Units')
plt.title('Trend of Objectives through generation (Average iteration objectives)')
plt.legend(handles = [cpatch, gpatch, kpatch])
plt.show()
x = db['CO2 (kg)']
plt.xlabel('CO2')
y = db['Discomfort (All Clothing) (hr)']
x2 = db.loc[db['Pareto'] == 'Pareto', 'CO2 (kg)']
y2 = db.loc[db['Pareto'] == 'Pareto', 'Discomfort (All Clothing) (hr)']
rpatch = mpatches.Patch(color = 'r', label = 'Pareto Point')
plt.ylabel('Discomfort')
plt.title('CO2 to Discomfort scatter')
plt.legend(handles = [rpatch], loc = 4)
plt.plot(x,y,'.')
plt.plot(x2,y2,'.', c = 'r')
dbSortCO2 = db.sort_values(by=['CO2 (kg)', 'Discomfort (All Clothing) (hr)'])
dbSortCO2.head(10)

dbSortCO2.loc[db['Pareto'] == 'Pareto']
len(dbSortCO2.loc[db['Pareto'] == 'Pareto'])
uniqueV = []
colorArray = []
uniqueV = list(db['Construction template'].unique())
print(uniqueV)
for x in range(1, len(uniqueV) + 1):
    value = x / len(uniqueV)
    colorArray.append(value)
len(db['Construction template'])
colorarrayG = []
for x in range(0, len(db['Construction template'])):
    test = db['Construction template'][x]
    arraynum = uniqueV.index(test)
    colorarrayG.append(colorArray[arraynum])
dbSortComfort = db.sort_values(by=['Discomfort (All Clothing) (hr)', 'CO2 (kg)'])
dbSortComfort.head(10)
zs = db['Generation']
xs = db['CO2 (kg)']
ys = db['Discomfort (All Clothing) (hr)']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs,ys,zs, c = colorarrayG)
ax.view_init(45, 45)
ax.set_xlabel('CO2 (kg)')
ax.set_ylabel('Discomfort (All Clothing) (hr)')
ax.set_zlabel('Generation')
plt.ioff()

for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.1)
    fig.savefig(str(angle) + '.png')
