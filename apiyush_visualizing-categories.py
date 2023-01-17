%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import math as math
# Importing the dataset

datatrain = pd.read_csv(r'../input/train.csv')


colors = {

 'a': 'seagreen',

 'aa': 'plum',

 'ab': 'darkgreen',

 'ac': 'cyan',

 'ad': 'mediumorchid',

 'ae': 'gold',

 'af': 'm',

 'ag': 'rosybrown',

 'ah': 'orange',

 'ai': 'lightskyblue',

 'aj': 'khaki',

 'ak': 'firebrick',

 'al': 'teal',

 'am': 'mediumspringgreen',

 'an': 'midnightblue',

 'ao': 'pink',

 'ap': 'limegreen',

 'aq': 'limegreen',

 'ar': 'slategray',

 'as': 'saddlebrown',

 'at': 'darkmagenta',

 'au': 'grey',

 'av': 'lavender',

 'aw': 'silver',

 'ax': 'burlywood',

 'ay': 'sandybrown',

 'az': 'darkolivegreen',

 'b': 'lightcyan',

 'ba': 'navajowhite',

 'bc': 'lightslategray',

 'c': 'darkorchid',

 'd': 'greenyellow',

 'e': 'lime',

 'f': 'g',

 'g': 'salmon',

 'h': 'red',

 'i': 'deeppink',

 'j': 'mediumblue',

 'k': 'springgreen',

 'l': 'darkorange',

 'm': 'rebeccapurple',

 'n': 'lightpink',

 'o': 'darkblue',

 'q': 'powderblue',

 'r': 'yellow',

 's': 'green',

 't': 'tomato',

 'u': 'turquoise',

 'v': 'thistle',

 'w': 'chartreuse',

 'x': 'lightcoral',

 'y': 'mediumaquamarine',

 'z': 'darkviolet',

 'p': 'navy'}
fig, ax = plt.subplots(facecolor='k',figsize=(18, 9))

#columns = ["X0","X1","X2","X3","X4","X5","X6","X8"]

columns = ["X0"]



x = 1

for column in columns:

    for label in np.unique(datatrain[column]):

        temp = datatrain[datatrain[column]==label]

        x1 = list(range(x,x+len(temp)))

        plt.scatter(x=x1,y=temp["y"],s=8,c=[colors[x] for x in list(temp[column])],label=label)

        #plt.axvline(x=x+len(temp),linewidth=1,color='black')

        x = x+len(temp)+1

#plt.figure(figsize=(12,12))

plt.legend(loc='best',

           labelspacing=0.001,fontsize='xx-large',ncol=10)

ax = plt.gca()

ax.patch.set_facecolor("k")

ax.spines['bottom'].set_color('#dddddd')

ax.spines['left'].set_color('#dddddd')

ax.tick_params(axis='x', colors='#dddddd')

ax.tick_params(axis='y', colors='#dddddd')

ax.yaxis.label.set_color('#dddddd')

ax.xaxis.label.set_color('#dddddd')

plt.show()
#columns = ["X0","X1","X2","X3","X4","X5","X6","X8"]

columns = ["X1"]

fig, ax = plt.subplots(facecolor='k',figsize=(18, 9))

x = 1

for column in columns:

    for label in np.unique(datatrain[column]):

        temp = datatrain[datatrain[column]==label]

        x1 = list(range(x,x+len(temp)))

        plt.scatter(x=x1,y=temp["y"],s=10,c=[colors[x] for x in list(temp[column])],label=label)

        #plt.axvline(x=x+len(temp),linewidth=1,color='black')

        x = x+len(temp)+1

#plt.figure(figsize=(12,12))

plt.legend(loc='best',

           labelspacing=0.001,fontsize='xx-large',ncol=10)

ax = plt.gca()

ax.patch.set_facecolor("k")

ax.spines['bottom'].set_color('#dddddd')

ax.spines['left'].set_color('#dddddd')

ax.tick_params(axis='x', colors='#dddddd')

ax.tick_params(axis='y', colors='#dddddd')

ax.yaxis.label.set_color('#dddddd')

ax.xaxis.label.set_color('#dddddd')

plt.show()
#columns = ["X0","X1","X2","X3","X4","X5","X6","X8"]

columns = ["X2"]

fig, ax = plt.subplots(facecolor='k',figsize=(18, 9))

x = 1

for column in columns:

    for label in np.unique(datatrain[column]):

        temp = datatrain[datatrain[column]==label]

        x1 = list(range(x,x+len(temp)))

        plt.scatter(x=x1,y=temp["y"],s=8,c=[colors[x] for x in list(temp[column])],label=label)

        #plt.axvline(x=x+len(temp),linewidth=1,color='black')

        x = x+len(temp)+1

#plt.figure(figsize=(12,12))

plt.legend(loc='best',

           labelspacing=0.001,fontsize='xx-large',ncol=10)

ax = plt.gca()

ax.patch.set_facecolor("k")

ax.spines['bottom'].set_color('#dddddd')

ax.spines['left'].set_color('#dddddd')

ax.tick_params(axis='x', colors='#dddddd')

ax.tick_params(axis='y', colors='#dddddd')

ax.yaxis.label.set_color('#dddddd')

ax.xaxis.label.set_color('#dddddd')

plt.show()
#columns = ["X0","X1","X2","X3","X4","X5","X6","X8"]

columns = ["X3"]

fig, ax = plt.subplots(facecolor='k',figsize=(18, 9))

x = 1

for column in columns:

    for label in np.unique(datatrain[column]):

        temp = datatrain[datatrain[column]==label]

        x1 = list(range(x,x+len(temp)))

        plt.scatter(x=x1,y=temp["y"],s=8,c=[colors[x] for x in list(temp[column])],label=label)

        #plt.axvline(x=x+len(temp),linewidth=1,color='black')

        x = x+len(temp)+1

#plt.figure(figsize=(12,12))

plt.legend(loc='best',

           labelspacing=0.001,fontsize='xx-large',ncol=7)

ax = plt.gca()

ax.patch.set_facecolor("k")

ax.spines['bottom'].set_color('#dddddd')

ax.spines['left'].set_color('#dddddd')

ax.tick_params(axis='x', colors='#dddddd')

ax.tick_params(axis='y', colors='#dddddd')

ax.yaxis.label.set_color('#dddddd')

ax.xaxis.label.set_color('#dddddd')

plt.show()
#columns = ["X0","X1","X2","X3","X4","X5","X6","X8"]

columns = ["X4"]

fig, ax = plt.subplots(facecolor='k',figsize=(18, 9))

x = 1

for column in columns:

    for label in np.unique(datatrain[column]):

        temp = datatrain[datatrain[column]==label]

        x1 = list(range(x,x+len(temp)))

        plt.scatter(x=x1,y=temp["y"],s=8,c=[colors[x] for x in list(temp[column])],label=label)

        #plt.axvline(x=x+len(temp),linewidth=1,color='black')

        x = x+len(temp)+1

#plt.figure(figsize=(12,12))

plt.legend(loc='best',

           labelspacing=0.001,fontsize='xx-large',ncol=4)

ax = plt.gca()

ax.patch.set_facecolor("k")

ax.spines['bottom'].set_color('#dddddd')

ax.spines['left'].set_color('#dddddd')

ax.tick_params(axis='x', colors='#dddddd')

ax.tick_params(axis='y', colors='#dddddd')

ax.yaxis.label.set_color('#dddddd')

ax.xaxis.label.set_color('#dddddd')

plt.show()
#columns = ["X0","X1","X2","X3","X4","X5","X6","X8"]

columns = ["X5"]

fig, ax = plt.subplots(facecolor='k',figsize=(18, 9))

x = 1

for column in columns:

    for label in np.unique(datatrain[column]):

        temp = datatrain[datatrain[column]==label]

        x1 = list(range(x,x+len(temp)))

        plt.scatter(x=x1,y=temp["y"],s=8,c=[colors[x] for x in list(temp[column])],label=label)

        plt.axhline(y=85,linewidth=1,color='yellow')

        x = x+len(temp)+1

#plt.figure(figsize=(12,12))

plt.legend(loc='best',

           labelspacing=0.001,fontsize='xx-large',ncol=10)

ax = plt.gca()

ax.patch.set_facecolor("k")

ax.spines['bottom'].set_color('#dddddd')

ax.spines['left'].set_color('#dddddd')

ax.tick_params(axis='x', colors='#dddddd')

ax.tick_params(axis='y', colors='#dddddd')

ax.yaxis.label.set_color('#dddddd')

ax.xaxis.label.set_color('#dddddd')

plt.show()
#columns = ["X0","X1","X2","X3","X4","X5","X6","X8"]

columns = ["X6"]

fig, ax = plt.subplots(facecolor='k',figsize=(18, 9))

x = 1

for column in columns:

    for label in np.unique(datatrain[column]):

        temp = datatrain[datatrain[column]==label]

        x1 = list(range(x,x+len(temp)))

        plt.scatter(x=x1,y=temp["y"],s=8,c=[colors[x] for x in list(temp[column])],label=label)

        #plt.axvline(x=x+len(temp),linewidth=1,color='black')

        x = x+len(temp)+1

#plt.figure(figsize=(12,12))

plt.axhline(y=85,linewidth=1,color='black')

plt.legend(loc='best',

           labelspacing=0.001,fontsize='xx-large',ncol=10)

ax = plt.gca()

ax.patch.set_facecolor("k")

ax.spines['bottom'].set_color('#dddddd')

ax.spines['left'].set_color('#dddddd')

ax.tick_params(axis='x', colors='#dddddd')

ax.tick_params(axis='y', colors='#dddddd')

ax.yaxis.label.set_color('#dddddd')

ax.xaxis.label.set_color('#dddddd')

plt.show()
#columns = ["X0","X1","X2","X3","X4","X5","X6","X8"]

columns = ["X8"]

fig, ax = plt.subplots(facecolor='k',figsize=(18, 9))

x = 1

for column in columns:

    for label in np.unique(datatrain[column]):

        temp = datatrain[datatrain[column]==label]

        x1 = list(range(x,x+len(temp)))

        plt.scatter(x=x1,y=temp["y"],s=8,c=[colors[x] for x in list(temp[column])],label=label)

        plt.axhline(y=85,linewidth=1,color='yellow')

        x = x+len(temp)+1

#plt.figure(figsize=(12,12))

plt.legend(loc='best',

           labelspacing=0.001,fontsize='xx-large',ncol=10)

ax = plt.gca()

ax.patch.set_facecolor("k")

ax.spines['bottom'].set_color('#dddddd')

ax.spines['left'].set_color('#dddddd')

ax.tick_params(axis='x', colors='#dddddd')

ax.tick_params(axis='y', colors='#dddddd')

ax.yaxis.label.set_color('#dddddd')

ax.xaxis.label.set_color('#dddddd')

plt.show()