# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np
#1

import os

os.listdir('../input')
#2

df = np.loadtxt('../input/University - University.csv', delimiter=',', skiprows=1)

print(df.shape)

df.dtype
#3

undstudents =df[:,1]

plt.hist(undstudents, bins=5)

plt.xlabel('Ox')

plt.title(r'График');
#4

undstudents = df[:,1]

poststudents = df[:,2]

acnumbers = df[:,4]

acrelnum = df[:,5]

a = sum(undstudents,poststudents)

b = sum(acnumbers,acrelnum)

print(a,b)

fig, ax = plt.subplots(figsize=(7, 7))

plt.scatter(a,b, marker= 'o', s=25, color='red')

plt.xlabel('undstudents,poststudents')

plt.ylabel('acnumbers,acrelnum')

plt.title('визуализация общего количества');



#5

stfees = df[:,9]

a = sum(undstudents,poststudents)

fig, ax = plt.subplots(figsize=(7, 7))

plt.xlabel('undstudents,poststudents')

plt.ylabel('stfees')

plt.scatter(a, stfees);
#6

acpay = df[:,10]

plt.boxplot(acpay,vert=False, labels=['acpay']);
#7

acnumbers = df[:,4]

acrelnum = df[:,5]

clernum = df[:,6]

compop = df[:,7]

techn = df[:,8]

q = sum(acnumbers)

w = sum(acrelnum)

qw = q+w

print(qw)

e = sum(clernum)

r = sum(compop)

t = sum(techn)

ert = e+r+t

print(ert)

z=np.array([qw,ert])

labels='acnumbers, acrelnum','clernum, compop, techn'

fig1, ax1 = plt.subplots(figsize=(20, 7))

plt.pie(z, labels = labels);
#8

fig, ax = plt.subplots(figsize=(10, 7))

acnumbers = df[:,4]

acpay = df[:,10]

agresrk = df[:,14]

resgr = df[:,17]

plt.scatter(resgr,acnumbers)

plt.scatter(resgr, acpay)

plt.scatter(resgr,agresrk);
#8

plt.xlabel('resgr')

plt.ylabel('acnumbers')

plt.scatter(resgr,acnumbers);
#8

plt.xlabel('resgr')

plt.ylabel('acpay')

plt.scatter(resgr, acpay);

#8

plt.xlabel('resgr')

plt.ylabel('agresrk')

plt.scatter(resgr,agresrk);
#9

fig, ax = plt.subplots()

undstudents = df[:,1]

poststudents = df[:,2]

a = sum(undstudents)

f = sum(poststudents)

plt.bar([1, 2], height=[a, f], color=['red', 'blue'])

ax.set_xticks([1, 2])

ax.set_xticklabels(['undstudents', 'poststudents']);
furneq = df[:,15]

landbuild = df[:,16]

resgr = df[:,17] 

fig, ax = plt.subplots(2, figsize=(10, 10))

ax[0].scatter(resgr,landbuild )

ax[0].set_xlabel('resgr') 

ax[0].set_ylabel('landbuild')

ax[1].scatter(resgr,furneq, color = 'red')

ax[1].set_xlabel('resgr') 

ax[1].set_ylabel('furneq');
