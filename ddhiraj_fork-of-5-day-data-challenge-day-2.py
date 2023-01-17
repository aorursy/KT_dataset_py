# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import math

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
a = (check_output(["ls", "../input"]).decode("utf8")).split('\n')  #extracting name of all the files.

print(a)
files =[pd.read_csv('../input/'+ a[i]) for i in range(len(a)-1)]  
shape = [files[i].shape for i in range(len(files)-1)]

print(shape)
files[1].head()   #every element of list 'files' is a Dataframe
files[3].describe()
for i in range(len(files)-1):

    files[i].reindex(files[i].id)

    del files[i]['id']  #removing clutter
files[2].head()
def diso(id):    # returns distace of a star from origin in all the snapshots

    dist=[]

    for i in range(len(files)-1):

        dist.append(math.sqrt(files[i].at[id,'x']**2 + files[i].at[id,'y']**2 + files[i].at[id,'z']**2))

    return dist
''''for i in range(45,50):

    plt.plot(diso(i))

    plt.title('Distance of star id-'+ str(i) + ' form origin with respect to time')

    plt.xlabel('time')

    plt.ylabel('distance form origin')

    plt.show()

'''
def disn(n,m):#returns a list containing distace from origin, for stars in id range (n,m).

    li=[]

    for i in range(n,m):

        li.append(diso(i))

    li = np.array(li)

    return li
def plot_d(n,m,time):                          

    data= disn(n,m)

    sns.distplot(data[:,time], label='Distance form origin')
plot_d(10,45,5)

plot_d(10,45,6)