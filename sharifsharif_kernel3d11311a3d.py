# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import threading as t

import time



def sort_list(first,last):

    size = (last-first)+1

    firstend = first+int(size/2)+(size%2)-1

    secondstart = firstend+1

    if(size<=int((len(list)*0.25)+1)):

        list[first:last+1]=sorted(list[first:last+1])

        return

    sort_list(first,firstend)

    sort_list(secondstart,last)

    ls,a,b = [], list[first:firstend+1],list[secondstart:last+1]

    i = a.pop(0)

    j = b.pop(0)

    while(True):

        if(len(a)==0):

            if(i<=j):

                ls.append(i)

                ls.append(j)

            else:

                ls.append(j)

                ls.append(i)

            for element in b:

                ls.append(element)

            break

        elif(len(b) == 0):

            if(i<=j):

                ls.append(i)

                ls.append(j)

            else:

                ls.append(j)

                ls.append(i)



            for element in a:

                ls.append(element)

            break;

        if(i<=j):

            ls.append(i)

            i=a.pop(0)

        else:

            ls.append(j)

            j=b.pop(0)

    list[first:last+1]=ls



def func():

    first=0

    last=len(list)-1

    size = (last-first)+1

    firstend = first+int(size/2)+(size%2)-1

    secondstart = firstend+1



    t1.start()

    t2.start()

    t1.join()

    t1.join()

    ls,a,b = [], list[first:firstend+1],list[secondstart:last+1]

    i = a.pop(0)

    j = b.pop(0)

    while(True):

        if(len(a)==0):

            if(i<=j):

                ls.append(i)

                ls.append(j)

            else:

                ls.append(j)

                ls.append(i)

            for element in b:

                ls.append(element)

            break

        elif(len(b) == 0):

            if(i<=j):

                ls.append(i)

                ls.append(j)

            else:

                ls.append(j)

                ls.append(i)



            for element in a:

                ls.append(element)

            break;

        if(i<=j):

            ls.append(i)

            i=a.pop(0)

        else:

            ls.append(j)

            j=b.pop(0)

    list[first:last+1]=ls





list=[]

for i in range(0,int(1e+6)):

    list.append(i)

list.reverse()

t1 = t.Thread(target=sort_list,args=(0,int(len(list)/2)-1))

t2=t.Thread(target=sort_list,args=(int(len(list)/2), len(list)))

start=time.time()

func()

finish=time.time()

print(finish-start)