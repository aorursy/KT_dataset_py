# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

file= pd.read_csv('../input/commute_data.csv', encoding = "ISO-8859-1")

import time

file1=file

tmps1=time.clock()

file1['drap']=False

zz=file1['drap'].count()

file1['col4']=0

i=0

for i in range (0,100) :

    #if  (file1.drap[i] == False) :

    file1['col'] = (file1['OSTFIPS'] == i)

    a=file1['col'].sum()-1

    file1['col1']=file1.col * a

    file1.drap=file1['col'] | file1['drap']

    file1['col4']=file1.col1+file1.col4

    #print(i)

tmps2=time.clock()

print ("Temps d'execution = %d\n" %(tmps2-tmps1))

print(file1.head(10))


