# White house Obama salary plot

#version 1.0

#faiz ul haque zeya

#Data Scientist and Associate professor

#CEO Transys. A software agent and data science company



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as pp

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

def rd(a):



    if(a[0]=='$'):



       a=a[1:]



    return a



    



#obama salary o.csv change the path



a=pd.read_csv('../input/obama_staff_salaries.csv')







#count the values



b=a['salary'].value_counts()











#get the salary



sal=a['salary']







#apply rd to sal elements removing $



sal1=map(rd,sal)



#sort and remove duplicates



l=list(set(sal1))



#convert to int for plotting



ll=list(map(int,map(float,l)))



pp.bar(ll,b,1500)



pp.show()