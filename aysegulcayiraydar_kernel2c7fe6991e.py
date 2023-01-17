# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import glob

import pandas as pd

import numpy as np

from collections import Counter

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# Any results you write to the current directory are saved as output.
files=glob.glob(dirname+'/py*')

all_data = [pd.read_excel(f, sheet_name=None, ignore_index=True,sort=False,index_col=0) for f in files ]

all_class = pd.DataFrame(columns=("Class", "Name", "True", "False", "Empty"), index=None)

total_examers=0

for i in range(len(files)):   

    class_names=['py_mind','py_opinion','py_science','py_sense']

    names=list(all_data[i].keys())  

    for j in all_data[i]:    

        true=all_data[i][j]["ogr.C"][20]

        false=all_data[i][j]["ogr.C"][21]

        blank=all_data[i][j]["ogr.C"][22]

        DYB=list(all_data[i][j]["ogr.C"][:20])        

        student = {'Class': class_names[i], 'Name': names[names.index(j)],'True': true, 'False': false, "Empty": blank}

        all_class = all_class.append(student, ignore_index=True)

    total_examers+=(names.index(j))+1       

#ogrenci_cevaplari    ":DYB

print(all_class)

print("Total examers:",total_examers)

all_true = all_class["True"]

all_false = all_class["False"]

all_empty = all_class["Empty"]

#print("dogru sayilari =", sum(all_true), "Yanlis sayilari =", sum(all_false), "Bos sayilari = ", sum(all_empty))

#soru = 3 tum siniflarin Dogru yanlis bos ortalamalari

print("All Classes:\nD Average =",all_true.mean(),"\nY Average = ", all_false.mean(), "\nB Average =", all_empty.mean())

#%%
#%%

c_mind = all_class['True'][:10]

c_mind_ort=sum(c_mind)/10

print("\npy_mind average:",c_mind_ort)

c_opinion = all_class['True'][10:20]

c_opinion_ort=sum(c_opinion)/10

print("py_opinion average:",c_opinion_ort)

c_science = all_class['True'][20:28]

c_science_ort=sum(c_science)/10

print("py_science average:",c_science_ort)

c_sense = all_class['True'][28:38]

c_sense_ort=sum(c_sense)/10

print("py_sense average:",c_sense_ort)
#%%

all_1st3= all_class.sort_values(by='True', ascending=False).head(3)

print("Most succesfull 3 students:\n",all_1st3)
#%%

c_mind = all_class[all_class["Class"]=="py_mind"]

print("py_mind most succesfull:",c_mind.sort_values(by='True', ascending=False).head(1))

c_opinion = all_class[all_class["Class"]=="py_opinion"]

print("py_opinion most succesfull:",c_opinion.sort_values(by='True', ascending=False).head(1))

c_science = all_class[all_class["Class"]=="py_science"]

print("py_science most succesfull:",c_science.sort_values(by='True', ascending=False).head(1))

c_sense = all_class[all_class["Class"]=="py_sense"]

print("py_sense most succesfull:",c_sense.sort_values(by='True', ascending=False).head(1))