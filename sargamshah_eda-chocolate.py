# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

f = open("../input/flavors_of_cacao.csv","r")

 

d = pd.read_csv("../input/flavors_of_cacao.csv")  



 

print (d.head())

print (d.tail())

print (d.info())

print (d.describe())

# Any results you write to the current directory are saved as output.



print ("Where are the best cocoa beans grown?")



best_beans_grown = d[d["Rating"] == 5]["Broad Bean\nOrigin"]



countries = d[d["Rating"]==5]["Company\nLocation"]

print (best_beans_grown)

print ("Which countries produce the highest-rated bars?")



print(countries)



print ("What’s the relationship between cocoa solids percentage and rating?")



     



g = sns.factorplot( y="Rating", x="Coca\nPercent", data=d, 

                   saturation=.5, ci=None, size=5, aspect=.8)


