# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas.io.gbq as bq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
%matplotlib notebook
import seaborn as sns
%matplotlib inline 
import networkx as nx# data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
cdeaths = pd.read_csv("../input/character-deaths.csv")
#the analysis of deaths
cdeaths.describe()
x={}
for i in cdeaths["Allegiances"]:
    x[i]=x.get(i,0)+1

plt.barh(range(len(x)), x.values())
plt.yticks(range(len(x)), x.keys())

plt.show()

print ("Number of male deaths :")
print (len(cdeaths[cdeaths["Gender"]!=0]))
print ("Number of female deaths: ")
print (len(cdeaths[cdeaths["Gender"]==0]))
