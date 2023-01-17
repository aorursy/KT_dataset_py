#This is my first homework at Data Science
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
import pandas as pd
data = pd.read_csv("../input/battles.csv")
x = data['year']>299
data[x]
#filter-To see battles later than year 300

#This is an histogram which shows defender_size in battles.csv file.
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../input/battles.csv")
data.defender_size.plot(kind = 'hist',bins = 50)
plt.show()
#Here i wanted to see top 5 names on character-deaths list.
import pandas as pd
x = pd.read_csv("../input/character-deaths.csv")
x.head()
#Here we see attacker sizes which are  bigger than 10000 and defender sizes which are bigger than 7000
import numpy as np
data[np.logical_and(data['attacker_size']>10000, data['defender_size']>7000 )]

#pandas example 1
x = data['defender_size']>500
data[x]

#pandas example 2
x = data['attacker_king']=="Robb Stark"
data[x]
import pandas as pd
data = pd.read_csv("../input/battles.csv")

data1 = data['attacker_king'].head(10)
data2= data['region'].head(10)
conc_data_col = pd.concat([data1,data2],axis =1) 
conc_data_col
data.plot(subplots = False)
plt.show()
data = pd.read_csv("../input/battles.csv")
data["attacker_king"][6]
data.loc[10,["defender_king"]]
data[["attacker_king","defender_king"]]
boolean = data.attacker_outcome =="win"
data[boolean]
first_filter = data.attacker_outcome == "win"
second_filter = data.attacker_king=="Robb Stark"
data[first_filter & second_filter]