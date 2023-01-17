import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as mp

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

data = pd.read_csv("../input/nypd-motor-vehicle-collisions.csv")
data.head()
#visualizing longitutde and latitude to find which part of city is more dangerous
#not useful!
#lng = data['LONGITUDE']
#lat = data['LATITUDE']
#mp.plot(lat,'bo')

    

#vehicle type accidents
Vtypes = list(data['VEHICLE TYPE CODE 1'].unique()) #vehicle types
Atypes = np.zeros(len(Vtypes)) #set each vehicle type accidents to zero

#CONTRIBUTING FACTOR VEHICLE 1
#search over all types, sum the number up
for itr in range(len(Vtypes)): 
    for j in range(5):
        strTemp = 'VEHICLE TYPE CODE ' + str(j+1);#check all five possible cars
        d = data[strTemp] #read the column
        ind = [i for i,x in enumerate(d) if x == Vtypes[itr]] #find all occurences
        Atypes[itr] = Atypes[itr] + len(ind) #add all occurences

mp.xticks(range(len(Vtypes)),Vtypes)
mp.plot(Atypes)
#mp.xticks(range(len(Vtypes)),Vtypes)
mp.bar(range(len(Atypes)),Atypes)
#factor type accidents
Ftypes = list(data['CONTRIBUTING FACTOR VEHICLE 1'].unique()) #vehicle types
Atypes = np.zeros(len(Ftypes)) #set each vehicle type accidents to zero

#CONTRIBUTING FACTOR VEHICLE 1
#search over all types, sum the number up
for itr in range(len(Ftypes)): 
    for j in range(5):
        strTemp = 'CONTRIBUTING FACTOR VEHICLE ' + str(j+1);#check all five possible cars
        d = data[strTemp] #read the column
        ind = [i for i,x in enumerate(d) if x == Ftypes[itr]] #find all occurences
        Atypes[itr] = Atypes[itr] + len(ind) #add all occurences

mp.xticks(range(len(Ftypes)),Ftypes)
mp.plot(Atypes)
mp.xticks(range(len(Ftypes)),Ftypes)
mp.bar(range(len(Atypes)),Atypes)
#factor type accidents
Btypes = list(data['BOROUGH'].unique()) #vehicle types
Atypes = np.zeros(len(Btypes)) #set each vehicle type accidents to zero

#CONTRIBUTING FACTOR VEHICLE 1
#search over all types, sum the number up
d = data['BOROUGH'] #read the column
for itr in range(len(Btypes)): 
    ind = [i for i,x in enumerate(d) if x == Btypes[itr]] #find all occurences
    Atypes[itr] = Atypes[itr] + len(ind) #add all occurences

mp.xticks(range(len(Btypes)),Btypes)
mp.bar(range(len(Btypes)),Atypes)