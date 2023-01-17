import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import statistics
my_data = pd.read_csv("../input/household-appliances/Dishwasher.csv")

my_data=my_data[my_data['Brand'].isin(['V-ZUG','SIEMENS', 'MIDEA', 'De Dietrich', 'HOOVER'])]
ratings=my_data.groupby('Brand')[['New Star', 'Tot Wat Cons']].mean()

fig = plt.figure(figsize=(15,10)) 
ax = fig.add_subplot(111) 
ax2 = ax.twinx() 

width = 0.3

ax.set_ylabel('Star rating', fontsize = 15)
ax.set_xlabel('Brand(Top five brands as per star rating)', fontsize = 15)
ax2.set_ylabel('Total water consumption(litres)', fontsize = 15)
ax.set_title("Top five brands and their water consumption", fontsize=15)



ratings['New Star'].plot(kind='bar',color='red', ax=ax, width=width, position=1)

ratings['Tot Wat Cons'].plot(kind='bar', color='blue', ax=ax2, width=width, position=0)
ax.legend(['New Star'], loc='upper right', frameon=True, fontsize=10)

ax2.legend(['Tot Wat Cons'], loc='upper left', frameon=True, fontsize=10)

#plt.legend()
plt.grid()
plt.show()

my_data=my_data[my_data['Brand'].isin(['V-ZUG','SIEMENS', 'MIDEA', 'De Dietrich', 'HOOVER'])]
ratings=my_data.groupby('Brand')[['New Star', 'CEC_']].mean()

fig = plt.figure(figsize=(15,10)) 
ax = fig.add_subplot(111) 
ax2 = ax.twinx() 

width = 0.3

ratings['New Star'].plot(kind='bar', color='red', ax=ax, width=width, position=1,  fontsize = '15')

ratings['CEC_'].plot(kind='bar', color='blue', ax=ax2, width=width, position=0, fontsize = '15')

ax.set_ylabel('Star rating', fontsize = 15)
ax.set_xlabel('Brand(Top five brands as per star rating)', fontsize = 15)
ax2.set_ylabel('Comparative Energy Consumption(kilowatt hours per years)', fontsize = 15)
ax.set_title("Top five brands and their energy consumption", fontsize=15)

ax.legend(['New Star'], loc='upper right', frameon=True, fontsize=10)

ax2.legend(['CEC_'], loc='upper left', frameon=True, fontsize=10)

#plt.legend()
plt.grid()
plt.show()
my_data = pd.read_csv("../input/household-appliances/Clothwasher.csv")
import statistics
ratings=my_data.groupby('Brand')[['New Star']].mean()
ratings=ratings.reset_index()
ratings=ratings.sort_values('New Star',ascending=False)
ratings=ratings[:5]
ratings
plt.figure(figsize=(10,7))
plt.plot(ratings['Brand'], ratings['New Star'])
plt.xlabel("Brand")
plt.ylabel("New Star rating")
plt.title("New star rating of different brand Clothwashers")
plt.show()

ratings=my_data.groupby('Brand')['CEC_'].mean()
ratings=ratings.to_frame().reset_index()
ratings=ratings.sort_values('CEC_',ascending=False)
ratings=ratings[:7]
ratings

my_data=my_data[my_data['Brand'].isin(['AEG','ASKO','Gorenje','MIELE','V-ZUG'])]
ratings=my_data.groupby('Brand')[['Cap', 'CEC_']].mean()

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)
ax2 = ax.twinx()

width = 0.3

ratings['Cap'].plot(kind='bar', color='red', ax=ax, width=width, position=1)
ratings['CEC_'].plot(kind='bar', color='blue', ax=ax2, width=width, position=0)

ax.set_ylabel('Comparitive energy consumption - kW/year)', color='red',fontsize=16)
ax2.set_ylabel('Comparitive energy consumption - kW/year)', color='Blue',fontsize=16)
ax.set_title("Top 5 brands with their capacity and warm wash mode energy consumption)", fontsize=26)
ax.legend(['Rated Capacity'], loc='upper right', frameon=True, fontsize=10)
ax2.legend(['CEC_'], loc='upper left', frameon=True, fontsize=10)
plt.show()



my_data = pd.read_csv("../input/household-appliances/Clothwasher.csv")
ratings=my_data.groupby('Brand')['CEC_'].mean()
ratings=ratings.to_frame().reset_index()
ratings=ratings.sort_values('CEC_',ascending=False)
ratings=ratings[:7]
ratings

my_data=my_data[my_data['Brand'].isin(['MIDEA','HOOVER','Euromaid','ELECTROLUX','Bertazzoni'])]
ratings=my_data.groupby('Brand')[['Cap', 'CEC_']].mean()

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)
ax2 = ax.twinx()

width = 0.3

ratings['Cap'].plot(kind='bar', color='red', ax=ax, width=width, position=1)
ratings['CEC_'].plot(kind='bar', color='blue', ax=ax2, width=width, position=0)

ax.set_ylabel('Comparitive energy consumption - kW/year)', color='red',fontsize=16)
ax2.set_ylabel('Comparitive energy consumption - kW/year)', color='Blue',fontsize=16)
ax.set_title("Top 4 brands based on energy consumption & Capacity)", fontsize=20)
ax.legend(['Rated Capacity'], loc='upper right', frameon=True, fontsize=10)
ax2.legend(['CEC_'], loc='upper left', frameon=True, fontsize=10)
plt.show()
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import DataFrame
import matplotlib.pyplot as plt
from statistics import stdev
my_data = pd.read_csv("../input/household-appliances/Fridge.csv")
ratings=my_data.groupby('Brand')['Star2009'].mean()
ratings=ratings.to_frame().reset_index()
ratings=ratings.sort_values('Star2009',ascending=False)
ratings=ratings[:10]
ratings 
plt.figure(figsize=(10,5))
plt.barh(ratings['Brand'], ratings['Star2009'])
plt.xlabel("Star Rating ")              
plt.ylabel("Brand ")
plt.title("Star Rating for the Top Ten Brands")
plt.show()
my_data=my_data[my_data['Brand'].isin(['JVD','Electrolux', 'ARISTON', 'Panasonic', 'Fairhall'])]
ratings=my_data.groupby('Brand')[['Star2009', 'S-MEPScutoff']].mean()

fig = plt.figure(figsize=(15,10))
ax.set_title("Cutoff energy vs Top Five Brands", fontsize=16)
ax.legend(['Star Rating', 'S-MEPScutoff'], loc='upper right', frameon=True, fontsize=14)
ax = fig.add_subplot(111)
ax2 = ax.twinx()

width = 0.4

ratings['Star2009'].plot(kind='bar', color='red', ax=ax, width=width, position=1)
ratings['S-MEPScutoff'].plot(kind='bar', color='blue', ax=ax2, width=width, position=0)


ax.set_ylabel('Star rating', color='red')
ax2.set_ylabel('S-MEPScutoff (kW)', color='blue')
ax.set_title("Cutoff energy(kW) for the Top Rated Five Brands", fontsize=16)
ax.legend(['Star Rating'], loc='upper right', frameon=True, fontsize=10)

ax2.legend(['S-MEPScutoff (kW)'], loc='upper left', frameon=True, fontsize=10)

plt.show()
my_data=my_data[my_data['Brand'].isin(['JVD','Electrolux', 'ARISTON', 'Panasonic', 'Fairhall'])]
ratings=my_data.groupby('Brand')[['Star2009', 'CEC_']].mean()

fig = plt.figure(figsize=(15,10))
ax.set_title("Comparative Energy Consumption (kWh/year) for the Top Rated Five Brands", fontsize=16)
ax.legend(['Star Rating', 'CEC_'], loc='upper right', frameon=True, fontsize=14)
ax = fig.add_subplot(111)
ax2 = ax.twinx()

width = 0.4

ratings['Star2009'].plot(kind='bar', color='red', ax=ax, width=width, position=1)
ratings['CEC_'].plot(kind='bar', color='blue', ax=ax2, width=width, position=0)


ax.set_ylabel('Star rating', color='red')
ax2.set_ylabel('CEC- Comparative Energy Consumption (kWh/year)', color='blue')
ax.set_title("Comparative Energy Consumption (kWh/year) for the Top Rated Five Brands", fontsize=16)
ax.legend(['Star Rating'], loc='upper right', frameon=True, fontsize=10)
ax2.legend(['CEC (Comparative Energy Consumption) (kWh/year)'], loc='upper left', frameon=True, fontsize=10)
plt.grid()
plt.show()
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import DataFrame
import matplotlib.pyplot as plt
from statistics import stdev 
my_data = pd.read_csv("../input/household-appliances/Television.csv")
ratings=my_data.groupby('Brand_Reg')['Star'].mean()
ratings=ratings.to_frame().reset_index()
ratings=ratings.sort_values('Star',ascending=False)
ratings=ratings[:10]
ratings  
ratings=my_data.groupby('Brand_Reg')['Star2'].mean()
ratings=ratings.to_frame().reset_index()
ratings=ratings.sort_values('Star2',ascending=True)
ratings=ratings[:10]
ratings  
plt.figure(figsize=(10,5))
plt.barh(ratings['Brand_Reg'], ratings['Star2'])
plt.xlabel("Star Rating ")              
plt.ylabel("Brand ")
plt.title("Top ten Brands based on Star Rating")
plt.show()
ratings=my_data.groupby('Brand_Reg')['Avg_mode_power'].mean()
ratings=ratings.to_frame().reset_index()
ratings=ratings.sort_values('Avg_mode_power',ascending=False)
ratings=ratings[:5]
ratings
plt.figure(figsize=(10,5))
plt.barh(ratings['Brand_Reg'], ratings['Avg_mode_power'])
plt.xlabel("Average Power Usage (kW)")              
plt.ylabel("Brand ")
plt.title("Brands ranked based on Average Power Usage")
plt.show()
my_data = pd.read_csv("../input/household-appliances/Aircon.csv")
import statistics
my_data=my_data[my_data['Brand'].isin(['SPECIALIZED ENGINEERING','SAI HVAC','Midea','Clivet','CAA'])]
ratings=my_data.groupby('Brand')[['H-Power_Inp_Rated', 'C-Power_Inp_Rated']].mean()

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
ax2 = ax.twinx()

width = 0.3

ratings['H-Power_Inp_Rated'].plot(kind='bar', color='red', ax=ax, width=width, position=1)
ratings['C-Power_Inp_Rated'].plot(kind='bar', color='blue', ax=ax2, width=width, position=0)

ax.set_ylabel('Electrical power used for heating upto 7 C - kW)', color='red',fontsize=16)
ax2.set_ylabel('Electrical power used to cool at 35 C kW)', color='Blue',fontsize=16)
ax.set_title("Top 5 brands tested EER of heating at 7C & cooling at 35C)", fontsize=16)
ax.legend(['H-Power_Inp_Rated'], loc='upper right', frameon=True, fontsize=10)
ax2.legend(['C-Power_Inp_Rated'], loc='upper left', frameon=True, fontsize=10)
plt.show()
my_data=my_data[my_data['Brand'].isin(['SPECIALIZED ENGINEERING','SAI HVAC','Midea','Clivet','CAA'])]
ratings=my_data.groupby('Brand')[['EERtestAvg', 'COPtestAvg']].mean()

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
ax2 = ax.twinx()

width = 0.3

ratings['EERtestAvg'].plot(kind='bar', color='red', ax=ax, width=width, position=1)
ratings['COPtestAvg'].plot(kind='bar', color='blue', ax=ax2, width=width, position=0)

ax.set_ylabel('Comparitive energy consumption - kW/year)', color='red',fontsize=16)
ax2.set_ylabel('Comparitive energy consumption - kW/year)', color='Blue',fontsize=16)
ax.set_title("Top 5 brands with EER of heating at 7C & cooling at 35C at Full load)", fontsize=16)
ax.legend(['EERtestAvg'], loc='upper right', frameon=True, fontsize=10)
ax2.legend(['COPtestAvg'], loc='upper left', frameon=True, fontsize=10)
plt.show()