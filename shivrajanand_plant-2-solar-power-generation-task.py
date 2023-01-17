# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #data visulaization

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#USED TO CLEAR THE OUTPUT KERNEL
def clear():
    print("\n"*50)

    
#CREATES A HATCH BLOCK TO MAKE OUTPUT READABLE    
def create_block():
    print("\n\n\n")
    print("#"*100)
    print("\n\n\n")
p2_gen = pd.read_csv("../input/solar-power-generation-data/Plant_2_Generation_Data.csv")
p2_wet = pd.read_csv("../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv")
def unique_cols(df, cols):
    
    l = df[cols].unique()
    
    print("There are total",len(l),"unique",cols,"\n")
    print("#"*100,"\n")

    for i in range(len(l)):
        print(cols,"NUMBER",i+1,":-  ", l[i] )
        
###############################################################################################################################################################################        
###############################################################################################################################################################################  
###############################################################################################################################################################################  


def column_list_creator(df, DS_name):
    col_list = df.columns                            #STORING COLUMNS IN LIST DATATYPE
    print("########## Columns In DataSet", DS_name, "##########\n")   #PRINTING HEADER FOR COLUMNS
    
    for i in range(len(col_list)):
        print("#",i+1,"...",col_list[i])        #PRINTING COLUMNS THROUGH ITERATION
        
    print("\n"*5)                          #CLEARING LITTLE SPACE FOR NEXT OUTPUT

    
    
###############################################################################################################################################################################  
###############################################################################################################################################################################  
###############################################################################################################################################################################  

def basic_stats(df, col):
    print("#####-----BASIC STATISTICS ON",col,"-----#####\n\n")
    print("MAXIMUM ",col," IS:- ", df[col].max())
    print("MINIMUM ",col," IS:- ", df[col].min())
    print("AVERAGE ",col," IS:- ", df[col].mean())
    create_block()



###############################################################################################################################################################################  
###############################################################################################################################################################################  
###############################################################################################################################################################################  
        

def unique_data(df, cols):
    print("ÜNIQUE VALUES OCCURENCE IN COLUMN", cols, ":-")
    print(df[cols].value_counts())

column_list_creator(p2_gen, "PLANT_2_GENERATION_DATA")
column_list_creator(p2_wet, "PLANT_2_WEATHER_SENSOR_DATA")

unique_cols(p2_gen, "SOURCE_KEY")
basic_stats(p2_gen,'TOTAL_YIELD')
basic_stats(p2_gen, 'DAILY_YIELD')
basic_stats(p2_gen, 'DC_POWER')
basic_stats(p2_gen, 'AC_POWER')
basic_stats(p2_wet, 'AMBIENT_TEMPERATURE')
basic_stats(p2_wet, 'MODULE_TEMPERATURE')
basic_stats(p2_wet, 'IRRADIATION')

unique_data(p2_gen, "SOURCE_KEY")
print("MEAN VALUE OF DAILY YIELD IS:-  ",p2_gen["DAILY_YIELD"].mean())
#CREATING DATE AND TIME COLUMNS SEPARATELY FOR WEATHER DATA

p2_wet['DATE_TIME'] = pd.to_datetime(p2_wet['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')

p2_wet['DATE'] = p2_wet['DATE_TIME'].apply(lambda x:x.date())
p2_wet['TIME'] = p2_wet['DATE_TIME'].apply(lambda x:x.time())

p2_wet['HOUR'] = pd.to_datetime(p2_wet['TIME'],format='%H:%M:%S').dt.hour
p2_wet['MINUTES'] = pd.to_datetime(p2_wet['TIME'],format='%H:%M:%S').dt.minute



#CREATING DATE AND TIME COLUMNS SEPARATELY FOR GENERATION DATA

p2_gen['DATE_TIME'] = pd.to_datetime(p2_gen['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')

p2_gen['DATE'] = p2_gen['DATE_TIME'].apply(lambda x:x.date())
p2_gen['TIME'] = p2_gen['DATE_TIME'].apply(lambda x:x.time())

p2_gen['HOUR'] = pd.to_datetime(p2_gen['TIME'],format='%H:%M:%S').dt.hour
p2_gen['MINUTES'] = pd.to_datetime(p2_gen['TIME'],format='%H:%M:%S').dt.minute



#GETING NEW COLUMN DATA
p2_gen.info()
print("\n"*5)
p2_wet.info()
p2_wet.groupby('DATE')['IRRADIATION'].sum()



#PLOTTING GRAPH
irddiation_sum = p2_wet.groupby('DATE')['IRRADIATION'].sum()    

unique_date = p2_wet["DATE"].unique()


fig= plt.figure(figsize=(16,9))

axes= fig.add_axes([0.1,0.1,0.8,0.8])

axes.plot(unique_date, irddiation_sum)

plt.xlabel("DATE")
plt.ylabel("TOTAL IRRADIATION")
plt.title("TOTAL IRRADIATION PER DAY")
plt.show()




print("MAXIMUM AMBIENT TEMPERATURE IS\n", p2_wet["AMBIENT_TEMPERATURE"].max())
create_block()
print("MAXIMUM MODULE TEMPERATURE IS\n", p2_wet["MODULE_TEMPERATURE"].max())
l = p2_gen['PLANT_ID'].unique() # creates a list of all unique plant id

for i in l:
    z = p2_gen[p2_gen['PLANT_ID']== i]['SOURCE_KEY'].unique()
    print("NUMBER OF INVERTERS FOR PLANT ", i, "ARE", len(z),":-\n\n")
    
    for j in z:
        print(j)
print("MAXIMUM DC POWER IN A DAY IS:-\n ", p2_gen.groupby('DATE')['DC_POWER'].max())
create_block()
print("MINIMUM DC POWER IN A DAY IS:-\n ", p2_gen.groupby('DATE')['DC_POWER'].min())




#PLOTTING GRAPH
max_dc_per_day =  p2_gen.groupby('DATE')['DC_POWER'].max()

min_dc_per_day =  p2_gen.groupby('DATE')['DC_POWER'].min()

unique_date = p2_wet["DATE"].unique()


fig= plt.figure(figsize=(16,9))

axes= fig.add_axes([0.1,0.1,0.8,0.8])

axes.plot(unique_date, max_dc_per_day, color='r', label="MAX DC OUTPUT")
axes.plot(unique_date, min_dc_per_day, color='b', label="MIN DC OUTPUT")

plt.xlabel("DATE")
plt.ylabel("DC POWER")
plt.title("MAX v/s MIN DC POWER PER DAY")

plt.legend(loc="upper left")

plt.show()
print("MAXIMUM AC POWER IN A DAY IS:-\n ", p2_gen.groupby('DATE')['AC_POWER'].max())
create_block()
print("MINIMUM AC POWER IN A DAY IS:-\n ", p2_gen.groupby('DATE')['AC_POWER'].min())

#PLOTTING GRAPH
max_ac_per_day =  p2_gen.groupby('DATE')['AC_POWER'].max()

min_ac_per_day =  p2_gen.groupby('DATE')['AC_POWER'].min()

unique_date = p2_wet["DATE"].unique()


fig= plt.figure(figsize=(16,9))

axes= fig.add_axes([0.1,0.1,0.8,0.8])

plt.xlabel("DATE")
plt.ylabel("AC POWER")
plt.title("MAX v/s MIN AC POWER PER DAY")

axes.plot(unique_date, max_ac_per_day, color='r', label="MAXIMUM AC OUTPUT")
axes.plot(unique_date, min_ac_per_day, color='b', label="MINIMUM AC OUTPUT")

plt.legend(loc="upper left")

plt.show()
max_ac_per_day =  p2_gen.groupby('DATE')['AC_POWER'].max()

min_dc_per_day =  p2_gen.groupby('DATE')['DC_POWER'].max()

unique_date = p2_wet["DATE"].unique()


fig= plt.figure(figsize=(16,9))

axes= fig.add_axes([0.1,0.1,0.8,0.8])

plt.xlabel("DATE")
plt.ylabel("OUTPUT POWER")
plt.title("MAX DC v/s AC POWER PER DAY")

axes.plot(unique_date, max_ac_per_day, color='r', label="MAXIMUM AC OUTPUT")
axes.plot(unique_date, min_dc_per_day, color='b', label="MAXIMUM DC OUTPUT")

plt.legend(loc="upper left")

plt.show()
avg_ac_per_day =  p2_gen.groupby('DATE')['AC_POWER'].mean()

avg_dc_per_day =  p2_gen.groupby('DATE')['DC_POWER'].mean()

unique_date = p2_wet["DATE"].unique()


fig= plt.figure(figsize=(16,9))

axes= fig.add_axes([0.1,0.1,0.8,0.8])

plt.xlabel("DATE")
plt.ylabel("OUTPUT POWER")
plt.title("AVERAGE DC v/s AC POWER PER DAY")

axes.plot(unique_date, avg_ac_per_day, color='r', label="AVERAGE AC OUTPUT")
axes.plot(unique_date, avg_dc_per_day, color='b', label="AVERAGE DC OUTPUT")

plt.legend(loc="upper left")

plt.show()
max_dc_inverter = p2_gen[p2_gen['DC_POWER']== p2_gen["DC_POWER"].max()]['SOURCE_KEY']
print("MAXIMUM DC POWER IS GIVEN BY INVERTER :- ", max_dc_inverter)

print("\n"*4)

max_ac_inverter = p2_gen[p2_gen['AC_POWER']== p2_gen["AC_POWER"].max()]['SOURCE_KEY']
print("MAXIMUM AC POWER IS GIVEN BY INVERTER :- ", max_ac_inverter)
def takeSecond(elem):
    return elem[1]

inv = p2_gen['SOURCE_KEY'].unique()
ac_inv =  p2_gen.groupby('SOURCE_KEY')['AC_POWER'].mean()

p_list=[]

for i in range(len(inv)):
     p_list.append([inv[i], ac_inv[i]])
    
p_list.sort(reverse = True ,key=takeSecond)

print("RANKING INVERTERS BASED ON AC OUTPUT\n\n")

for i in range(len(p_list)):
    print("RANK", i+1, ":- ##",p_list[i][0], "## AVERAGE AC OUPUT IS:-", p_list[i][1], "\n" )
def takeSecond(elem):
    return elem[1]

inv = p2_gen['SOURCE_KEY'].unique()
dc_inv =  p2_gen.groupby('SOURCE_KEY')['DC_POWER'].mean()

p_list=[]

for i in range(len(inv)):
     p_list.append([inv[i], dc_inv[i]])
    
p_list.sort(reverse = True ,key=takeSecond)

print("RANKING INVERTERS BASED ON DC OUTPUT\n\n")

for i in range(len(p_list)):
    print("RANK", i+1, ":- ##",p_list[i][0], "## AVERAGE DC OUPUT IS:-", p_list[i][1], "\n" )
print("Ideally 22 Inverters are working for 24 hours(1 day), and we are getting data every 15 min(i.e. 4 times per hour)")

print("Hence we can say ideally there are",22*24*4,"number of data everyday")
print("This is the data collected per day:-\n\n")
p2_gen['DATE'].value_counts().sort_index()

print("Now i will plot the graph for number of data per day\n")
print("And there will be one particular line which corresponds to the ideal data per day")

data_per_day = p2_gen['DATE'].value_counts().sort_index()

unique_date = p2_gen["DATE"].unique()

ideal_data = []
#CREATING IDEAL DATA LIST
for i in range(len(unique_date)):
    ideal_data.append(22*24*4)

fig= plt.figure(figsize=(16,9))

axes= fig.add_axes([0.1,0.1,0.8,0.8])

plt.xlabel("DATE")
plt.ylabel("NUMBER OF DATA")
plt.title("AVERAGE DATA PER DAY")

axes.plot(unique_date, data_per_day, color='r', label="DATA PER DAY")
axes.plot(unique_date,ideal_data , color='b', label="IDEAL DATA PER DAY")

plt.legend(loc="lower right")

plt.show()
