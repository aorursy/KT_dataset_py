from matplotlib import pyplot as plt
from matplotlib import style
style.use('ggplot')
import seaborn as sns
%matplotlib inline


import pandas as pd
import numpy as np
df = pd.read_csv('../input/waterbody-classifications.csv')
df.head()
df.shape
water_type = df.groupby('Waterbody Type')
for typ,group in water_type:
    print(typ)
#Let's find out how many...

Estuary = df[df['Waterbody Type'] == 'Estuary']['Waterbody Type'].count()
Ponds = df[df['Waterbody Type'] == 'Ponds']['Waterbody Type'].count()
Shoreline = df[df['Waterbody Type'] == 'Estuary']['Waterbody Type'].count()
Streams = df[df['Waterbody Type'] == 'Streams']['Waterbody Type'].count()

print('Estuary:',Estuary)
print('Ponds:',Ponds)
print('Shoreline:',Shoreline)
print('Streams:',Streams)
##Lets visualize it with a bar chart.

plt.figure(figsize=(15,8))  #setting the size
sns.barplot(x=['Streams','Ponds','Estuary','Shoreline'],y=[Streams,Ponds,Estuary,Shoreline])
#Max Segment Miles
df[df['Segment Miles'] == df['Segment Miles'].max()]

#Min Segment Miles
df[df['Segment Miles'] == df['Segment Miles'].min()]
basin = df.groupby('Basin')
for basin_n,basin_g in basin:
    print(basin_n)
df = df.replace(to_replace='Atlantic Ocean/Long Island Soun',value='Atlantic Ocean/Long Island Sound')
#Lets try again!

basin = df.groupby('Basin')
for basin_n,basin_g in basin:
    print(basin_n)
AR = basin.get_group('Allegheny River')

Estuary = AR[AR['Waterbody Type'] == 'Estuary']['Waterbody Type'].count()
Ponds = AR[AR['Waterbody Type'] == 'Ponds']['Waterbody Type'].count()
Shoreline = AR[AR['Waterbody Type'] == 'Estuary']['Waterbody Type'].count()
Streams = AR[AR['Waterbody Type'] == 'Streams']['Waterbody Type'].count()

print('Estuary:',Estuary)
print('Ponds:',Ponds)
print('Shoreline:',Shoreline)
print('Streams:',Streams)
plt.figure(figsize=(15,8))
sns.barplot(x=['Streams','Ponds','Estuary','Shoreline'],y=[Streams,Ponds,Estuary,Shoreline])
waterquality = df.groupby('Water Quality Class')

#Fresh.
b1 = waterquality.get_group('A')['Name'].count()
b2 = waterquality.get_group('AA')['Name'].count()
b3 = waterquality.get_group('A-S')['Name'].count()
b4 = waterquality.get_group('AA-S')['Name'].count()
b5 = waterquality.get_group('B')['Name'].count()
b6 = waterquality.get_group('C')['Name'].count()
b7 = waterquality.get_group('D')['Name'].count()

#Saline
s1 = waterquality.get_group('SA')['Name'].count()
s2 = waterquality.get_group('SB')['Name'].count()
s3 = waterquality.get_group('SC')['Name'].count()
s4 = waterquality.get_group('I')['Name'].count()
s5 = waterquality.get_group('SD')['Name'].count()

print(b1,b2,b3,b4,b5,b6,b7)
print(s1,s2,s3,s4,s5)

sum_of_fresh = b1+b2+b3+b4+b5+b6+b7
sum_of_saline = s1+s2+s3+s4+s5

plt.figure(figsize=(15,8))
sns.barplot(x=['Fresh','Saline'],y=[sum_of_fresh,sum_of_saline],palette="rocket")