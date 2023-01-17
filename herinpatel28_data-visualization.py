import numpy as np 
import pandas as pd 

from subprocess import check_output
stations = pd.read_csv('../input/station.csv')
trips = pd.read_csv('../input/trip.csv', error_bad_lines=False)

weathers = pd.read_csv('../input/weather.csv')
#display stations first 5 records
stations.head(5)


stations.describe()
#display trips first 5 records
trips.head(5)


trips.describe()
#display weathers first 5 records
weathers.head(5)


weathers.describe()
#print stations total Record,Attribute 
print("Record,Attribute : ",stations.shape)
#print trips total Records & Attributes 
print("Record,Attribute : ",trips.shape)
#print weathers total Records & Attributes 
print("Record,Attribute : ",weathers.shape)
trips.hist()
plt.show()
import matplotlib.pyplot as plt

trips.gender

from collections import defaultdict

# count genders
counts = defaultdict(int)
for e in trips.gender:
    counts[e] += 1
# Gender distribution
y = [e[1] for e in counts.items()][:4]
x = range(len(y))
cols = [e[0] for e in counts.items()][:4]
    
plt.figure(figsize=(8,5))
plt.bar(x, y, color='red', width=1/1.5)
plt.xticks(x, cols)
plt.title('Gender Distribution')
plt.show()
# Bikes with trip duration

#print(trips.birthyear)

bike_stats = defaultdict(int)
dur_stats = defaultdict(float)
by_stats = defaultdict(int)

for t in trips.itertuples():
    bike_stats[t.bikeid] += 1
    dur_stats[t.bikeid] += t.tripduration
    by_stats[str(t.birthyear)] += 1
    #break
    
#for bid in dur_stats.keys():
#    dur_stats[bid] /= bike_stats[bid]
    
    
#for e, k in zip(dur_stats, bike_stats):
    #print(dur_stats[k], " ", bike_stats[e])
    
print(len(by_stats.keys()))
#print(by_stats)
keys = sorted(by_stats.keys())[:-1] # without nan


# Age distribution

y = [by_stats[k] for k in keys]

cols = [by_stats[k] for k in keys]
    
    
minyear = int(min(keys)[:-2])
maxyear = int(max(keys)[:-2])
print(minyear, maxyear)
minamp=0
maxamp=40

x = range(minyear, maxyear, 50)
x = range(minyear, maxyear)

x = x[:61]

# evalutate with histogram
plt.figure(figsize=(14,5))
plt.bar(x, y, color='blue', width=1/1.5)
#plt.hist(cols, 10)
plt.xticks(x, [e for e in x])
#plt.axis([minyear, maxyear, minamp, maxamp])
plt.title('Age Distribution')
plt.show()

