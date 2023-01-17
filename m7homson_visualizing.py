import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv('../input/ottawa_bike_counters.csv')
data.Date = pd.to_datetime(data.Date)
for col in data.columns:
    if col != 'Date':
        data.plot(x='Date',y=col, figsize=(32,8),)

data.describe()
### Weekday Tendancies 
import calendar
import matplotlib.pyplot as plt


sums = [0 for i in range(7)]
counts = [0 for i in range(7)]
data = data.fillna(0)

for row in range(data.shape[0]):
    day_of_week = data.loc[row,'Date'].weekday()
    for col in range(data.shape[1]):
        if col == 0: continue
        sums[day_of_week] += data.loc[row,data.columns[col]]
        counts[day_of_week] += 1
        
averages = [summ/count for summ, count in zip(sums, counts)]


for day_number in range(7):
    print(calendar.day_name[day_number], counts[day_number], ' ', sums[day_number], ' ', averages[day_number])
    
plt.figure(figsize=(10,5))
plt.bar(list(range(7)),averages)
plt.xticks(list(range(7)), [calendar.day_name[day] for day in range(7)])
plt.show()
### Monthly Tendancies 
import calendar
import matplotlib.pyplot as plt


sums = [0 for i in range(12)]
counts = [0 for i in range(12)]
data = data.fillna(0)

for row in range(data.shape[0]):
    month_of_year = data.loc[row,'Date'].month - 1
    for col in range(data.shape[1]):
        if col == 0: continue
        sums[month_of_year] += data.loc[row,data.columns[col]]
        counts[month_of_year] += 1
        
averages = [summ/count for summ, count in zip(sums, counts)]

for month_of_year in range(12):
    print(calendar.month_name[month_of_year + 1], counts[month_of_year], ' ', sums[month_of_year], ' ', averages[month_of_year])
    
plt.figure(figsize=(15,5))
plt.bar(list(range(12)),averages)
plt.xticks(list(range(12)), [calendar.month_name[month + 1] for month in range(12)])
plt.show()