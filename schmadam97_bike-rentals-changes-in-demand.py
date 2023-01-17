import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

day_df = pd.read_csv('/kaggle/input/bike-sharing-dataset/day.csv')
hour_df = pd.read_csv('/kaggle/input/bike-sharing-dataset/hour.csv')
day_df.head()
plt.plot(day_df.dteday, day_df.casual, color='black', linewidth=.7, linestyle=':')
plt.plot(day_df.dteday, day_df.casual.rolling(window=14).mean(), color='blue', label='14-Day Rolling Average - Casual')

plt.ylabel('Users')

plt.plot(day_df.dteday, day_df.registered, color='black', linewidth=.7, linestyle=':')
plt.plot(day_df.dteday, day_df.registered.rolling(window=14).mean(), color='red', label='14-Day Rolling Average - Registered')
plt.title('Registered and Casual Users (2011-2012)')
plt.ylim([0,7500])
plt.legend()

plt.xticks([i for i in range(len(day_df.dteday)) if i % 45 == 0 ],[day_df.dteday.tolist()[i].split('-')[1] + '/' + day_df.dteday.tolist()[i].split('-')[2] + '/' + day_df.dteday.tolist()[i].split('-')[0][2:] for i in range(len(day_df.dteday)) if i % 45 == 0], rotation=70)

plt.show()
day_df['casual_14MA'] = day_df.casual.rolling(window=14).mean()
day_df['registered_14MA'] = day_df.registered.rolling(window=14).mean()
day_df['cnt_14MA'] = day_df.cnt.rolling(window=14).mean()
day_df['casual_14MA_diff'] = day_df.casual/day_df['casual_14MA'] - 1
day_df['registered_14MA_diff'] = day_df.registered/day_df['registered_14MA'] - 1
day_df['cnt_14MA_diff'] = day_df.cnt/day_df['cnt_14MA'] - 1
print('Weekday')
print('All: ' + str(day_df[day_df.workingday==1].cnt.mean()))
print('Casual: ' + str(day_df[day_df.workingday==1].casual.mean()))
print('Registered: ' + str(day_df[day_df.workingday==1].registered.mean()))
print('Weekend')
print('All: ' + str(round(day_df[day_df.workingday==0].cnt.mean(),2)))
print('Casual: ' + str(round(day_df[day_df.workingday==0].casual.mean(),2)))
print('Registered: ' + str(round(day_df[day_df.workingday==0].registered.mean(),2)))
data = [(day_df[day_df.workingday==0].casual.mean()/day_df[day_df.workingday==1].casual.mean()-1)*100, \
        (day_df[day_df.workingday==0].registered.mean()/day_df[day_df.workingday==1].registered.mean()-1)*100, \
        (day_df[day_df.workingday==0].cnt.mean()/day_df[day_df.workingday==1].cnt.mean()-1)*100]
positive_vals = [x if x > 0 else 0 for x in data]
negative_vals = [x if x < 0 else 0 for x in data]
plt.bar(['Casual','Registered','All'], positive_vals, color="forestgreen")
plt.bar(['Casual','Registered','All'], negative_vals, color="red")
plt.axhline(y=0, color='black', linestyle="--", linewidth=.9)
plt.ylim([-50,150])
for i in range(len(positive_vals)):
    if positive_vals[i] != 0:
        plt.text(i, positive_vals[i]+5, ha='center', s=str(round(positive_vals[i],2)) + '%')
for i in range(len(negative_vals)):
    if negative_vals[i] != 0:
        plt.text(i, negative_vals[i]-10, ha='center', s=str(round(negative_vals[i],2)) + '%')
plt.ylabel('Average Change (%)', fontsize=15)
plt.xlabel('Type of User', fontsize=15)
plt.title('Going into the Weekend:\nChange in Bike Rentals by Type of User', fontsize=15)
data = [(day_df[day_df.workingday==1].casual.mean()/day_df[day_df.workingday==0].casual.mean()-1)*100, \
        (day_df[day_df.workingday==1].registered.mean()/day_df[day_df.workingday==0].registered.mean()-1)*100, \
        (day_df[day_df.workingday==1].cnt.mean()/day_df[day_df.workingday==0].cnt.mean()-1)*100]
positive_vals = [x if x > 0 else 0 for x in data]
negative_vals = [x if x < 0 else 0 for x in data]
plt.bar(['Casual','Registered','All'], positive_vals, color="forestgreen")
plt.bar(['Casual','Registered','All'], negative_vals, color="red")
plt.axhline(y=0, color='black', linestyle="--", linewidth=.9)
plt.ylim([-70,50])
for i in range(len(positive_vals)):
    if positive_vals[i] != 0:
        plt.text(i, positive_vals[i]+5, ha='center', s=str(round(positive_vals[i],2)) + '%')
for i in range(len(negative_vals)):
    if negative_vals[i] != 0:
        plt.text(i, negative_vals[i]-10, ha='center', s=str(round(negative_vals[i],2)) + '%')
plt.ylabel('Average Change (%)', fontsize=15)
plt.xlabel('Type of User', fontsize=15)
plt.title('Going into the Workweek:\nChange in Bike Rentals by Type of User', fontsize=15)
hr_df = hour_df[(hour_df.workingday==1)|(hour_df.holiday==0)].groupby('hr').cnt.mean().reset_index(name='cnt_count')
hr_df = pd.merge(hr_df, hour_df[(hour_df.workingday==1)|(hour_df.holiday==0)].groupby('hr').registered.mean().reset_index(name='reg_count'), on='hr')
hr_df = pd.merge(hr_df, hour_df[(hour_df.workingday==1)|(hour_df.holiday==0)].groupby('hr').casual.mean().reset_index(name='cas_count'), on='hr')
width = .3
plt.plot(hr_df.hr, hr_df.cnt_count, label='All')
plt.plot(hr_df.hr, hr_df.reg_count, label='Registered')
plt.plot(hr_df.hr, hr_df.cas_count, label='Casual', color='red')
plt.legend()
plt.xlabel('Hour of Day')
plt.ylabel('Users')
plt.title('Users by Hour of Day, Workday')
hr_df = hour_df[(hour_df.workingday==0)|(hour_df.holiday==1)].groupby('hr').cnt.mean().reset_index(name='cnt_count')
hr_df = pd.merge(hr_df, hour_df[(hour_df.workingday==0)|(hour_df.holiday==1)].groupby('hr').registered.mean().reset_index(name='reg_count'), on='hr')
hr_df = pd.merge(hr_df, hour_df[(hour_df.workingday==0)|(hour_df.holiday==1)].groupby('hr').casual.mean().reset_index(name='cas_count'), on='hr')
width = .3
plt.plot(hr_df.hr, hr_df.cnt_count, label='All')
plt.plot(hr_df.hr, hr_df.reg_count, label='Registered')
plt.plot(hr_df.hr, hr_df.cas_count, label='Casual', color='red')
plt.legend()
plt.xlabel('Hour of Day')
plt.ylabel('Users')
plt.title('Users by Hour of Day, Weekend')
width = .3
labels = [1,2,3]

data = [day_df[(day_df.weathersit==1)].casual.mean()-day_df.casual.mean(), \
        day_df[(day_df.weathersit==2)].casual.mean()-day_df.casual.mean(), \
        day_df[(day_df.weathersit==3)].casual.mean()-day_df.casual.mean()]
positive_vals = [x if x > 0 else 0 for x in data]
negative_vals = [x if x < 0 else 0 for x in data]
plt.bar([x-width for x in labels], positive_vals, color='blue',width = width)
plt.bar([x-width for x in labels], negative_vals, color='blue',width = width, label='Casual Users')
for i in range(len(positive_vals)):
    if positive_vals[i] != 0:
        plt.text(i+1-width, positive_vals[i]+50, ha='center', s=str(round(positive_vals[i],2)), fontsize=7)
for i in range(len(negative_vals)):
    if negative_vals[i] != 0:
        plt.text(i+1-width, negative_vals[i]-120, ha='center', s=str(round(negative_vals[i],2)), fontsize=7)

data = [day_df[(day_df.weathersit==1)].registered.mean()-day_df.registered.mean(), \
        day_df[(day_df.weathersit==2)].registered.mean()-day_df.registered.mean(), \
        day_df[(day_df.weathersit==3)].registered.mean()-day_df.registered.mean()]
positive_vals = [x if x > 0 else 0 for x in data]
negative_vals = [x if x < 0 else 0 for x in data]
plt.bar([x for x in labels], positive_vals, color='red',width = width)
plt.bar([x for x in labels], negative_vals, color='red',width = width, label='Registered Users')
for i in range(len(positive_vals)):
    if positive_vals[i] != 0:
        plt.text(i+1, positive_vals[i]+50, ha='center', s=str(round(positive_vals[i],2)), fontsize=7)
for i in range(len(negative_vals)):
    if negative_vals[i] != 0:
        plt.text(i+1, negative_vals[i]-120, ha='center', s=str(round(negative_vals[i],2)), fontsize=7)
        
data = [day_df[(day_df.weathersit==1)].cnt.mean()-day_df.cnt.mean(), \
        day_df[(day_df.weathersit==2)].cnt.mean()-day_df.cnt.mean(), \
        day_df[(day_df.weathersit==3)].cnt.mean()-day_df.cnt.mean()]
positive_vals = [x if x > 0 else 0 for x in data]
negative_vals = [x if x < 0 else 0 for x in data]
plt.bar([x + width for x in labels], positive_vals, color='green',width = width)
plt.bar([x + width for x in labels], negative_vals, color='green',width = width, label='All Users')
for i in range(len(positive_vals)):
    if positive_vals[i] != 0:
        plt.text(i+1 + width, positive_vals[i]+50, ha='center', s=str(round(positive_vals[i],2)), fontsize=7)
for i in range(len(negative_vals)):
    if negative_vals[i] != 0:
        plt.text(i+1 + width, negative_vals[i]-120, ha='center', s=str(round(negative_vals[i],2)), fontsize=7)
    

plt.axhline(y=0, color='black', linestyle="--", linewidth=.9)

plt.title('All Users: Impact of Weather on Bike Rentals on the Weekend')
plt.ylabel('Total Change')
plt.xlabel('Weather Situation')
plt.xticks([1,2,3])
plt.ylim([-3200,700])
plt.legend()
width = .3
labels = [1,2,3]

data = [day_df[(day_df.weathersit==1)].casual.mean()/day_df.casual.mean()-1, \
        day_df[(day_df.weathersit==2)].casual.mean()/day_df.casual.mean()-1, \
        day_df[(day_df.weathersit==3)].casual.mean()/day_df.casual.mean()-1]
positive_vals = [x if x > 0 else 0 for x in data]
negative_vals = [x if x < 0 else 0 for x in data]
plt.bar([x-width for x in labels], positive_vals, color='blue',width = width)
plt.bar([x-width for x in labels], negative_vals, color='blue',width = width, label='Casual Users')
for i in range(len(positive_vals)):
    if positive_vals[i] != 0:
        plt.text(i+1-width, positive_vals[i]+.10, ha='center', s=str(round(positive_vals[i]*100,1)) + '%', fontsize=8)
for i in range(len(negative_vals)):
    if negative_vals[i] != 0:
        plt.text(i+1-width, negative_vals[i]-.10, ha='center', s=str(round(negative_vals[i]*100,1)) + "%", fontsize=8)

data = [day_df[(day_df.weathersit==1)].registered.mean()/day_df.registered.mean()-1, \
        day_df[(day_df.weathersit==2)].registered.mean()/day_df.registered.mean()-1, \
        day_df[(day_df.weathersit==3)].registered.mean()/day_df.registered.mean()-1]
positive_vals = [x if x > 0 else 0 for x in data]
negative_vals = [x if x < 0 else 0 for x in data]
plt.bar([x for x in labels], positive_vals, color='red',width = width)
plt.bar([x for x in labels], negative_vals, color='red',width = width, label='Registered Users')
for i in range(len(positive_vals)):
    if positive_vals[i] != 0:
        plt.text(i+1, positive_vals[i]+.10, ha='center', s=str(round(positive_vals[i]*100,1)) + '%', fontsize=8)
for i in range(len(negative_vals)):
    if negative_vals[i] != 0:
        plt.text(i+1, negative_vals[i]-.10, ha='center', s=str(round(negative_vals[i]*100,1)) + "%", fontsize=8)
        
data = [day_df[(day_df.weathersit==1)].cnt.mean()/day_df.cnt.mean()-1, \
        day_df[(day_df.weathersit==2)].cnt.mean()/day_df.cnt.mean()-1, \
        day_df[(day_df.weathersit==3)].cnt.mean()/day_df.cnt.mean()-1]
positive_vals = [x if x > 0 else 0 for x in data]
negative_vals = [x if x < 0 else 0 for x in data]
plt.bar([x + width for x in labels], positive_vals, color='green',width = width)
plt.bar([x + width for x in labels], negative_vals, color='green',width = width, label='All Users')
for i in range(len(positive_vals)):
    if positive_vals[i] != 0:
        plt.text(i+1 + width, positive_vals[i]+.10, ha='center', s=str(round(positive_vals[i]*100,1)) + '%', fontsize=8)
for i in range(len(negative_vals)):
    if negative_vals[i] != 0:
        plt.text(i+1 + width, negative_vals[i]-.10, ha='center', s=str(round(negative_vals[i]*100,1)) + "%", fontsize=8)
    

plt.axhline(y=0, color='black', linestyle="--", linewidth=.9)

plt.title('All Users: Impact of Weather on Bike Rentals on the Weekend')
plt.ylabel('Percent Change')
plt.xlabel('Weather Situation')
plt.xticks([1,2,3])
plt.ylim([-1,1])
plt.legend()