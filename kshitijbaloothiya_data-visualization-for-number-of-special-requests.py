# In this notebook I tried to visualize the data for the cases where special requests were made by the customers 

import numpy as np 
import pandas as pd 



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
hotel = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")
SR = hotel['total_of_special_requests'].tolist()
import matplotlib.pyplot as plt


#DATA VISUALIZATION IN TERMS OF BAR GRAPH

#1. WHETHER THE HOTEL IS A RESORT OR CITY HOTEL
hot = hotel['hotel'].tolist() #IMPORTING THE HOTEL DATA
hot1 = 0
hot2 = 0
for i in range(0,len(hot)):
    if hot[i]=='Resort Hotel':
        if SR[i]!=0:
            hot1 = hot1 + 1
    else:
        if SR[i]!=0:
            hot2 = hot2 + 1
ar = [hot1, hot2]
lab = ['resort hotel', 'city hotel']
ypos=np.arange(len(lab))
plt.xticks(ypos, lab)
plt.ylabel('Number of special request ordered')
plt.bar(ypos, ar)
plt.show()
# DATA VISUALIZATION WITH BAR GRAPH

# 2. With respect to number of week nights stayed
days = hotel['stays_in_week_nights'].tolist()
day1 = 0
day2 = 0
day3 = 0
day4 = 0 
for i in range(0,len(days)):
    if (days[i]>=0 & days[i]<3 & SR[i]!=0):
        day1 = day1 + 1  
    elif days[i]>=3 & days[i]<6 & SR[i]!=0:
        day2 = day2 + 1
    elif days[i]>=6 & days[i]<10 & SR[i]!=0:
        day3 = day3 + 1
    else:
        day4 = day4 + 1

ct1 = [day1, day2, day3, day4]
import matplotlib.pyplot as plt
lab1 = ['0-2','3-5','6-9','>=10']
ypos1 = np.arange(len(lab1))
plt.xticks(ypos1, lab1)
plt.ylabel('Number of special services requested')
plt.bar(ypos1, ct1)
plt.show()

# DATA VISUALIZATION WITH BAR GRAPH

# 3.adults with babies and/or children
aud = hotel['adults'].tolist()
chi = hotel['children'].tolist()
bab = hotel['babies'].tolist()
ctr1 = 0
ctr2 = 0
ctr3 = 0
ctr4 = 0
for i in range(0,len(aud)):
    if chi[i]==0 & SR[i]==0:
        if bab[i] == 0:
            ctr2 = ctr2 +1
        elif bab[i]>0:
            ctr4 = ctr4 + 1
    elif chi[i]>0 & SR[i]==0:
        if bab[i] == 0:
            ctr1 = ctr1 + 1
        elif bab[i]>0:
            ctr3 = ctr3 + 1


ct2 = [ctr1, ctr2, ctr3, ctr4]
lab2 = ['child only', 'adult only', 'both', 'babies only']
ypos2 = np.arange(len(lab2))
plt.xticks(ypos2, lab2)
plt.ylabel('Number of special services not requested')
plt.bar(ypos2, ct2)
plt.show()
#DATA VISUALIZATION WITH CONTINOUS GRAPH

#4. Based on country of guests
cou = hotel['country'].tolist()
cou_sr = []

for i in range(0,len(cou)):
    if SR[i]!=0:
        cou_sr.append(cou[i])
       
df1 = pd.Index(cou_sr)
count1 = df1.value_counts()

count1.plot()
#DATA VISUALIZATION WITH PLOTTING

# 5. Based on customer type
ct = hotel['customer_type'].tolist()
ct_sr = []
for i in range(0,len(ct)):
    if SR[i]!=0:
        ct_sr.append(ct[i])
df2 = pd.Index(ct_sr)
count2 = df2.value_counts()
count2.plot()
#DATA VISUALIZATION WITH BAR GRAPH

# 6.Based on required car parking space
park = hotel['required_car_parking_spaces'].tolist()
ctr1 = 0
ctr2 = 0
for i in range(0, len(park)):
    if SR[i]!=0:
        if park[i]!=0:
            ctr1 = ctr1 + 1
    elif SR[i]==0:
        if park[i]==0:
            ctr2 = ctr2 + 1
lab3 = ['Parking required', 'No parking required']

ctr = [ctr1, ctr2]
print(ctr)
ypos3 = np.arange(len(lab3))
plt.xticks(ypos3, lab3)
plt.ylabel('Number of services demanded')
plt.bar(ypos3, ctr)