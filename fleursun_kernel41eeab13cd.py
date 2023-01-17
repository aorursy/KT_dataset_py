# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

plt.style.use('seaborn')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



%matplotlib inline       

# Any results you write to the current directory are saved as output.
#import the data and do some inspection

data_uncln = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')

data_uncln.head()
#To more cleaner about the data, we may show the all columns to know if there are some features in the data.

data_uncln.columns
data_uncln.describe()
#check the missing values

data_uncln.isnull().sum()



#we can know about "country", "agent", "company" labels have the missing values.
#Replace missing values:

#use '0' to replace "agent" and "company", and 'Unknown' to "counrty"

nan_replacements = {"children": 0, "country" : "Unknown", "agent" : 0, "company" : 0}

data_cln = data_uncln.fillna(nan_replacements)





#there are some conditions to find the actual information about the guest who at least has one person.

invaild_guests = list(data_cln.loc[data_cln["adults"] + data_cln["children"] + data_cln["babies"] == 0].index)

data_cln.drop(data_cln.index[invaild_guests], inplace=True)



#abandon 180 invaild samples

len(invaild_guests)
#hotel style and arrival year can reflect the trend of lodging. 

#City hotel = 79330, Resort hotel = 40060 



hotel_style = data_cln['hotel'].value_counts()

arrival_year = data_cln['arrival_date_year'].value_counts()



f, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15,7))

axes = ax.flatten()



plt.subplot(1, 2, 1)



x, y = hotel_style.index, hotel_style.values

y_ = hotel_style.values / (hotel_style.values.sum())

#plt.ylim(0, 1.0)



plt.xlabel('Hotel style', fontsize = '14')

plt.ylabel('Count', fontsize = '14')



plt.bar(x = x, height = y, width =0.35, align = 'center', color = 'blue', alpha = 0.5);

plt.title('Hotel solution', fontsize= '20')





plt.subplot(1, 2, 2)

ay = arrival_year.sort_index(ascending = True)

x, y = ay.index, ay.values

plt.xlabel('Arrival year', fontsize = '14')



plt.ylabel('Number of visitors', fontsize = '14')



plt.bar(x = x, height = y, width = 0.4, align = 'center', color = 'gray', alpha = 0.5);

plt.title('Number with arrival year', fontsize = '20')

#now, to analyse what's the most hot time to travel, we will use the following features

#it contains the "is_canceled":

f, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (27, 7))

plt.subplot(1, 2, 1)



week_num = data_cln["arrival_date_week_number"].value_counts().sort_index(ascending = True) #week-number

x, y = week_num.index, week_num.values



# calculate the average number of guests every four weeks

temp_week_num = []

total_num = []

total, max_n = 0.0, 0

for j in x :

    temp_week_num.append(y[j-1])

    if j < len(y)-2:

        for k in np.arange(1, 4):

            temp_week_num.append(y[j-1+k]) 

        #print(temp_week_num)

        for m in np.arange(0, 4):

            

            total += temp_week_num[m]

        temp_week_num = []

        total_num.append(total)

        total = 0

        #print(total_num)

for item in range(len(total_num)):

    total_num[item] /= 4

    

plt.xlabel("week (th)", fontsize = '14')

plt.ylabel("number of arrive",)

plt.yticks(rotation = 45)



plt.bar(x = x, height = y, width = 0.5, align = 'center', label = "num of guests per week", color = 'blue', alpha = 0.3);

plt.plot(np.arange(3,53), total_num, 'r|-', label = 'average every four weeks');

plt.title("Arrive time", fontsize = '20' )

plt.legend(loc = 0, fontsize = '14' )





a = total_num.index(max(total_num))

print("The maximum of guests with every four weeks appears between {0}th and {1}th. Here we may meet the peak value of guests.".format(a, a+3))


