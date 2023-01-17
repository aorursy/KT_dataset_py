import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

uber_drives = pd.read_csv('../input/My Uber Drives - 2016.csv')

uber_drives.head(5)
# Get the initial data with dropping the data

uber_drives.describe()

uber_drives = uber_drives.dropna()
# Get the starting destination, unique destination

start_destination = uber_drives['START*'].dropna()

unique_start = set(start_destination)

print (len(unique_start))
# Get the stop destination, unique destination

stop_destination = uber_drives['STOP*'].dropna()

unique_stop = set(stop_destination)

print (len(unique_stop))
# Dataframe for starting and stop date



start_date = uber_drives['START_DATE*']

stop_date = uber_drives['END_DATE*']

print (start_date.head(5))
# Splitting date into date and time and assignining the respective values



start_time = []

start_date = []



for i in uber_drives['START_DATE*']:

    start_time.append(i.split(' ')[1])

    start_date.append(i.split(' ')[0])



end_time = []

end_date = []



for i in uber_drives['END_DATE*']:

    end_time.append(i.split(' ')[1])

    end_date.append(i.split(' ')[0])

# Converting the time and date into the dataframe 



start_time = pd.Series(start_time)

start_date = pd.Series(start_date)

end_time = pd.Series(end_time)

end_date = pd.Series(end_date)

# Length of all the time and date is equal 

print (len(start_time), len(start_date), len(end_time),len(end_date))
# Getting the categories under the driving 



categories = uber_drives['CATEGORY*']

unique_categories = set(categories)

unique_categories
# Segregating the unique categories



total_business_trip = uber_drives[uber_drives['CATEGORY*'] == 'Business']

total_personal_trip = uber_drives[uber_drives['CATEGORY*'] == 'Personal']



print (len(total_business_trip))

print (len(total_personal_trip))
# Getting the purpose of visit

purpose_of_visit = uber_drives['PURPOSE*']

print (set(purpose_of_visit))



Airport = uber_drives[uber_drives['PURPOSE*'] == 'Airport/Travel']

between_offices = uber_drives[uber_drives['PURPOSE*'] == 'Between Offices']

charity = uber_drives[uber_drives['PURPOSE*'] == 'Charity ($)']

commute = uber_drives[uber_drives['PURPOSE*'] == 'Temporary Site']

supplies = uber_drives[uber_drives['PURPOSE*'] == 'Errand/Supplies']

moving = uber_drives[uber_drives['PURPOSE*'] == 'Moving']

entertain = uber_drives[uber_drives['PURPOSE*'] == 'Meal/Entertain']

meeting = uber_drives[uber_drives['PURPOSE*'] == 'Meeting']

customer_visit = uber_drives[uber_drives['PURPOSE*'] == 'Customer Visit']



print (len(Airport), len(between_offices), len(charity))

print (len(commute), len(supplies),len(moving))

print (len(entertain), len(meeting), len(customer_visit))
# break up of purpose of visit



df = pd.DataFrame([len(Airport), len(between_offices), len(charity), len(commute), len(supplies),len(moving),

                  len(entertain), len(meeting), len(customer_visit)], index =['Airport/Travel','Between Offices','Charity ($)',

                                                                            'Temporary Site','Errand/Supplies','Moving',

                                                                            'Meal/Entertain','Meeting','Customer Visit'],

                                                                     columns = pd.Index(['Count']))

print (df)



df.plot(kind='bar')

plt.show()
# Plotting for Category of Visit



df = pd.DataFrame([len(total_business_trip), len(total_personal_trip)], index = ['business', 'personal'], columns = pd.Index(['Count']))



print (df)

df.plot(kind='bar')

plt.show()



# It seems the guy is not software engineer, as being on business trip and spent good amount on the entertainment or the guy is 

# very excitement loving


#pd.to_datetime(start_date).month # Get the Day/Month/Year from the Series

#pd.to_datetime(uber_drives['START_DATE*']).dt.month # Get the Day/Month/Year from the DataFrame



Jan = uber_drives[pd.to_datetime(uber_drives['START_DATE*']).dt.month == 1]

Feb = uber_drives[pd.to_datetime(uber_drives['START_DATE*']).dt.month == 2]

Mar = uber_drives[pd.to_datetime(uber_drives['START_DATE*']).dt.month == 3]

Apr = uber_drives[pd.to_datetime(uber_drives['START_DATE*']).dt.month == 4]

May = uber_drives[pd.to_datetime(uber_drives['START_DATE*']).dt.month == 5]

Jun = uber_drives[pd.to_datetime(uber_drives['START_DATE*']).dt.month == 6]

Jul = uber_drives[pd.to_datetime(uber_drives['START_DATE*']).dt.month == 7]

Aug = uber_drives[pd.to_datetime(uber_drives['START_DATE*']).dt.month == 8]

Sep = uber_drives[pd.to_datetime(uber_drives['START_DATE*']).dt.month == 9]

Oct = uber_drives[pd.to_datetime(uber_drives['START_DATE*']).dt.month == 10]

Nov = uber_drives[pd.to_datetime(uber_drives['START_DATE*']).dt.month == 11]

Dec = uber_drives[pd.to_datetime(uber_drives['START_DATE*']).dt.month == 12]
Jan.loc[:,'day'] = pd.to_datetime(Jan['START_DATE*']).dt.day

Feb.loc[:,'day'] = pd.to_datetime(Feb['START_DATE*']).dt.day

Mar.loc[:,'day'] = pd.to_datetime(Mar['START_DATE*']).dt.day

Apr.loc[:,'day'] = pd.to_datetime(Apr['START_DATE*']).dt.day

May.loc[:,'day'] = pd.to_datetime(May['START_DATE*']).dt.day

Jun.loc[:,'day'] = pd.to_datetime(Jun['START_DATE*']).dt.day

Jul.loc[:,'day'] = pd.to_datetime(Jul['START_DATE*']).dt.day

Aug.loc[:,'day'] = pd.to_datetime(Aug['START_DATE*']).dt.day

Sep.loc[:,'day'] = pd.to_datetime(Sep['START_DATE*']).dt.day

Oct.loc[:,'day'] = pd.to_datetime(Oct['START_DATE*']).dt.day

Nov.loc[:,'day'] = pd.to_datetime(Nov['START_DATE*']).dt.day

Dec.loc[:,'day'] = pd.to_datetime(Dec['START_DATE*']).dt.day



Jan_group = Jan.groupby(['day']).agg('sum')

Feb_group = Feb.groupby(['day']).agg('sum')

Mar_group = Mar.groupby(['day']).agg('sum')

Apr_group = Apr.groupby(['day']).agg('sum')

May_group = May.groupby(['day']).agg('sum')

Jun_group = Jun.groupby(['day']).agg('sum')

Jul_group = Jul.groupby(['day']).agg('sum')

Aug_group = Aug.groupby(['day']).agg('sum')

Sep_group = Sep.groupby(['day']).agg('sum')

Oct_group = Oct.groupby(['day']).agg('sum')

Nov_group = Nov.groupby(['day']).agg('sum')

Dec_group = Dec.groupby(['day']).agg('sum')



#Jan_group.plot()

#plt.show()

miles_day_frame = pd.concat([Jan_group, Feb_group,Mar_group,Apr_group,May_group,Jun_group,Jul_group,Aug_group,Sep_group,Oct_group,

               Nov_group,Dec_group],ignore_index=True, axis=1)

miles_day_frame.columns = ['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

miles_day_frame.fillna(0,inplace=True)
miles_day_frame.plot()

plt.show()
# Miles Range



range_ = ["<5","5-10","10-20","20-35",">35"]



dict_range = dict()

for x in range_:

    dict_range[x] = 0



for i in uber_drives['MILES*']:

    if i < 5:

        dict_range["<5"] += 1

    elif i < 10:

        dict_range["5-10"] += 1

    elif i < 20:

        dict_range["10-20"] += 1

    elif i < 35:

        dict_range["20-35"] += 1

    else:

        dict_range[">35"] += 1

    

miles = pd.Series(dict_range)

miles.sort_values(inplace=True, ascending=False)

X = plt.bar(range(1,len(miles.index)+1),miles.values)

plt.title("Miles")

plt.xlabel("Miles")

plt.ylabel("Count of Trips")

plt.xticks(range(1,len(miles.index)+1),miles.index)

plt.show()       



from collections import OrderedDict





hour_ = [i.split(":")[0] for i in start_time]

range_ = ["<7","7-9","9-12","12-15","15-18","18-20","20-24"]

dict_time = OrderedDict()

for x in range_:

    dict_time[x] = 0



for i in hour_:

    if int(i) < 7:

        dict_time["<7"] += 1

    elif int(i) < 9:

        dict_time["7-9"] += 1

    elif int(i) < 12:

        dict_time["9-12"] += 1

    elif int(i) < 15:

        dict_time["12-15"] += 1

    elif int(i) < 18:

        dict_time["15-18"] += 1

    elif int(i) < 20:

        dict_time["18-20"] += 1

    else:

        dict_time["20-24"] += 1

    

time_ = pd.Series(dict_time)

plt.bar(range(1,len(time_.index)+1),time_.values)

plt.title("Time")

plt.xlabel("Time")

plt.ylabel("Count of Trips")

plt.xticks(range(1,len(time_.index)+1),time_.index)

plt.show()     

# Proportion of distance spent for each activities



purpose = set(uber_drives['PURPOSE*'])

total_dist = []



for i in purpose:

    temp = uber_drives[uber_drives['PURPOSE*'] == i]

    total_dist.append(sum(temp['MILES*']))



df = pd.DataFrame(total_dist, index = purpose, columns = pd.Index(['Distance']))

df.plot(kind='bar')

plt.show()
# Proportion of No. of visit for each activities

total_visit = []



for i in purpose:

    temp = uber_drives[uber_drives['PURPOSE*'] == i]

    total_visit.append(len(temp['MILES*']))



df = pd.DataFrame(total_visit, index = purpose, columns = pd.Index(['Distance']))

df.plot(kind='bar')

plt.show()

location_start = []

for i in unique_start:

    temp = uber_drives[uber_drives['START*'] == i]

    location_start.append(len(temp))

      

df = pd.DataFrame(location_start, index = unique_start, columns = pd.Index(['Trips']))

df.plot(kind='bar')

plt.show()



df.sort_values(['Trips'], ascending=False, inplace=True)

df.head(5)
location_end = []

for i in unique_stop:

    temp = uber_drives[uber_drives['STOP*'] == i]

    location_end.append(len(temp))



df = pd.DataFrame(location_end, index = unique_stop, columns = pd.Index(['Trips']))

df.plot(kind='bar')

plt.show()



df.sort_values(['Trips'], ascending=False, inplace=True)

df.head(5)