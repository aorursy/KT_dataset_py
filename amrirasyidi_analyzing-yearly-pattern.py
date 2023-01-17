# import module needed to analyze the dataset



from csv import reader

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.style as style

style.use('fivethirtyeight')

%matplotlib inline
# creating a dataframe type of data



firepd = pd.read_csv('../input/amazon2/amazon2.csv', encoding = 'latin1', thousands = ".")



print("First 5 rows of the dataset: ")

print(firepd.head())

print("\nData Description")

print(firepd.describe())

print("\nData details")

print(firepd.info())
print('The proof of our suspiciousness: \n','\n',firepd.date.unique())

firepd = firepd.drop(['date'], axis = 1)

firepd.head()
# change the month format into number



mo_in_span = list(firepd.month.unique())



mo_num_dict = {}

for i in range (1,13):

    mo_num_dict[mo_in_span[i-1]] = i



firepd = firepd.replace({'month': mo_num_dict})



print("First 5 rows of the dataset: ")

firepd.head()
# this dataframe stores the MAXIMUM number of forest fire in the respective year

temp1 = firepd[['year','number']].groupby('year').max().reset_index() 



max_year_pd = firepd[(firepd['year'].isin(temp1['year'])) & 

                     (firepd['number'].isin(temp1['number']))].sort_values(by = 'year')



# this dataframe stores the TOTAL number of forest fire in the respective year

data = firepd[['year','number']].groupby(['year']).sum().reset_index() 

    

# Visualization



fig, ax1 = plt.subplots(figsize=(14,6))



color = 'tab:blue'

ax1.set_title('Forest Fire Record', fontsize = 20)

ax1.set_xlabel('year')

ax1.set_ylabel('Maximum number reported', color=color)

ax1.bar(list(max_year_pd['year']), list(max_year_pd['number']), color = color)

ax1.grid(axis = 'x')

ax1.set_xticks(list(data['year']))



ax2 = ax1.twinx() # Create a twin Axes sharing the x axis



color = 'tab:red'

ax2.set_ylabel('Total reported', color=color)

ax2.plot(list(data['year']), list(data['number']), color = color)

ax2.grid(None)

ax2.tick_params(axis='y', labelcolor=color)
# Create a frequency table of month vs max reported fire each year



month_freq = max_year_pd['month'].value_counts().to_frame().reset_index()



month_freq = month_freq.rename(columns={'index':'month', 'month':'frequency'})



none_month = []



for number in range(1,13):

    if number not in list(month_freq['month']):

        none_month.append([number, 0])

        

none_pd = pd.DataFrame(none_month, columns = ['month', 'frequency'])



month_freq = month_freq.append(none_pd, ignore_index = True)



# Visualize the table



month_freq_graph = month_freq.sort_values('month').plot(x = 'month', y = 'frequency',

               kind = 'bar',

               rot = 0, figsize = (10,6))



box_month = firepd.boxplot(column = ['number'], by = ['month'], fontsize = 12, figsize = (15,8), showfliers = False)
for i in range (1,13):

    if i == 1:

        print("There are ",len(firepd[(firepd.month == i) & (firepd.number > 0)])," forest fire reported on ",str(i)+"st month")

    elif i == 2:

        print("There are ",len(firepd[(firepd.month == i) & (firepd.number > 0)])," forest fire reported on ",str(i)+"nd month")

    elif i == 3:

        print("There are ",len(firepd[(firepd.month == i) & (firepd.number > 0)])," forest fire reported on ",str(i)+"rd month")

    else:

        print("There are ",len(firepd[(firepd.month == i) & (firepd.number > 0)])," forest fire reported on ",str(i)+"th month")
mean = firepd['number'].mean()



above_avg_month_dict = {}



for i in range(1,13):

    above_avg_month_dict[i] = len(firepd[(firepd.month == i) & (firepd.number >= mean)])



above_avg_month_pd = pd.DataFrame(list(above_avg_month_dict.items()), columns = ['month', 'count above avg'])

above_avg_month_pd['rank'] = above_avg_month_pd['count above avg'].rank(method='first', ascending = False).astype('int64')

above_avg_month_pd = above_avg_month_pd[['rank', 'count above avg', 'month']] # rearrange the column order

above_avg_month_pd = above_avg_month_pd.sort_values('rank')

above_avg_month_pd