import csv

import numpy as np

import matplotlib.pyplot as plt

from collections import OrderedDict

from subprocess import check_output

print(check_output(['ls', '../input']).decode('utf-8'))
with open('../input/Seasons_Stats.csv') as csv_file:

    stats = list(csv.reader(csv_file))
categories = stats[0] # Categories of features

name_i = categories.index('Player')  # Player's names

year_i = categories.index('Year')  # Season Year

pos_i = categories.index('Pos')  # Player's Position

thr_poi = categories.index('3P%')  # Player's Three Point Percentage

print('Indices of features: \nName - {n} \nYear - {y} \nPosition - {p}\n3P% - {t}'.format(n=name_i, y=year_i, p=pos_i, t=thr_poi))
not_valid = ['', ' ', '0']  # List of values to exclude for 3P%

valid_pos = ['PG', 'SG', 'SF', 'PF', 'C']  # Positions being considered



players = list()



# Making a list of players relevant information who meet the criteria above

for st_line in stats[1:]:



    if st_line[pos_i] in valid_pos and st_line[thr_poi] not in not_valid:

        players.append([st_line[name_i], st_line[year_i], st_line[pos_i], float(st_line[thr_poi])])

        

players = np.array(players)

print(players)

print(players.shape)


# Splitting relevant information into columns

select_name = np.repeat(0, len(players))

names = np.choose(select_name, players.T)

print(names)

select_year = np.repeat(1, len(players))

years = set(np.choose(select_year, players.T))

print(years)



# Creates dictionary to hold positional three-point average of each year.

pos_three_by_year = {year: {pos: 0 for pos in valid_pos} for year in years}



# Holds the count of each position for each year

count = {year: OrderedDict({pos: 0 for pos in valid_pos}) for year in years}



# Add up all the percentages for each position yearly and get counts of 

# number of players in each positions

for name, year, pos, thr_poi in players:

    percent = float(thr_poi)

    pos_three_by_year[year][pos] += percent

    count[year][pos] += 1



# Divide each sum by the number of players of that position

for year, positions in pos_three_by_year.items():

    for pos, percentage in positions.items():

        positions[pos] = percentage/count[year][pos]



# Create OrderedDict to maintain order of years

pos_three_by_year = OrderedDict(sorted(pos_three_by_year.items(), key=lambda t: t[0]))

print(pos_three_by_year)
end_dates = list(pos_three_by_year.keys())

begin_dates = ['1979'] 



for dates in end_dates[:len(end_dates)-1]:

    begin_dates.append(dates)

print('Begin:', begin_dates)

print('End:', end_dates)

x_values = list()



for begin, end in zip(begin_dates, end_dates):

    x_values.append(begin + '-' + end[2:])

x = x_values

print('X:', x_values)
position_percent = list()



for year, position in pos_three_by_year.items():

    temp = list()

    for pos, percent in position.items():

        temp.append(percent)

    position_percent.append(temp)



# Percentages are relative to positions thanks to OrderedDict

position_percent = np.array(position_percent) 

print(position_percent)

print(position_percent.shape)
# Getting column of each position and plotting them on scatter plot

pg = np.repeat(0, len(position_percent))

sg = np.repeat(1, len(position_percent))

sf = np.repeat(2, len(position_percent))

pf = np.repeat(3, len(position_percent))

c = np.repeat(4, len(position_percent))

for select in [pg, sg, sf, pf, c]:

    y = np.choose(select, position_percent.T)

    plt.scatter(x, y)



plt.xlabel('years')

plt.xticks(rotation=90)

plt.ylabel('three_point_percentage')



plt.title('Three Point Percentage in the NBA by Position')

plt.legend(valid_pos) 



height = 15

width = 15

plt.rcParams['figure.figsize'] = [height, width]

plt.show()
total_percent_by_year = {year: 0 for year in years}



for year, positions in pos_three_by_year.items():

    for pos, percentage in positions.items():

        total_percent_by_year[year] += percentage

            

    # Divide sum of percentages of positions per year and divide that by the number of positions accounted for

    total_percent = round(total_percent_by_year[year]/len(valid_pos), 3) 

    total_percent_by_year[year] = total_percent

    

total_percent_by_year = OrderedDict(sorted(total_percent_by_year.items(), key=lambda t: t[0]))   



y = total_percent_by_year.values()

plt.xlabel('years')

plt.ylabel('three_point_percentage')

plt.yticks(np.arange(min(y), max(y), 0.005))

plt.xticks(rotation=90)

plt.title('3P% in the NBA')

plt.grid(True)

plt.plot(x, y)

plt.show()