# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
police_shootings_filepath = "../input/fatal-police-shootings/fatal-police-shootings-data.csv"

police_shootings_data = pd.read_csv(police_shootings_filepath, index_col="date", parse_dates=True)

police_shootings_data.head()

police_shootings_data.tail()
# KDE plot 

sns.kdeplot(data=police_shootings_data['age'], shade=True)
total = 0

male = 0

female = 0

unknown_gender = 0



for person in police_shootings_data['gender']:

    if (person == 'M') or (person == 'f'):        

        male += 1

        total += 1

    elif (person == 'F') or (person == 'f'):

        female += 1

        total += 1

    else:

        unknown_gender += 1

        total += 1

print('Total number of people shot:', total)

print('Total number of men shot:', male)

print('Total number of women shot:', female)

print('Total number of people shot with no gender listed:', unknown_gender)

print('Percentage of people shot who were male:', ((male/total)*100),'%')

print('Percentage of people shot who were female:', ((female/total)*100),'%')

print('Ratio of men to women who were shot:', (male/female), 'to 1')
total = 0

white = 0

black = 0

latinx = 0

asian = 0

other_races = 0



for person in police_shootings_data['race']:

    if (person == 'W') or (person == 'w'):        

        white += 1

        total += 1

    elif (person == 'B') or (person == 'b'):

        black += 1

        total += 1

    elif (person == 'H') or (person == 'h'):

        latinx += 1

        total += 1

    elif (person == 'A') or (person == 'a'):

        asian += 1

        total += 1

    elif (person != 'B') and (person != 'W') and (person != 'H') and (person != 'A'):

        other_races += 1

        total += 1

print('Total number of people shot:', total)

print('Total number of white people shot:', white)

print('Total number of black people shot:', black)

print('Total number of Asian people shot:', asian)

print('Total number of Latinx people shot:', latinx)

print('Total number of people of other races shot:', other_races)

print('Percentage of people shot who were white:', ((white/total)*100),'%')

print('Percentage of people shot who were black:', ((black/total)*100),'%')

print('Percentage of people shot who were Asian:', ((asian/total)*100),'%')

print('Percentage of people shot who were Latinx:', ((latinx/total)*100),'%')

print('Percentage of people shot who were other races:', ((other_races/total)*100), '%')
# How many people were armed and what weapons did they have

total_unarmed = 0

total_gun = 0

total_knife = 0

total_vehicle = 0

total_toy_weapon = 0

total_other_weapon = 0

total_ppl = 0



for person in police_shootings_data['armed']:

    if person == 'unarmed':

        total_unarmed += 1

        total_ppl += 1

    elif person == 'gun':

        total_gun += 1

        total_ppl += 1

    elif person == 'knife':

        total_knife += 1

        total_ppl += 1

    elif person == 'vehicle':

        total_vehicle += 1

        total_ppl += 1

    elif person == 'toy weapon':

        total_toy_weapon += 1

        total_ppl += 1

    else:

        total_other_weapon += 1

        total_ppl += 1



print('Percentage of people shot who were unarmed:', ((total_unarmed/total_ppl)*100), '%')

print('Percentage of people shot who were armed with guns:', (((total_gun)/total_ppl)*100), '%')

print('Percentage of people shot who were armed with knives:', (((total_knife)/total_ppl)*100), '%')

print('Percentage of people shot who were armed with vehicles:', (((total_vehicle)/total_ppl)*100), '%')

print('Percentage of people shot who were armed with toy weapons:', (((total_toy_weapon)/total_ppl)*100), '%')

print('Percentage of people shot who were armed with other weapons:', (((total_other_weapon)/total_ppl)*100), '%')
# Total number of people of each race shot

white_total = 1631

black_total = 818

latinx_total = 579

asian_total = 55



# Total number of people of each race shot, unarmed

white_unarmed = 0

black_unarmed = 0

latinx_unarmed = 0

asian_unarmed = 0 



# I got some help with this for loop from another CRLS student, Pratyush

for _, race, armed in police_shootings_data[['race', 'armed']].itertuples():

    if (race == 'W') and (armed == 'unarmed'):

        white_unarmed += 1

    elif (race == 'B') and (armed == 'unarmed'):

        black_unarmed += 1

    elif (race == 'H') and (armed == 'unarmed'):

        latinx_unarmed += 1

    elif (race == 'A') and (armed == 'unarmed'):

        asian_unarmed += 1



print('Percentage of white people shot who were unarmed:', ((white_unarmed/white_total)*100),'%')

print('Percentage of black people shot who were unarmed:', ((black_unarmed/black_total)*100),'%')

print('Percentage of Asian people shot who were unarmed', ((asian_unarmed/asian_total)*100),'%')

print('Percentage of Latinx people shot who were unarmed', ((latinx_unarmed/latinx_total)*100),'%')