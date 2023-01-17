from IPython.core.display import HTML
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 

from numpy import arange

%matplotlib inline

import statsmodels.api as sm

from statsmodels.formula.api import ols

from scipy import stats



import warnings

warnings.filterwarnings('ignore')



plt.style.use('fivethirtyeight')
#get the csv file

data = pd.read_csv('https://raw.githubusercontent.com/washingtonpost/data-police-shootings/master/fatal-police-shootings-data.csv')
data.columns
data.head()
#data summary

total_shootings = len(data.index)



#date range

first_date = data['date'][0]

last_date = data.iloc[-1]['date']



#ages

avg_age = round(data['age'].mean())

youngest = round(data['age'].min())

oldest = round(data['age'].max())



#gender

male = data.apply(lambda x: True if x['gender'] == 'M' else False , axis=1)

male = len(male[male == True].index)

female = data.apply(lambda x: True if x['gender'] == 'F' else False , axis=1)

female = len(female[female == True].index)

perc_male = round(male / total_shootings * 100)

perc_female = round(female / total_shootings * 100)



#race

white = data.apply(lambda x: True if x['race'] == 'W' else False , axis=1)

white = len(white[white == True].index)

white_perc = round(white / total_shootings * 100)

black = data.apply(lambda x: True if x['race'] == 'B' else False , axis=1)

black = len(black[black == True].index)

black_perc = round(black / total_shootings * 100)

hispanic = data.apply(lambda x: True if x['race'] == 'H' else False , axis=1)

hispanic = len(hispanic[hispanic == True].index)

hispanic_perc = round(hispanic / total_shootings * 100)

asian = data.apply(lambda x: True if x['race'] == 'A' else False , axis=1)

asian = len(asian[asian == True].index)

asian_perc = round(asian / total_shootings * 100)

native_american = data.apply(lambda x: True if x['race'] == 'N' else False , axis=1)

native_american = len(native_american[native_american == True].index)

native_american_perc = round(native_american / total_shootings * 100)

other = data.apply(lambda x: True if x['race'] == '0' else False , axis=1)

other = len(other[other == True].index)

other_perc = round(other / total_shootings * 100)



#summary text

"There are {} total police shooting fatalities from {} to {}. The average age was {}. The youngest was {} years old. The oldest was {} years old. {} were male ({}%) and {} were female ({}%). The racial breakdown is as follows: White {} ({}%), Black {} ({}%), Hispanic {} ({}%), Asian {} ({}%), Native American {} ({}%), Other {} ({}%).".format(total_shootings, first_date, last_date, avg_age, youngest, oldest, male, perc_male, female, perc_female, white, white_perc, black, black_perc, hispanic, hispanic_perc, asian, asian_perc, native_american, native_american_perc, other, other_perc)
plotdata = pd.DataFrame({

    "Population":[60, 12, 17, 6, 1],

    "Fatalities":[white_perc, black_perc, hispanic_perc, asian_perc, native_american_perc],

    }, 

    index=['White', 'Black', 'Hispanic', 'Asian', 'Native American']

)

plotdata.plot(kind="bar", color=['gray','red'], figsize=(15,10))

plt.title("Fatalities Versus Population By Race")

plt.xlabel("Race")

plt.ylabel("Percentage")
#drop race H, A, N, O and missing variables (unknown)

data.dropna()



h = data.loc[data['race'] == 'H'].index

data.drop(h , inplace=True)



a = data.loc[data['race'] == 'A'].index

data.drop(a , inplace=True)



n = data.loc[data['race'] == 'N'].index

data.drop(n , inplace=True)



o = data.loc[data['race'] == 'O'].index

data.drop(o , inplace=True)



data.shape
data.drop(columns=['id', 'name', 'date', 'manner_of_death', 'signs_of_mental_illness', 'body_camera'], inplace=True)

data.head()
data['location'] = data['city'] + ', ' + data['state']

data.drop(columns=['city', 'state'], inplace=True)



data.head()
#convert armed

data["armed"].replace({"gun": 1, "unarmed": 0}, inplace=True)

data.loc[(data['armed'] != 0) & (data['armed'] != 1) , 'armed'] = 0.5



#convert threat_level

data["threat_level"].replace({"attack": 1, "undetermined": 0.5, "other": 0}, inplace=True)



#convert flee

data["flee"].replace({"Not fleeing": 1, "Other": 0.5, "Car": 0, "Foot": 0}, inplace=True)



#convert gender

data["gender"].replace({"M": 1, "F": 0}, inplace=True)



#convert race

data["race"].replace({"B": 1, "W": 0}, inplace=True)



data.head()
discretion = data[['armed', 'threat_level','flee']] 

data['discretion'] = round(discretion.mean(axis=1),2)



data.head()
score_groups_bottom_third = data.loc[data['discretion'] < 0.33]

count_bottom_third = len(score_groups_bottom_third.index)

perc_bottom_third = round(count_bottom_third / total_shootings * 100)



score_groups_middle = data.loc[(data['discretion'] > .33) & (data['discretion'] < .66)]

count_middle = len(score_groups_middle.index)

perc_middle = round(count_middle / total_shootings * 100)



score_groups_upper_third = data.loc[data['discretion'] > 0.66]

count_upper_third = len(score_groups_upper_third.index)

perc_upper_third = round(count_upper_third / total_shootings * 100)



"There are {} ({}%) shootings with a high level of discretion and least likely to be justified (score below 0.33), {} ({}%) with medium discretion (score between .33 and .66), and {} ({}%) with low discretion, most likely to be justified (score above .66).".format(count_bottom_third, perc_bottom_third, count_middle, perc_middle, count_upper_third, perc_upper_third)

disc_third_one = data.loc[data['discretion'] < .33]

disc_third_one = len(disc_third_one.index)

black_third_one = data.loc[data['race'] == 1]

black_third_one = len(black_third_one.index) / disc_third_one



disc_third_two = data.loc[(data['discretion'] > .33) & (data['discretion'] < .66)]

disc_third_two = len(disc_third_two.index)



disc_third_three = data.loc[data['discretion'] > .66]

disc_third_three = len(disc_third_three.index)



disc_thirds = [disc_third_one, disc_third_two, disc_third_three]

index = ["High Discretion", "Medium Discretion", "Low Discretion"]

df = pd.DataFrame({'Discretion': disc_thirds}, index=index)

ax = df.plot.bar(rot=0, color=['red'], figsize=(15,10), title='Number Of Incidents Per Discretion Level')
black_bottom_third = score_groups_bottom_third.loc[score_groups_bottom_third['race'] == 1.0]

black_bottom_third_count = len(black_bottom_third.index)

black_bottom_third_perc = round(black_bottom_third_count / count_bottom_third * 100)



black_middle = score_groups_middle.loc[score_groups_middle['race'] == 1.0]

black_middle_count = len(black_middle.index)

black_middle_perc = round(black_middle_count / count_middle * 100)



black_upper_third = score_groups_upper_third.loc[score_groups_upper_third['race'] == 1.0]

black_upper_third_count = len(black_upper_third.index)

black_upper_third_perc = round(black_upper_third_count / count_upper_third * 100)



"Black subjects represent {}% of total fatalities with a high level of discretion and least likely to be justified (score below 0.33), {}% of shootings with medium discretion (score between .33 and .66), and {}% of shootings with low discretion, most likely to be justified (score above 0.66).".format(black_bottom_third_perc, black_middle_perc, black_upper_third_perc)

white_bottom_third = score_groups_bottom_third.loc[score_groups_bottom_third['race'] == 0.0]

white_bottom_third_count = len(white_bottom_third.index)

white_bottom_third_perc = round(white_bottom_third_count / count_bottom_third * 100)



white_middle = score_groups_middle.loc[score_groups_middle['race'] == 0.0]

white_middle_count = len(white_middle.index)

white_middle_perc = round(white_middle_count / count_middle * 100)



white_upper_third = score_groups_upper_third.loc[score_groups_upper_third['race'] == 0.0]

white_upper_third_count = len(white_upper_third.index)

white_upper_third_perc = round(white_upper_third_count / count_upper_third * 100)



"White subjects represent {}% of total fatalities with a high level of discretion and least likely to be justified (score below 0.5 ), {}% of shootings with medium discretion (score at 0.5), and {}% of shootings with low discretion, most likely to be justified (score above 0.5).".format(white_bottom_third_perc, white_middle_perc, white_upper_third_perc)

total_justified = data.loc[data['discretion'] == 1]

total_justified = len(total_justified.index)

total_unjustified = data.loc[data['discretion'] == 0]

total_unjustified = len(total_unjustified.index)



#justified shootings by race: white

white_justified_perc = data.loc[(data['race'] == 0) & (data['discretion'] == 1)]

white_justified_perc = len(white_justified_perc.index)

white_justified_perc = round(white_justified_perc / total_justified * 100)

#justified shootings by race: black

black_justified_perc = data.loc[(data['race'] == 1) & (data['discretion'] == 1)]

black_justified_perc = len(black_justified_perc.index)

black_justified_perc = round(black_justified_perc / total_justified * 100)



#unjustified shootings by race: white

white_unjustified_perc = data.loc[(data['race'] == 0) & (data['discretion'] == 0)]

white_unjustified_perc = len(white_unjustified_perc.index)

white_unjustified_perc = round(white_unjustified_perc / total_unjustified * 100)

#unjustified shootings by race: black

black_unjustified_perc = data.loc[(data['race'] == 1) & (data['discretion'] == 0)]

black_unjustified_perc = len(black_unjustified_perc.index)

black_unjustified_perc = round(black_unjustified_perc / total_unjustified * 100)



white = [white_justified_perc, white_unjustified_perc]

black = [black_justified_perc, black_unjustified_perc]

index = ['Justified Shootings', 'Unjustified Shootings']

df = pd.DataFrame({'White': white, 'Black': black}, index=index)

ax = df.plot.bar(rot=0, color=['gray','red'], figsize=(15,10), title='Racial Disparity In Justified And Unjustified Shootings')