import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set(style="whitegrid")



from collections import Counter
data = pd.read_csv('../input/ign.csv', index_col=0)
data.head(2)
data.info()
score_counts = data['score_phrase'].value_counts()
ordered_score = ['Disaster', 'Unbearable' ,'Painful' ,'Awful' ,'Bad', 'Mediocre', 

                 'Okay' ,'Good' ,'Great', 'Amazing', 'Masterpiece']

counts = []

for score in ordered_score:

    counts.append(score_counts[score])
fig, ax = plt.subplots(figsize=(11, 4))

sns.barplot(x=ordered_score, y=counts, color='#833ab4')

ax.set(ylabel="Count", xlabel="Score")

ticks = plt.setp(ax.get_xticklabels(), rotation=30, fontsize=9)
def get_platform_type(platform):    

    if platform in ['Nintendo 64', 'PlayStation']:

        return 'Fifth generation'

    elif platform in ['Dreamcast', 'PlayStation 2', 'GameCube', 'Xbox']:

        return 'Sixth generation'

    elif platform in ['Xbox 360', 'PlayStation 3', 'Wii', 'Nintendo DS', 'PlayStation Portable']:

        return 'Seventh generation'

    elif platform in ['Wii U', 'PlayStation 4', 'Xbox One', 'Nintendo 3DS', 'PlayStation Vita']:

        return 'Eighth generation'

    else:

        return 'Other'



data['platform_type'] = data['platform'].apply(get_platform_type)
current_data = dict(data[data['platform_type'] == 'Fifth generation']['platform'].value_counts())



f, ax = plt.subplots(figsize=(4, 4))

labels = list(current_data.keys())

colors = ['#ffdc80', '#fcaf45']

plt.pie(list(current_data.values()), 

        labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

axis = plt.axis('equal')
current_data = dict(data[data['platform_type'] == 'Sixth generation']['platform'].value_counts())



f, ax = plt.subplots(figsize=(4, 4))

labels = list(current_data.keys())

colors = ['#ffdc80', '#fcaf45', '#f56040', '#e1306c']

plt.pie(list(current_data.values()), 

        labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

axis = plt.axis('equal')
current_data = dict(data[data['platform_type'] == 'Seventh generation']['platform'].value_counts())



f, ax = plt.subplots(figsize=(4, 4))

labels = list(current_data.keys())

colors = ['#ffdc80', '#fcaf45', '#f56040', '#e1306c', '#c13584']

plt.pie(list(current_data.values()), 

        labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

axis = plt.axis('equal')
current_data = dict(data[data['platform_type'] == 'Eighth generation']['platform'].value_counts())



f, ax = plt.subplots(figsize=(4, 4))

labels = list(current_data.keys())

colors = ['#ffdc80', '#fcaf45', '#f56040', '#e1306c', '#c13584']

plt.pie(list(current_data.values()), 

        labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

axis = plt.axis('equal')
f, ax = plt.subplots(figsize=(4, 4))

plt.pie(data['editors_choice'].value_counts().tolist(), 

        labels=['No', 'Yes'], colors=['#fcaf45', '#ffdc80'], 

        autopct='%1.1f%%', startangle=90)

axis = plt.axis('equal')
data['genre'].value_counts()[0:15]
scores = data['score']
years = data['release_year']

months = data['release_month']

days = data['release_day']
all_games_by_years = dict(years.value_counts().sort_values())

all_games_by_months = dict(months.value_counts().sort_values())

all_games_by_days = dict(days.value_counts().sort_values())
good_games_by_years = dict()

good_games_by_months = dict()

good_games_by_days = dict()
for (year, month, day, score) in zip(years, months, days, scores):

    if (score > 7.0):

        for (period, good_games_dict) in zip([year, month, day],

                                             [good_games_by_years, 

                                              good_games_by_months, 

                                              good_games_by_days]):

            if period not in good_games_dict:

                good_games_dict[period] = 1

            else:

                good_games_dict[period] += 1
fig, ax = plt.subplots(figsize=(11, 4))



sns.barplot(x=list(all_games_by_years.keys()), y=list(all_games_by_years.values()), color='#ffdc80')

sns.barplot(x=list(good_games_by_years.keys()), y=list(good_games_by_years.values()), color='#405de6')



ax.set(ylabel="Count", xlabel="Years")

ticks = plt.setp(ax.get_xticklabels(), rotation=30, fontsize=9)
fig, ax = plt.subplots(figsize=(11, 4))



sns.barplot(x=list(all_games_by_months.keys()), y=list(all_games_by_months.values()), color='#ffdc80')

sns.barplot(x=list(good_games_by_months.keys()), y=list(good_games_by_months.values()), color='#405de6')



ax.set(ylabel="Count", xlabel="Months")

ticks = plt.setp(ax.get_xticklabels(), fontsize=9)
fig, ax = plt.subplots(figsize=(11, 4))



sns.barplot(x=list(all_games_by_days.keys()), y=list(all_games_by_days.values()), color='#ffdc80')

sns.barplot(x=list(good_games_by_days.keys()), y=list(good_games_by_days.values()), color='#405de6')



ax.set(ylabel="Count", xlabel="Days")

ticks = plt.setp(ax.get_xticklabels(), fontsize=9)
years = sorted(data['release_year'].unique())
for year in years:

    curr_year_data = list(data[(data['release_year'] == year) & (data['score'] == 10)]['title'])

    if (curr_year_data != []):

        print(year, end=':\n')

        print(', '.join(curr_year_data))

        print()