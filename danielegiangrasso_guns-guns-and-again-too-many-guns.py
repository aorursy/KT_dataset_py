import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.ticker import StrMethodFormatter

import matplotlib.ticker as ticker
police_deaths = pd.read_csv("../input/police-violence-in-the-us/police_deaths_538.csv")

police_deaths.head()
new = police_deaths['eow'].str.split(" ", n = 4, expand = True) 

police_deaths['weekday'] = new[1].str.replace(',', '')

colors =  ["green",'red', 'gold', 'orange', 'pink', 'lightcoral', 'lightskyblue']

cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

police_deaths_wk = (police_deaths.groupby(['weekday']).count())

police_deaths_wk = police_deaths_wk['person']

police_deaths_wk = police_deaths_wk.reset_index()

police_deaths_wk['weekday'] = pd.Categorical(police_deaths_wk['weekday'], categories=cats, ordered=True)

police_deaths_wk = police_deaths_wk.sort_values('weekday')

police_deaths_wk.set_index('weekday',inplace = True)

police_deaths_wk.rename(columns = {"person" : ""} , inplace=True)

police_deaths_wk.plot.pie(figsize=(8,7),

                          shadow=False,

                          colors=colors,

                          explode=(0, 0, 0, 0, 0.05, 0.09, 0.05),

                          startangle=90,

                          subplots=True,

                          legend=False,

                          autopct='%1.2f%%')

plt.title("% Day of the week on which Police Officers were killed 1796-2016")

plt.axis('equal')

plt.tight_layout()

plt.show()
ax = pd.pivot_table(police_deaths, index='cause_short', values=['cause'], aggfunc="count").sort_values(

    "cause", axis = 0, ascending = True).plot.barh(figsize=(12,14),color='lightskyblue', width=0.85, xlim=[0, 12500], legend=False)

ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.spines['bottom'].set_visible(False)

ax.xaxis.set_major_locator(ticker.MultipleLocator(750))



vals = ax.get_xticks()

for tick in vals:

    ax.axvline( x =tick, linestyle='dotted', alpha=0.6, color='red', zorder=2)



    

ax.set_xlabel("Number of deaths", labelpad=25, size=16)

ax.set_ylabel("Cause", labelpad=25, size=16)

ax.set_title("Causes of Police Officer Death 1791 - 2016 ", size=18)

ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

totals = []

total = sum(totals)



# set individual bar lables using above list

for i in ax.patches:

    # get_width pulls left or right; get_y pushes up or down

    ax.text(i.get_width()+.3, i.get_y()+.40, str(round((i.get_width()))),fontsize=10,

color='dimgrey')
police_deaths_by_state = pd.pivot_table(police_deaths,index='state', values=['cause'], columns = ['cause_short'], aggfunc="count").fillna(value=0)

police_deaths_by_state.columns = police_deaths_by_state.columns.droplevel()

police_deaths_by_state.loc[:,'Total'] = police_deaths_by_state.sum(axis=1)

police_deaths_by_state.sort_values("Total", axis = 0, ascending = True, inplace=True)

police_deaths_by_state

police_deaths_by_state.drop(['Total'], axis = 1).head()

#the below it is to make the visualization a bit more clear, 

min_number_per_cat = 100

police_deaths_by_state = police_deaths_by_state.loc[: ,police_deaths_by_state.sum(axis=0) >= min_number_per_cat]

police_deaths_by_state.head()





colors = plt.cm.tab20c(np.linspace(0, 1, 15))

ax1 = police_deaths_by_state.drop(['Total'], axis = 1).plot.barh(figsize=(12,16),stacked=True, width=0.80,color=colors )



ax1.spines['right'].set_visible(False)

ax1.spines['top'].set_visible(False)



ax1.set_xlabel("Number Police Officer Death ", labelpad=20, weight='bold',size=14)

ax1.set_ylabel("States", labelpad=20, weight='bold', size=14)

ax1.set_title("Causes of Police Officer Death 1791 - 2016 ", size=18)

ax1.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))



ax = pd.pivot_table(police_deaths,index=['year'],values = ['cause_short'], aggfunc="count").nlargest(15,'cause_short').reset_index()

ax.rename(columns = {"cause_short" : "People Killed"} , inplace=True)

ax.plot.bar(x='year', figsize=(8,6),rot=0, legend = False, color='Red')

plt.xlabel("Year", labelpad=20, weight='bold',size=14)

plt.ylabel("People Killed", labelpad=20, weight='bold',size=14)

plt.title("Top 15 Year")
fatal_encounters= pd.read_csv("../input/police-violence-in-the-us/fatal_encounters_dot_org.csv")

#fatal_encounters_dot_org.info()

#fatal_encounters.head()
fatal_encounters = fatal_encounters.iloc[:,[0,2,1,3,4,5,8,10,11,15,16,18,19,22,27]]

fatal_encounters.head()
Death_Police_per_year =  pd.pivot_table(police_deaths.rename(columns = {"year" : "Year"}),index=['Year'],values = ['cause_short'], aggfunc="count")

Death_Police_per_year.rename(columns = {"cause_short" : "Police Officer killed"} , inplace=True)

Death_People_per_Year = pd.pivot_table(fatal_encounters.rename(columns = {"Date (Year)" : "Year"} ),index=['Year'],values = ['Unique ID'],aggfunc="count").fillna(value=0)

Death_People_per_Year = Death_People_per_Year.iloc[2:]

Death_People_per_Year.rename(columns = {"Unique ID" : "People killed"} , inplace=True)

Death_People_per_Year = Death_People_per_Year.reset_index()

Death_Police_per_year = Death_Police_per_year.reset_index()

Death_People_per_Year['Year'] = pd.to_numeric(Death_People_per_Year['Year'])

Killed = Death_People_per_Year.merge(Death_Police_per_year, left_on='Year', right_on='Year')

x = Killed['Year']

Killed.plot(x='Year', figsize=(12,6) )

plt.legend(loc='best')

plt.title("People Deaths vs Police Officer Deaths 2000-2016")

plt.xlabel("Year")

plt.xticks(np.arange(min(x), max(x)+1, 1.0))

plt.ylabel("Deaths")

#police_deaths_by_state = pd.pivot_table(police_deaths,index='state', values=['cause'], columns = ['cause_short'], aggfunc="count").fillna(value=0)

fatal_encounters_pv = pd.pivot_table(fatal_encounters,index=['Location of death (state)'],values = ['Unique ID'],columns = ['Cause of death'], aggfunc="count").fillna(value=0)

fatal_encounters_pv.columns = fatal_encounters_pv.columns.droplevel()

fatal_encounters_pv.loc[:,'Total'] = fatal_encounters_pv.sum(axis=1)

fatal_encounters_pv.sort_values("Total", axis = 0, ascending = True, inplace=True)

colors2 = plt.cm.nipy_spectral(np.linspace(0, 1, 15))

ax2 = fatal_encounters_pv.drop(['Total'], axis = 1).plot.barh(figsize=(12,16),stacked=True, width=0.80,color=colors2 )

ax2.set_xlabel("Number People Killed", labelpad=20, weight='bold',size=14)

ax2.set_ylabel("States", labelpad=20, weight='bold', size=14)

ax2.set_title("Causes of People Death per State 2000-2016 ", size=20)

#fatal_encounters_pv.head()
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

fatal_encounters.rename(columns = {"Location of death (city)" : "Location"} , inplace=True)

text_list = []

for i in range(len(fatal_encounters['Location'])):

    text_list.append(str(fatal_encounters['Location'][i]))



from collections import Counter

word_could_dict=Counter(text_list)



wordcloud = WordCloud(

    width = 2000,

    height = 1000,

    background_color = 'black',max_words=20).generate_from_frequencies(word_could_dict)

fig = plt.figure(

    figsize = (30, 20),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
ax = fatal_encounters["Subject's age"].value_counts().nlargest(15).plot(

    kind='barh', title ='Top 10 most frequent ages of killed people', grid= False);

ax.invert_yaxis()

ax.set_xlabel("Number People Killed", labelpad=10,size=14)

ax.set_ylabel("Age", labelpad=20, size=14)



totals = []



# find the values and append to list

for i in ax.patches:

    totals.append(i.get_width())



# set individual bar lables using above list

total = sum(totals)



# set individual bar lables using above list

for i in ax.patches:

    # get_width pulls left or right; get_y pushes up or down

    ax.text(i.get_width()+.3, i.get_y()+.40, str(round((i.get_width()))),fontsize=10,

color='dimgrey')

fatal_encounters.rename(columns = {"Subject's race" : "Race"} , inplace=True)

fatal_encounters["Race"] = np.where(fatal_encounters["Race"]=='HIspanic/Latino', 'Hispanic/Latino',fatal_encounters["Race"])

fatal_encounters["Race"].loc[fatal_encounters["Race"] == "Asian/Pacific Islander"] = 'Asian'

fatal_encounters["Race"].loc[fatal_encounters["Race"] == "African-American/Black"] = 'African-American'

fatal_encounters["Race"].loc[fatal_encounters["Race"] == "European-American/White"] = 'European-American'

fatal_encounters["Race"].loc[fatal_encounters["Race"] == "Hispanic/Latino"] = 'Hispanic/Latino'

fatal_encounters["Race"].loc[fatal_encounters["Race"] == "Native American/Alaskan"] = 'Native-American'

fatal_encounters["Race"].loc[fatal_encounters["Race"] == "Race unspecified"] = 'Unspecified'

race = pd.pivot_table(fatal_encounters,index=["Race"],columns = ["Cause of death"],values = ["Subject's name"] , aggfunc = "count").fillna(value=0)

race.columns = [f'{j}' for i, j in race.columns]

race.head()
fig = race.plot(figsize=(14,11),kind='bar', stacked=True)

plt.legend(loc = 'best', fontsize=9.5)

plt.xticks(rotation=90, fontsize=20)

fig.set_xticklabels(fig.get_xticklabels(),rotation=0, fontweight='light', fontsize=11)

plt.xlabel("Race", labelpad=10,size=14)

plt.ylabel("Number of People Killed", labelpad=20, size=14)

plt.title("People Killed by Race and Causes",weight = "bold", size=20)

plt.show
shootings = pd.read_csv("../input/police-violence-in-the-us/shootings_wash_post.csv")

shootings.describe()

d = {'W': 'White', 'B': 'Black','A': 'Asian','N': 'Native','H':'Hispanic', 'O': 'Other'}

gend = {'M': 'Male', 'F': 'Female'}

shootings["gender"] = shootings["gender"].map(gend)

shootings["race"] = shootings["race"].map(d)

shootings.isna().sum()

shootings['armed'].fillna('No', inplace=True)

shootings['age'].fillna(0, inplace=True)

shootings['race'].fillna('Other', inplace=True)

shootings['flee'].fillna('Not specified', inplace=True)

shootings.rename(columns = {"signs_of_mental_illness" : "Signs Mental illness"} , inplace=True)

shootings.isna().sum()

shootings.head()
text_list = []

for i in range(len(shootings["city"])):

    text_list.append(str(shootings["city"][i]))



from collections import Counter

word_could_dict=Counter(text_list)



wordcloud = WordCloud(

    width = 2000,

    height = 1000,

    background_color = 'White',max_words=10).generate_from_frequencies(word_could_dict)

fig = plt.figure(

    figsize = (30, 20),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
text_list = []

for i in range(len(shootings["armed"])):

    text_list.append(str(shootings["armed"][i]))



from collections import Counter

word_could_dict=Counter(text_list)



wordcloud = WordCloud(

    width = 2000,

    height = 1000,

    background_color = 'white',max_words=30).generate_from_frequencies(word_could_dict)

fig = plt.figure(

    figsize = (30, 20),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
from statsmodels.graphics.mosaicplot import mosaic

from bokeh.transform import factor_cmap

plt.rcParams['font.size'] = 14

plt.rcParams["figure.figsize"] = [12, 8]

plt.rcParams["figure.edgecolor"] = 'w'

plt.rcParams['text.color'] = 'Black'

mosaic(shootings, ['race','gender'],statistic=True,gap=0.02, axes_label = True, title='Split by Race and Gender',labelizer=lambda k: '')

plt.show()
import seaborn as sns

shootings["age"] = pd.to_numeric(shootings["age"])

#sns.catplot(x='flee', y='age', hue='gender',palette='Set1',

#            data=shootings,jitter=0.15)

sns.swarmplot(x='race', y='age',hue='Signs Mental illness',data=shootings, palette=["r", "b"],split=True)

#sns.swarmplot(x="day",y="total_bill",hue='sex',data=t,palette="Set1", split=True)