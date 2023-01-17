import numpy as np 

import pandas as pd 

from cycler import cycler

import matplotlib.pyplot as plt

from pylab import rcParams

from statistics import mean, mode

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# MatPlotLib PyPlot parameters 

rcParams['figure.figsize'] = (12, 6)

def get_hex(rgb): 

    return '#%02x%02x%02x' % (int(rgb[0]*255),int(rgb[1]*255),int(rgb[2]*255))

colors = [get_hex(plt.get_cmap("Pastel1")(i)) for i in range(10)]

rcParams['axes.prop_cycle'] = cycler('color', colors)
data = pd.read_csv("../input/master.csv")

print(data.shape)

data.head()
print("Is 'country-year' redundant? ", all(data['country-year'] == data['country'] + data['year'].apply(str)))

print("Is 'suicides/100k pop' redundant? ", all(data['suicides/100k pop'] == (100000*data['suicides_no']/data['population']).apply(lambda x: round(x, 2))))

print("    Inconsistent data: ", [round(x, 2) for x in data['suicides/100k pop'] - (100000*data['suicides_no']/data['population']).apply(lambda x: round(x, 2)) if x != 0])

data.drop(columns=['country-year'], inplace=True)

data.rename({'suicides/100k pop': 'rate'}, axis=1, inplace=True)
print(all(le.fit_transform(data['age']) == le.fit_transform(data['generation'])))
print(list(data['age'].unique()))

print(list(data['generation'].unique()))

age_order = ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years']

generation_order = ['G.I. Generation', 'Silent', 'Boomers', 'Generation X', 'Millenials', 'Generation Z']
rates = []

for age in age_order: 

    age_data = data[data['age'] == age]

    overall_rate = 100000 * age_data['suicides_no'].sum() / age_data['population'].sum() 

    rates.append(overall_rate)

y_pos = np.arange(len(age_order))

plt.figure(figsize=(12,6))

plt.bar(y_pos, rates, align='center')

plt.xticks(y_pos, age_order, size="large")

plt.ylabel('Suicides per 100K people', size="x-large")

plt.title("Suicides per 100K people, sorted by age of victim", size="xx-large", pad=13)

plt.show() 
male_rates = []; female_rates = []

for age in age_order: 

    age_data = data[data['age'] == age]

    male_age_data = age_data[age_data['sex'] == 'male']

    female_age_data = age_data[age_data['sex'] == 'female']

    male_rate = 100000 * male_age_data['suicides_no'].sum() / male_age_data['population'].sum() 

    female_rate = 100000 * female_age_data['suicides_no'].sum() / female_age_data['population'].sum() 

    male_rates.append(male_rate)

    female_rates.append(female_rate)

y_pos = np.arange(len(age_order))

bar_width = 0.35

fig, ax = plt.subplots(figsize=(12,6))

plt.bar(y_pos, female_rates, bar_width, label="Women")

plt.bar(y_pos + bar_width, male_rates, bar_width, label="Men")

plt.xticks(y_pos, age_order, size="large")

plt.ylabel('Suicides per 100K people', size="x-large")

plt.title("Suicides per 100K people, sorted by age & sex of victim", size="xx-large", pad=13)

plt.legend()

plt.show() 
female_data = data[data['sex'] == 'female']

male_data = data[data['sex'] == 'male']

gender_rates = [list(female_data['rate']), list(male_data['rate'])]

fig, ax = plt.subplots(figsize=(18,6))

parts = plt.violinplot(gender_rates, showmeans=False, showextrema=False, vert=False)

plt.xlim(left=-2, right=100)

for i, body in enumerate(parts['bodies']): 

    body.set_facecolor(colors[i])

    body.set_edgecolor("black")

plt.title("Suicide rates, by sex", size="xx-large", pad=10)

ax.set_yticks(np.arange(1, 3))

ax.set_yticklabels(["Female", "Male"], size="large")

ax.set_xlabel('Suicides per 100K people', size="x-large") 

plt.show() 

print(f"Average suicides (per 100K people) for men: {mean(male_data['rate']):.2f}, "+\

      f"for women: {mean(female_data['rate']):.2f}")