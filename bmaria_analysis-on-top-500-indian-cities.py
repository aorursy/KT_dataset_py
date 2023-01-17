# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.ticker as tick



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df=pd.read_csv("../input/cities_r2.csv")

%matplotlib inline

group_by_states=df.sort_values(by='sex_ratio').groupby(['state_name'])[["population_male", "population_female"]].sum()

group_by_states=group_by_states.reset_index()

group_by_states=group_by_states.sort_values(by='population_female',ascending=True)
female = np.array(group_by_states['population_female'])

male   = np.array(group_by_states['population_male'])



X = np.arange(female.size)

yLabel = group_by_states.state_name

fig, ax = plt.subplots(figsize=(20,20))

ax.set_title("Gender Composition in States")

width = 0.4

ax.barh(X, male, width, color='#0080ff', label='Male')

ax.barh(X + width, female, width, color='#FF69B4', label='Female')

ax.set(yticks=X + width, yticklabels=yLabel, ylim=[2*width - 1, len(male)])

ax.tick_params(axis='both', labelsize=20)

ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%8d'))

ax.legend()

plt.show()
group_by1=df.groupby(['state_name'])[["0-6_population_male", "0-6_population_female"]].sum()

group_by1=group_by1.reset_index()

group_by1=group_by1.sort_values(by='0-6_population_male',ascending=True)



kids_F = np.array(group_by1['0-6_population_female'])

kids_M   = np.array(group_by1['0-6_population_male'])



X = np.arange(kids_F.size)

yLabel = group_by1.state_name

fig, ax = plt.subplots(figsize=(20,20))

ax.set_title("Gender Composition by Age below 6")

width = 0.4

ax.barh(X, kids_M, width, color='#0080ff', label='Male Kids')

ax.barh(X + width, kids_F, width, color='#FF69B4', label='Female Kids')

ax.set(yticks=X + width, yticklabels=yLabel, ylim=[2*width - 1, len(kids_M)])

ax.tick_params(axis='both', labelsize=20)

ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%8d'))

ax.legend()

plt.show()
group_by2=df.groupby(['state_name'])[['female_graduates', 'male_graduates']].sum()

group_by2=group_by2.reset_index()

group_by2=group_by2.sort_values(by='female_graduates',ascending=True)



grad_F = np.array(group_by2['female_graduates'])

grad_M   = np.array(group_by2['male_graduates'])



X = np.arange(grad_F.size)

yLabel = group_by2.state_name

fig, ax = plt.subplots(figsize=(20,20))

ax.set_title("Graduates by Gender")

width = 0.4

ax.barh(X, grad_M, width, color='#0080ff', label='Male Graduates')

ax.barh(X + width, grad_F, width, color='#FF69B4', label='Female Graduates')

ax.set(yticks=X + width, yticklabels=yLabel, ylim=[2*width - 1, len(grad_F)])

ax.tick_params(axis='both', labelsize=20)

ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%8d'))

ax.legend()

plt.show()

print('In Kerala & Punjab Female graduates are higher than male graduates')
group_by3=df[['population_male','population_female']].sum()



fig, ax = plt.subplots(figsize=(16,8))

# plot chart

ax = plt.subplot(121, aspect='equal')

ax.set_title("Gender Composition")

group_by3.plot(kind='pie', y = 'population_total', ax=ax, autopct='%1.1f%%', 

 startangle=90, shadow=False, labels=group_by3.index, legend = False, fontsize=10)
group_by4=df[['male_graduates','female_graduates']].sum()



fig, ax = plt.subplots(figsize=(16,8))

# plot chart

ax = plt.subplot(121, aspect='equal')

ax.set_title("Male vs Female graduates")

group_by4.plot(kind='pie', y = 'graduates_total', ax=ax, autopct='%1.1f%%', 

 startangle=90, shadow=False, labels=group_by4.index, legend = False, fontsize=10)