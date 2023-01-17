import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
df = pd.read_csv("../input/outbreaks.csv")

df2 = pd.pivot_table(df, index='Month', values='Illnesses', aggfunc='count')

df2 = df2.reindex(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])

ax = df2.plot(kind='bar', color='steelblue', grid=True)

plt.title('Foodborne Illnesses Cases By Month')

plt.ylabel('Illiness Cases')



rects = ax.patches

labels = [df2.iloc[i] for i in range(len(rects))]

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')





fig_size = plt.rcParams["figure.figsize"]

print("Current size") , fig_size

fig_size[0] = 12

fig_size[1] = 9

plt.rcParams["figure.figsize"] = fig_size
# Foodborne Disease Outbreaks, 1998 - 2015

# 1. Has foodborne disease outbreaks gotten worse over the years? 2. Which month do they tend to occur the most? 

# 3. Which state of the US do they appear the most frequently?



# Has foodborne disease outbreaks gotten worse over the years? And which month do they tend to occur the most, and what state of 

# the United States do they appear more frequently? Thanks to the public health agencies that operate in all 50 states for 

# investigating outbreaks and gathering all this data from 1998 through 2015 and submitting it to the Centers for Disease Control 

# and Prevention (CDC). We will be able to display a clearer picture and pattern for foodborne disease outbreaks. A foodborne 

# disease outbreak is considered an outbreak when two or more people get the same illness after ingestion of a common food/liquid

# and epidemiological analysis implicates the food as the source of the illness, according to the CDC. 
df2 = pd.pivot_table(df, index='State', values='Illnesses', aggfunc='count')

ax = df2.plot(kind='bar', color='steelblue', grid=True)

plt.title('Foodborne Illnesses Cases By State')

plt.ylabel('Illiness Cases')



rects = ax.patches

labels = [df2.iloc[i] for i in range(len(rects))]

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

fig_size = plt.rcParams["figure.figsize"]

print("Current size") , fig_size

fig_size[0] = 12

fig_size[1] = 9

plt.rcParams["figure.figsize"] = fig_size



plt.show()
# In this graph it's pretty clear that California and Florida is the center of foodborne illnesses apparently. The high rate of 

# incidents could be related to their population rate during the years of 1998-2015. One would have to investigate significant 

# events that has transpired in those years that might have increased the chances of causing more foodborne illnesses in other

# states than others. An exceptional circumstance would be the outbreak of Salmonella Infections linked to Peanut Butter, 

# 2008 - 2009. Majority of peanut butter was sent to California and Ohio.

df2 = pd.pivot_table(df, index='Year', values='Illnesses', aggfunc='count')

ax = df2.plot(kind='bar', color='steelblue')

plt.title('Foodborne Illnesses Cases By Year')

plt.ylabel('Illiness Cases')



rects = ax.patches

labels = [df2.iloc[i] for i in range(len(rects))]

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

fig_size = plt.rcParams["figure.figsize"]

print("Current size") , fig_size

fig_size[0] = 12

fig_size[1] = 9

plt.rcParams["figure.figsize"] = fig_size



plt.show()

# As it can be seen on this graph, the trend of foodborne illnesses has decreased overall since 1998. Significant drop is from 

# 2008 - 2009 but now seems to be steady at the 800 - 900 cases ranges. Looking at this data, one can come to the conclusion that 

# foodborne diseases are occurring less frequently.  After my research, I would have to agree, but still skeptical. Reasoning

# behind that is because most foodborne infections go undiagnosed and unreported, either because the sick person doesn't see a

# doctor(that's me all the time), or the doctor doesn’t made a specific diagnoses according to the 

# CDC. 



# The data shows the CDC and FDA(food and drug administration) that foodborne illnesses is decreasing and which months its at its

# highest and which states it affects the populace more. But little information as to why the data displays it. Gathering data on 

# the after match of regulations and policies that were created by the FDA could explain what impact they have or don't have.
df.isnull().sum()
# data anaylsis quality as you can tell from this function is missing a lot of data from ingredients of food that were the

# cause of illinesses. Also, missing serotype(classification of a bacteria) and Genotype(genetic trait). Both of these better

# classify what exactly disease caused the outbreak illnesses. I felt it best to graph what had the least amount of null data

# points, which was illnesses, year, month, and state. 