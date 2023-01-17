import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go
df = pd.read_csv("../input/indian-food-101/indian_food.csv");

df.head()
df.shape
df=df.replace(-1,np.nan)

df=df.replace('-1',np.nan)

df.fillna(0)
df.info()
total_state = list(df['state'].value_counts().index)

total_state
Veg = (df['diet'].str[0].str.lower() == 'v').sum()



NonVeg = (df['diet'].str[0].str.lower() == 'n').sum()



names = ["Vegetarian", "Non-Vegetarian"]

values = [Veg, NonVeg]



fig, axs = plt.subplots(1, 1, figsize=(8, 6), sharey=True)

axs.bar(names, values)
labels = names

values = [Veg, NonVeg]



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()
print("Total Number of Vegetarian dishes: ", Veg)

print("Total Number of Non-Vegetarian dishes: ", NonVeg)
plt.rcParams['figure.figsize'] = (24,6)

sns.countplot(df['state'])

plt.title("State wise Count")

plt.xticks(rotation=90)

plt.show()
fig = px.histogram(df.dropna(),x='region',color = 'region', title="Region wise count")

fig.show()
total_flavor_profiles = list(df['flavor_profile'].value_counts().index)

total_flavor_profiles 



flavor_count = list(df['flavor_profile'].value_counts())

flavor_count



print(total_flavor_profiles)

print(flavor_count)
labels = total_flavor_profiles

values = flavor_count



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()
total_course = list(df['course'].value_counts().index)

total_course 



course_count = list(df['course'].value_counts())

course_count



print(total_course)

print(course_count)
labels = total_course

values = course_count



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()
fig = px.bar(df, x="state", y="course", color="course", title="State wise Courses")

fig.show()
fig = px.histogram(df, x='prep_time',title = 'Preperation Time')

fig.show()
fig = px.histogram(df, x='cook_time', title = 'Cooking Time')

fig.show()