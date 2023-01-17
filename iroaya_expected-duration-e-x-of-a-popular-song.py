import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import plotly.express as px

import random
df = pd.read_csv("../input/top-spotify-songs-from-20102019-by-year/top10s.csv", encoding="ISO-8859-1" )
df.head()
df.describe()

#seems that there is an error where 0's are the values
df.loc[df["bpm"] == 0 ]
df.drop(index=442, inplace = True)
df.describe()
grouped_by_bpm= df.groupby(['bpm'])['Unnamed: 0']

grouped_by_bpm = grouped_by_bpm.count()

# Convert to DataFrame

df_grouped_by_bpm = pd.DataFrame(grouped_by_bpm)

# Preview DataFrame

df_grouped_by_bpm.reset_index(inplace = True)

df_grouped_by_bpm

df_bpm = df_grouped_by_bpm
plt.figure(figsize=(16,10), dpi= 80)

plt.bar(df_bpm['bpm'], df_bpm['Unnamed: 0'], color="blue", width=.5)



grouped_by_dB= df.groupby(['dB'])['Unnamed: 0']

grouped_by_dB = grouped_by_dB.count()

# Convert to DataFrame

df_dB = pd.DataFrame(grouped_by_dB)

df_dB.rename(columns={"Unnamed: 0": "Count of Songs"}, inplace = True)

# Preview DataFrame

df_dB.reset_index(inplace = True)

df_dB.sort_values('dB')





fig = px.line(df_dB, 'dB', 'Count of Songs')

fig.show()

df_dB
grouped_by_dur= df.groupby(['dur'])['Unnamed: 0']

grouped_by_dur = grouped_by_dur.count()

df_dur = pd.DataFrame(grouped_by_dur)

df_dur.reset_index(inplace = True)

df_dur["Durration in Minutes"] = [float(x/60) for x in df_dur['dur']]

df_dur.rename(columns={"Unnamed: 0": "Count of Songs"}, inplace = True)

fig = px.bar(df_dur, 'Durration in Minutes','Count of Songs')

fig.show()



df_dur_min = pd.DataFrame(df_dur.groupby(['Durration in Minutes'])['Count of Songs'].sum())

df_dur_min.reset_index(inplace = True)



count_of_durrations = df["dur"].count()





df_dur_min["weight_of_durration"] = [x/(count_of_durrations) for x in df_dur_min["Count of Songs"]]

df_dur_min["multiplied_by_weight"] = df_dur_min["Durration in Minutes"]*df_dur_min["weight_of_durration"]



ex = sum(df_dur_min["multiplied_by_weight"])



print('The expected value a popular song durration is ' +  str(ex))
#vocal presense analysis

grouped_by_spch= df.groupby(['spch'])['Unnamed: 0']

grouped_by_spch = grouped_by_spch.count()

df_spch = pd.DataFrame(grouped_by_spch)

df_spch.reset_index(inplace = True)

df_spch.rename(columns={"Unnamed: 0": "Count of Songs"}, inplace = True)



top_10_spch = df_spch.sort_values('Count of Songs', ascending = False).head(10)

fig = px.bar(top_10_spch, 'spch','Count of Songs')

fig.show()