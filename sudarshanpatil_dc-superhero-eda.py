import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import warnings as ws

ws.filterwarnings("ignore")
df = pd.read_csv("/kaggle/input/comic-characters/dc-wikia-data_csv.csv")

df.head()
print(df.isna().sum().to_frame())

print(df.shape)

# df.drop("gsm", axis = 1, inplace = True)
df["alive"].value_counts()
df["align"] = df["align"].fillna("bad characters")



df["id"] = df["id"].fillna ("publi identity")



df["eye"] = df["eye"].fillna("blue eyes")



df["hair"] = df["hair"].fillna("black hair")



df["sex"] = df["sex"].fillna("male characters")



df["alive"] = df["alive"].fillna("living characters")

# Marking the NA values as 0 for the seck of the analaysis

df["appearances"] = df["appearances"].fillna(0)



df["first appearance"] = df["first appearance"].fillna(0)



df["year"] = df["year"].fillna(0)



df[["appearances", "year"]] = df[["appearances", "year"]].astype(int)
print(df.isna().sum().to_frame())
top_appearances = df.sort_values(by= "appearances", ascending=False)[:10][["name", "appearances"]]



fig = px.bar(data_frame=top_appearances, x="name", y="appearances", color= "appearances", text = "appearances")

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', title_text ="Most Appeared Superheros", xaxis_title = "The Superheros", yaxis_title = "Appearance Count")

fig.show()
import plotly.graph_objects as go



df["id"] = df["id"].replace({"publi identity": "public identity" })



identity = df["id"].value_counts().reset_index().rename(columns = {"index": "identity", "id": "count"})



fig = go.Figure([go.Pie(labels=identity['identity'], values=identity['count'])])



fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=15,insidetextorientation='radial')



fig.update_layout(title="Identity status of Superhero",title_x=0.5)

fig.show()
publically_known_superhero = df[df["id"] == "public identity"].sort_values(by="appearances", ascending = False)[["name", "appearances"]][:5]
fig = go.Figure(go.Bar(

    x=publically_known_superhero["name"],y= publically_known_superhero["appearances"],

    marker = { 'color': publically_known_superhero["appearances"], 'colorscale': "darkmint"},

             text = publically_known_superhero["appearances"],

             textposition = 'outside'

             

))



fig.update_layout(title_text= "Top five publically known Superheros", xaxis_title = "Superheros", yaxis_title = "Count")



fig.show()
publically_unknown_superhero = df[df["id"] == "secret identity"].sort_values(by="appearances", ascending = False)[["name", "appearances"]][:5]

fig = go.Figure(go.Bar(

    x=publically_unknown_superhero["name"],y= publically_unknown_superhero["appearances"],

    marker = { 'color': publically_unknown_superhero["appearances"], 'colorscale': "deep"},

             text = publically_unknown_superhero["appearances"],

             textposition = 'outside'

             

))



fig.update_layout(title_text= "Top five publically un-known(hiddent) Superheros", xaxis_title = "Superheros", yaxis_title = "Count")



fig.show()
import plotly.graph_objects as go



status = df["align"].value_counts().reset_index().rename(columns = {"index": "status", "align": "count"})



fig = go.Figure([go.Pie(labels=status['status'], values=status['count'])])



fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=15,insidetextorientation='radial', hole = 0.2)



fig.update_layout(title="Character Status of Superhero (Good or BAD )",title_x=0.5)

fig.show()
fig, ax = plt.subplots(1,2, figsize = (30,10))

g = sns.barplot(data = df[df["align"] == "good characters"][["name", "appearances"]][:10], x= "name", y="appearances", ax = ax[0], palette="Blues")

g.set_xticklabels(g.get_xticklabels(), rotation=20)

ax[0].set_title("Top 10 Good Characters")



g = sns.barplot(data= df[df["align"] == "bad characters"][["name", "appearances"]][:10],  x= "name", y="appearances", ax = ax[1], palette="Greens")

g.set_xticklabels(g.get_xticklabels(), rotation=20)

ax[1].set_title("Top 10 BAD Characters")



plt.show()
eye_type = df["eye"].value_counts().reset_index().rename(columns = {"index": "eye_type", "eye": "count"})

fig = go.Figure(go.Bar(

    x=eye_type["eye_type"],y= eye_type["count"],

    marker = { 'color': eye_type["count"], 'colorscale': "inferno"},

             text = eye_type["count"],

             textposition = 'outside'

             

))



fig.update_layout(title_text= "Distinct eye types of the Superheros", xaxis_title = "Eye types of superhero", yaxis_title = "Count")



fig.show()
sns.set()

plt.figure(figsize= (30,10))

top_appearances_with_living = df.sort_values(by= "appearances", ascending=False)[:10][["name", "appearances", "alive"]]

g = sns.barplot(data = top_appearances_with_living, x="name", y="appearances", palette="inferno", hue = "alive")

g.set_xlabel("Superheros")

g.set_title("Living status oftop 10 superheros", size = 20)

g.set_ylabel("count")



plt.show()
before_2000 = df[df["year"] > 2000][["name", "appearances"]][:5]

fig = go.Figure(go.Bar(

    x=before_2000["name"],y= before_2000["appearances"],

    marker = { 'color': before_2000["appearances"], 'colorscale': "Blues"},

             text = before_2000["appearances"],

             textposition = 'outside'

             

))



fig.update_layout(title_text= "Top five Superheros appeared after the year 2000", xaxis_title = "Superheros", yaxis_title = "Count")



fig.show()
after_2000 = df[df["year"] < 2000][["name", "appearances"]][:5]

fig = go.Figure(go.Bar(

    x=after_2000["name"],y= after_2000["appearances"],

    marker = { 'color': after_2000["appearances"], 'colorscale': "greens"},

             text = after_2000["appearances"],

             textposition = 'outside'

             

))



fig.update_layout(title_text= "Top five Superheros appeared before the year 2000", xaxis_title = "Superheros", yaxis_title = "Count")



fig.show()
sex_classificatio = df["sex"].value_counts().reset_index().rename(columns = {"index": "gender", "sex": "count"})
tp = px.pie(sex_classificatio, values='count', names='gender', color_discrete_sequence=px.colors.sequential.RdBu)





tp.update_traces(hoverinfo='label+percent', textinfo='value+percent')



tp.update_layout(title="Gender Wise classification of the superheros",title_x=0.5)

tp.show()
sns.set()

plt.figure(figsize= (30,10))

top_appearances_with_living = df.sort_values(by= "appearances", ascending=False)[:10][["name", "appearances", "alive", "sex"]]

g = sns.barplot(data = top_appearances_with_living, x="name", y="appearances", palette="inferno", hue = "sex")

g.set_xticklabels(g.get_xticklabels(), rotation=20)

g.set_xlabel("Superheros")

g.set_title("Male-female distribution in top 10 superheros", size = 20)

g.set_ylabel("count")



plt.show()