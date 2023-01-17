import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # used to draw barplot

# Read museum file into dataframe
museum_df = pd.read_csv("../input/museums.csv")
museum_df = museum_df.dropna(axis=0,subset=["State (Administrative Location)"])
museum_df = museum_df[museum_df["Museum Type"].isin(["ART MUSEUM", "SCIENCE & TECHNOLOGY MUSEUM OR PLANETARIUM"])]
# Groupby state, museum type, count museums
museums_by_state = museum_df.groupby(["State (Administrative Location)","Museum Type"]).count()

museums_by_state = museums_by_state.reset_index()
museums_by_state = museums_by_state.rename(columns={"State (Administrative Location)": "State", "Museum Name": "Number of Museums"})
# Shorten names
museums_by_state["Museum Type"].replace(["ART MUSEUM", "SCIENCE & TECHNOLOGY MUSEUM OR PLANETARIUM"], ["Art Museum", "Science Museum"], inplace=True)

# Need to make extra long so that all 50 states fit
sns.set(style="whitegrid", font_scale=0.8, rc={'figure.figsize':(16,6)})
art_science_plot = sns.barplot("State","Number of Museums",hue="Museum Type",data=museums_by_state, palette="bright", saturation=1.0)