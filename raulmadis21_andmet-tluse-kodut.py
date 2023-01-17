import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline 

df = pd.read_csv("../input/Mass Shootings Dataset Ver 5.csv", encoding='latin-1')

df.columns = df.columns.str.lower().str.replace(' ', '_')













df.policeman_killed.plot.hist();
df["age"] = pd.to_numeric(df['age'], errors='coerce') # Kõik floatideks, kui error siis muudab NaN väärtuseks

df = df[pd.notnull(df['age'])] # Kustutab read kus Age on NaN väärtus

df.plot.scatter("age", "fatalities", alpha=0.2); # Teeb scatterploti

 
df["race"] = df["race"].str.capitalize()

df.groupby("race")["fatalities"].mean().round()