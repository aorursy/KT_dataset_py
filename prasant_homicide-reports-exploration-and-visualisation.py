import pandas as pd

import numpy as np
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams["figure.figsize"] = (15,5) 
df = pd.read_csv("../input/database.csv")
df.head()
df.shape
df.set_index(["Record ID"],inplace = True)
df.head()
df.apply(lambda x: sum(x.isnull()),axis=0)
df.describe()
df["Weapon"].head()
plt.rcParams["figure.figsize"] = (12,4)

df["Weapon"].value_counts().plot(kind = "bar")

plt.title('Deaths Attributable to Various Weapons')
plt.rcParams["figure.figsize"] = (10,10)

df["Weapon"].value_counts().plot(kind = "bar")

plt.title('Deaths Attributable to Various Weapons')
plt.rcParams["figure.figsize"] = (12,4)

plt.yscale('log', nonposy='clip')

df["Weapon"].value_counts().plot(kind = "bar")

plt.title('Deaths Attributable to Various Weapons')
df[df["Crime Solved"] != "Yes"].shape
unsolved = df[df["Crime Solved"] != "Yes"]
unsolved.head()
unsolved.describe()
plt.rcParams["figure.figsize"] = (12,4)

unsolved['Year'].value_counts().sort_index(ascending=True).plot(kind='line')

plt.title('Number of Unsolved Homicides: 1980 to 2014')
dict_states = {'Alaska':'AK','Alabama':'AL','Arkansas':'AR','Arizona':'AZ', 'California':'CA', 'Colorado':'CO', 'Connecticut':'CT', 

'District of Columbia':'DC', 'Delaware':'DE', 'Florida':'FL', 'Georgia':'GA', 'Hawaii':'HI', 'Iowa':'IA', 

'Idaho':'ID', 'Illinois':'IL', 'Indiana':'IN', 'Kansas':'KS', 'Kentucky':'KY', 'Louisiana':'LA', 

'Massachusetts':'MA', 'Maryland':'MD', 'Maine':'ME', 'Michigan':'MI', 'Minnesota':'MN', 'Missouri':'MO', 

'Mississippi':'MS', 'Montana':'MT', 'North Carolina':'NC', 'North Dakota':'ND', 'Nebraska':'NE', 

'New Hampshire':'NH', 'New Jersey':'NJ', 'New Mexico':'NM', 'Nevada':'NV', 'New York':'NY', 'Ohio':'OH', 

'Oklahoma':'OK', 'Oregon':'OR', 'Pennsylvania':'PA', 'Puerto Rico':'PR', 'Rhode Island':'RI', 

'South Carolina':'SC', 'South Dakota':'SD', 'Tennessee':'TN', 'Texas':'TX', 'Utah':'UT', 

'Virginia':'VA', 'Vermont':'VT', 'Washington':'WA', 'Wisconsin':'WI', 'West Virginia':'WV', 'Wyoming':'WY'}
abb_st = [val for val in dict_states.values()]    

len(abb_st)
plt.rcParams["figure.figsize"] = (12,4)

ax = sns.countplot(x="State", hue="Weapon", data=unsolved[unsolved["Weapon"]=="Handgun"])

ax.set_xticklabels(abb_st)

plt.title("Unsolved Homicides Caused By Handguns")
unsolved['Weapon'].value_counts()
plt.rcParams["figure.figsize"] = (12,4)
unsolved['Weapon'].value_counts().plot(kind='bar')
rel = unsolved['Weapon'].groupby(unsolved['Victim Sex'])
rel.size().plot(kind='bar')
unsolved["Month"].value_counts().plot(kind="bar")
unsolved["Agency Type"].value_counts().plot(kind="bar")

#plt.yscale('log', nonposy='clip')
unsolved["Crime Type"].unique()
pot_sk = unsolved[unsolved["Crime Type"] == "Murder or Manslaughter"]

pot_sk.head()
pot_sk.shape
pot_sk["City"].value_counts().head(10).plot(kind="bar")

plt.title("Top 10 Cities: Unsolved Murders or Manslaughters")
pot_sk["City"].value_counts().tail(10).plot(kind="bar")

plt.title("Bottom 10 Cities: Unsolved Murders or Manslaughters")



pot_sk.groupby("City").filter(lambda x: len(x)>5)
two_or_more = pot_sk.groupby("City").filter(lambda x: len(x)>5)

two_or_more["City"].value_counts().tail(10).plot(kind="bar")
df["Relationship"].unique()
known = df[df["Relationship"] != "Unknown"]

known.head()
known["Relationship"].value_counts()
plt.rcParams["figure.figsize"] = (12,4)

known["Relationship"].value_counts().plot(kind="bar")

plt.title("Relationsip of Victim to Perpetrator")

plt.yscale('log', nonposy='clip')
df.head(2)
df["Perpetrator Race"].unique()
df.columns
pd.pivot_table(known,index=["Victim Race","Perpetrator Race"],values=["Victim Count"],aggfunc=[np.sum])

               #columns=["Product"],aggfunc=[np.sum])