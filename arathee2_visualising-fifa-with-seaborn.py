import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv("../input/FullData.csv", parse_dates = ["Club_Joining", "Contract_Expiry", "Birth_Date"])

pd.set_option('display.max_columns', df.shape[1])

df.head()
print(df.shape)

print(df.info())
# check for missing values

df.isnull().sum()
# print the name of the player with missing Club_Position

df.Name.loc[df['Club_Position'].isnull()]
# extracting height and weight

df.Height = df.Height.astype(str)

df.Weight = df.Weight.astype(str)



df.Height = df.Height.apply(lambda height: height.replace(" cm", ""))

df.Weight = df.Weight.apply(lambda weight: weight.replace(" kg", ""))



df.head()



df.Height = df.Height.astype("float64")

df.Weight = df.Weight.astype("float64")

df.head(10)
# ball control

plt.clf()

sns.distplot(df.Ball_Control, color = "#f15453")

plt.show()
# Nationality

plt.figure(figsize=(15, 30))

sns.countplot(y = df.Nationality)

plt.xlabel("Player Count")

plt.show()
# National_Position

sns.countplot(data=df, x = "National_Position")

plt.ylabel("Player Count")

plt.xticks(rotation=45)

plt.show()
sns.stripplot(data=df, x = "National_Position", y = "Rating", jitter=True)

plt.xticks(rotation="45")

plt.show()
# Swarm plot

sns.swarmplot(data=df, x = "National_Position", y = "Rating")

plt.xticks(rotation="45")

plt.show()
# Boxplot

sns.boxplot(data=df, x = "National_Position", y = "Rating")

plt.xticks(rotation="45")

plt.show()
plt.figure(figsize=(30,10))

sns.violinplot(data=df, x = "National_Position", y = "Rating")

plt.xticks(rotation="45")

plt.show()
plt.clf()

sns.lvplot(data=df, x = "National_Position", y = "Rating", palette=sns.color_palette("muted", n_colors=27))

plt.xticks(rotation="45")

plt.show()
sns.pointplot(data=df, x = "National_Position", y = "Rating")

plt.xticks(rotation="45")

plt.show()
# ball control and vision

plt.clf()

sns.regplot(y = "Ball_Control", x = "Vision", data = df, color = "#faaf87", scatter_kws={"alpha":0.2, "s":40})

plt.show()
# ball control and agility

plt.clf()

sns.regplot(y = "Ball_Control", x = "Agility", data = df, color = "#873687", scatter_kws={"alpha":0.2, "s":40})

plt.show()
plt.clf()

plt.figure(figsize=(25,25))

cmap = sns.diverging_palette(20, h_pos=220, s=75, l=50, sep=10, center='light', as_cmap=True)

corr_matrix = df.iloc[1:1000, 18:df.shape[1]].corr()

sns.heatmap(corr_matrix, cmap=cmap, annot=True)

plt.show()
sns.rugplot(df.Agility)

plt.show()
sns.kdeplot(df.Agility, df.Rating)

plt.show()
sns.residplot(data=df, x="Agility", y="Rating", scatter_kws={"s":20, "alpha":0.3})

plt.show()
sns.lmplot(data=df, x="Agility", y="Rating", scatter_kws={"s":20, "alpha":0.3})

plt.show()
sns.jointplot(data=df, x="Agility", y="Rating", joint_kws={"alpha":0.2}, color = "#3423a1")

plt.show()
print(df.iloc[:,19:22].head())

sns.pairplot(data=df.iloc[:,19:22])

plt.show()
sns.factorplot(data=df, x="National_Position", y="Rating", row = "Preffered_Foot", kind="box")

plt.xticks(rotation=90)

plt.show()
# clustering

X = df.iloc[:, 18:df.shape[1]]

from sklearn.cluster import KMeans

model = KMeans(n_clusters = 2)

model.fit(X)
print(model.inertia_)

print(model.cluster_centers_)

Clusters = model.predict(X)

print(Clusters[1:10])

df["Clusters"] = pd.Series(Clusters)

df.Clusters.value_counts()
# bring "clusters" column to first place

df = df.loc[:, ['Clusters', 'Name', 'Nationality', 'National_Position', 'National_Kit', 'Club',

       'Club_Position', 'Club_Kit', 'Club_Joining', 'Contract_Expiry',

       'Rating', 'Height', 'Weight', 'Preffered_Foot', 'Birth_Date', 'Age',

       'Preffered_Position', 'Work_Rate', 'Weak_foot', 'Skill_Moves',

       'Ball_Control', 'Dribbling', 'Marking', 'Sliding_Tackle',

       'Standing_Tackle', 'Aggression', 'Reactions', 'Attacking_Position',

       'Interceptions', 'Vision', 'Composure', 'Crossing', 'Short_Pass',

       'Long_Pass', 'Acceleration', 'Speed', 'Stamina', 'Strength', 'Balance',

       'Agility', 'Jumping', 'Heading', 'Shot_Power', 'Finishing',

       'Long_Shots', 'Curve', 'Freekick_Accuracy', 'Penalties', 'Volleys',

       'GK_Positioning', 'GK_Diving', 'GK_Kicking', 'GK_Handling',

       'GK_Reflexes']]

df.head(10)
sns.boxplot(data=df, x="Clusters", y="Rating")

plt.show()