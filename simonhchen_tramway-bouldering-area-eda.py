# Data Handling and Visualization

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np
# Read the data and take a quick look at the data.

tramway_data = pd.read_csv("../input/san-jacinto-mountain-tramway-bouldering/formatted_tramway_data.csv")

tramway_data = tramway_data.drop(['Unnamed: 0'], axis=1)

tramway_data.head()
tramway_areas = pd.DataFrame(tramway_data['Area'].value_counts())

tramway_areas = pd.DataFrame(tramway_areas.reset_index())

tramway_areas = tramway_areas.rename(columns={"index": "Area", "Area": "Area Problem Count"})

tramway_areas
Area = tramway_areas["Area"]

Area_problem_count = tramway_areas["Area Problem Count"]



plt.style.use('classic')



plt.subplots(figsize=(6,8))

plt.barh(Area, Area_problem_count)

plt.ylabel("Area")

plt.xlabel("Area Problem Count")

plt.title("Tramway Areas by Number of Problems")

plt.yticks(Area)

plt.gca().invert_yaxis()



plt.show()
fig1, ax1 = plt.subplots(figsize=(15,15))



plt.style.use('classic')

plt.title("Tramway Areas by Number of Problems", y=1.05)

ax1.pie(Area_problem_count, labels=Area, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
Conquest_Ridge = tramway_data[tramway_data['Area'] == 'Conquest Ridge']

Conquest_Ridge_Ratings = pd.DataFrame(Conquest_Ridge['Rating'].value_counts())

Conquest_Ridge_Ratings = pd.DataFrame(Conquest_Ridge_Ratings.reset_index())

Conquest_Ridge_Ratings = Conquest_Ridge_Ratings.rename(columns={"index": "Rating", "Rating": "Rating Count"})



Ratings = Conquest_Ridge_Ratings["Rating"]

Rating_count = Conquest_Ridge_Ratings["Rating Count"]



plt.style.use('classic')



plt.subplots(figsize=(6,8))

plt.barh(Ratings, Rating_count)

plt.ylabel("Ratings")

plt.xlabel("Rating Count")

plt.title("Boulder Problems in Conquest Ridge by Rating")

plt.yticks(Ratings)

plt.gca().invert_yaxis()



plt.show()
fig1, ax1 = plt.subplots(figsize=(15,15))



plt.style.use('classic')

plt.title("Boulder Problems in Conquest Ridge by Rating", y=1.05)

ax1.pie(Rating_count, labels=Ratings, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
print(Conquest_Ridge['Sub-Area'].value_counts())

print(Conquest_Ridge.shape)

Conquest_Ridge
SL_Valley = tramway_data[tramway_data['Area'] == 'Shangri-La Valley']

SL_Valley_Ratings = pd.DataFrame(SL_Valley['Rating'].value_counts())

SL_Valley_Ratings = pd.DataFrame(SL_Valley_Ratings.reset_index())

SL_Valley_Ratings = SL_Valley_Ratings.rename(columns={"index": "Rating", "Rating": "Rating Count"})



Ratings = SL_Valley_Ratings["Rating"]

Rating_count = SL_Valley_Ratings["Rating Count"]



plt.style.use('classic')



plt.subplots(figsize=(6,8))

plt.barh(Ratings, Rating_count)

plt.ylabel("Ratings")

plt.xlabel("Rating Count")

plt.title("Boulder Problems in Shangri-La Valley by Rating")

plt.yticks(Ratings)

plt.gca().invert_yaxis()



plt.show()
# TO-DO: Clean Pie Chart Up so easier to read



fig1, ax1 = plt.subplots(figsize=(20,20))



plt.style.use('classic')

plt.title("Boulder Problems in Shangri-La Valley by Rating", y=1.05)

ax1.pie(Rating_count, labels=Ratings,

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
Long_Valley = tramway_data[tramway_data['Area'] == 'Long Valley']

Long_Valley_Ratings = pd.DataFrame(Long_Valley['Rating'].value_counts())

Long_Valley_Ratings = pd.DataFrame(Long_Valley_Ratings.reset_index())

Long_Valley_Ratings = Long_Valley_Ratings.rename(columns={"index": "Rating", "Rating": "Rating Count"})



Ratings = Long_Valley_Ratings["Rating"]

Rating_count = Long_Valley_Ratings["Rating Count"]



plt.style.use('classic')



plt.subplots(figsize=(6,8))

plt.barh(Ratings, Rating_count)

plt.ylabel("Ratings")

plt.xlabel("Rating Count")

plt.title("Boulder Problems in Long Valley by Rating")

plt.yticks(Ratings)

plt.gca().invert_yaxis()



plt.show()
fig1, ax1 = plt.subplots(figsize=(20,20))



plt.style.use('classic')

plt.title("Boulder Problems in Long Valley by Rating", y=1.05)

ax1.pie(Rating_count, labels=Ratings,

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
FG_Area = tramway_data[tramway_data['Area'] == 'Flash Gordon Area']

FG_Area_Ratings = pd.DataFrame(FG_Area['Rating'].value_counts())

FG_Area_Ratings = pd.DataFrame(FG_Area_Ratings.reset_index())

FG_Area_Ratings = FG_Area_Ratings.rename(columns={"index": "Rating", "Rating": "Rating Count"})



Ratings = FG_Area_Ratings["Rating"]

Rating_count = FG_Area_Ratings["Rating Count"]



plt.style.use('classic')



plt.subplots(figsize=(6,8))

plt.barh(Ratings, Rating_count)

plt.ylabel("Ratings")

plt.xlabel("Rating Count")

plt.title("Boulder Problems in Flash Gordon Area by Rating")

plt.yticks(Ratings)

plt.gca().invert_yaxis()



plt.show()
fig1, ax1 = plt.subplots(figsize=(20,20))



plt.style.use('classic')

plt.title("Boulder Problems in Flash Gordon Area by Rating", y=1.05)

ax1.pie(Rating_count, labels=Ratings,

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
UD_Country = tramway_data[tramway_data['Area'] == 'Undiscovered Country']

UD_Country_Ratings = pd.DataFrame(UD_Country['Rating'].value_counts())

UD_Country_Ratings = pd.DataFrame(UD_Country_Ratings.reset_index())

UD_Country_Ratings = UD_Country_Ratings.rename(columns={"index": "Rating", "Rating": "Rating Count"})



Ratings = UD_Country_Ratings["Rating"]

Rating_count = UD_Country_Ratings["Rating Count"]



plt.style.use('classic')



plt.subplots(figsize=(6,8))

plt.barh(Ratings, Rating_count)

plt.ylabel("Ratings")

plt.xlabel("Rating Count")

plt.title("Boulder Problems in Undiscovered Country by Rating")

plt.yticks(Ratings)

plt.gca().invert_yaxis()



plt.show()
fig1, ax1 = plt.subplots(figsize=(20,20))



plt.style.use('classic')

plt.title("Boulder Problems in Undiscovered Country by Rating", y=1.05)

ax1.pie(Rating_count, labels=Ratings,

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
Trailside_Boulders = tramway_data[tramway_data['Area'] == 'Trailside Boulders']

Trailside_Boulders_Ratings = pd.DataFrame(Trailside_Boulders['Rating'].value_counts())

Trailside_Boulders_Ratings = pd.DataFrame(Trailside_Boulders_Ratings.reset_index())

Trailside_Boulders_Ratings = Trailside_Boulders_Ratings.rename(columns={"index": "Rating", "Rating": "Rating Count"})



Ratings = Trailside_Boulders_Ratings["Rating"]

Rating_count = Trailside_Boulders_Ratings["Rating Count"]



plt.style.use('classic')



plt.subplots(figsize=(6,8))

plt.barh(Ratings, Rating_count)

plt.ylabel("Ratings")

plt.xlabel("Rating Count")

plt.title("Boulder Problems in Trailside Boulders by Rating")

plt.yticks(Ratings)

plt.gca().invert_yaxis()



plt.show()
fig1, ax1 = plt.subplots(figsize=(20,20))



plt.style.use('classic')

plt.title("Boulder Problems in Trailside Boulders by Rating", y=1.05)

ax1.pie(Rating_count, labels=Ratings,

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
KF_Canyon = tramway_data[tramway_data['Area'] == 'Kung Fu Canyon']

KF_Canyon_Ratings = pd.DataFrame(KF_Canyon['Rating'].value_counts())

KF_Canyon_Ratings = pd.DataFrame(KF_Canyon_Ratings.reset_index())

KF_Canyon_Ratings = KF_Canyon_Ratings.rename(columns={"index": "Rating", "Rating": "Rating Count"})



Ratings = KF_Canyon_Ratings["Rating"]

Rating_count = KF_Canyon_Ratings["Rating Count"]



plt.style.use('classic')



plt.subplots(figsize=(6,8))

plt.barh(Ratings, Rating_count)

plt.ylabel("Ratings")

plt.xlabel("Rating Count")

plt.title("Boulder Problems in Kung Fu Canyon by Rating")

plt.yticks(Ratings)

plt.gca().invert_yaxis()



plt.show()
fig1, ax1 = plt.subplots(figsize=(20,20))



plt.style.use('classic')

plt.title("Boulder Problems in Kung Fu Canyon by Rating", y=1.05)

ax1.pie(Rating_count, labels=Ratings,

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
Icebergs = tramway_data[tramway_data['Area'] == 'The Icebergs']

Icebergs_Ratings = pd.DataFrame(Icebergs['Rating'].value_counts())

Icebergs_Ratings = pd.DataFrame(Icebergs_Ratings.reset_index())

Icebergs_Ratings = Icebergs_Ratings.rename(columns={"index": "Rating", "Rating": "Rating Count"})



Ratings = Icebergs_Ratings["Rating"]

Rating_count = Icebergs_Ratings["Rating Count"]



plt.style.use('classic')



plt.subplots(figsize=(6,8))

plt.barh(Ratings, Rating_count)

plt.ylabel("Ratings")

plt.xlabel("Rating Count")

plt.title("Boulder Problems in The Icebergs by Rating")

plt.yticks(Ratings)

plt.gca().invert_yaxis()



plt.show()
fig1, ax1 = plt.subplots(figsize=(20,20))



plt.style.use('classic')

plt.title("Boulder Problems in The Icebergs by Rating", y=1.05)

ax1.pie(Rating_count, labels=Ratings,

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
A51_DC = tramway_data[tramway_data['Area'] == 'Area 51/Deer Canyon']

A51_DC_Ratings = pd.DataFrame(A51_DC['Rating'].value_counts())

A51_DC_Ratings = pd.DataFrame(A51_DC_Ratings.reset_index())

A51_DC_Ratings = A51_DC_Ratings.rename(columns={"index": "Rating", "Rating": "Rating Count"})



Ratings = A51_DC_Ratings["Rating"]

Rating_count = A51_DC_Ratings["Rating Count"]



plt.style.use('classic')



plt.subplots(figsize=(6,8))

plt.barh(Ratings, Rating_count)

plt.ylabel("Ratings")

plt.xlabel("Rating Count")

plt.title("Boulder Problems in Area 51/Deer Canyon by Rating")

plt.yticks(Ratings)

plt.gca().invert_yaxis()



plt.show()
fig1, ax1 = plt.subplots(figsize=(20,20))



plt.style.use('classic')

plt.title("Boulder Problems in Area 51/Deer Canyon by Rating", y=1.05)

ax1.pie(Rating_count, labels=Ratings,

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
Cat_Gully = tramway_data[tramway_data['Area'] == 'Year of the Cat Gully']

Cat_Gully_Ratings = pd.DataFrame(Cat_Gully['Rating'].value_counts())

Cat_Gully_Ratings = pd.DataFrame(Cat_Gully_Ratings.reset_index())

Cat_Gully_Ratings = Cat_Gully_Ratings.rename(columns={"index": "Rating", "Rating": "Rating Count"})



Ratings = Cat_Gully_Ratings["Rating"]

Rating_count = Cat_Gully_Ratings["Rating Count"]



plt.style.use('classic')



plt.subplots(figsize=(6,8))

plt.barh(Ratings, Rating_count)

plt.ylabel("Ratings")

plt.xlabel("Rating Count")

plt.title("Boulder Problems in Year of the Cat Gully by Rating")

plt.yticks(Ratings)

plt.gca().invert_yaxis()



plt.show()
fig1, ax1 = plt.subplots(figsize=(20,20))



plt.style.use('classic')

plt.title("Boulder Problems in Year of the Cat Gully by Rating", y=1.05)

ax1.pie(Rating_count, labels=Ratings,

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
Edge_of_Nowhere = tramway_data[tramway_data['Area'] == 'The Edge of Nowhere']

Edge_of_Nowhere_Ratings = pd.DataFrame(Edge_of_Nowhere['Rating'].value_counts())

Edge_of_Nowhere_Ratings = pd.DataFrame(Edge_of_Nowhere_Ratings.reset_index())

Edge_of_Nowhere_Ratings = Edge_of_Nowhere_Ratings.rename(columns={"index": "Rating", "Rating": "Rating Count"})



Ratings = Edge_of_Nowhere_Ratings["Rating"]

Rating_count = Edge_of_Nowhere_Ratings["Rating Count"]



plt.style.use('classic')



plt.subplots(figsize=(6,8))

plt.barh(Ratings, Rating_count)

plt.ylabel("Ratings")

plt.xlabel("Rating Count")

plt.title("Boulder Problems in The Edge of Nowhere by Rating")

plt.yticks(Ratings)

plt.gca().invert_yaxis()



plt.show()
fig1, ax1 = plt.subplots(figsize=(20,20))



plt.style.use('classic')

plt.title("Boulder Problems in The Edge of Nowhere by Rating", y=1.05)

ax1.pie(Rating_count, labels=Ratings,

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()