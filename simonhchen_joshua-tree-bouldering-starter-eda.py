# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read the data and take a quick look at the data.
jtree_data = pd.read_csv("/kaggle/input/joshua-tree-bouldering-problems/formatted_J-Tree_data.csv")
jtree_data = jtree_data.drop(['Unnamed: 0'], axis=1)
jtree_data.head()
jtree_areas = pd.DataFrame(jtree_data['Bouldering Area'].value_counts())
jtree_areas = pd.DataFrame(jtree_areas.reset_index())
jtree_areas = jtree_areas.rename(columns={"index": "Bouldering Area", "Bouldering Area": "Bouldering Area Problem Count"})
jtree_areas
Area = jtree_areas["Bouldering Area"]
Area_problem_count = jtree_areas["Bouldering Area Problem Count"]

plt.style.use('classic')

plt.subplots(figsize=(6,8))
plt.barh(Area, Area_problem_count)
plt.ylabel("Area")
plt.xlabel("Area Problem Count")
plt.title("Joshua Tree Areas by Number of Problems")
plt.yticks(Area)
plt.gca().invert_yaxis()

plt.show()
fig1, ax1 = plt.subplots(figsize=(15,15))

plt.style.use('classic')
plt.title("Joshua Tree Bouldering Areas by Number of Problems", y=1.05)
ax1.pie(Area_problem_count, labels=Area, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
Quail_Springs = jtree_data[jtree_data['Bouldering Area'] == 'Quail Springs Bouldering']
Quail_Springs_Ratings = pd.DataFrame(Quail_Springs['Rating'].value_counts())
Quail_Springs_Ratings = pd.DataFrame(Quail_Springs_Ratings.reset_index())
Quail_Springs_Ratings = Quail_Springs_Ratings.rename(columns={"index": "Rating", "Rating": "Rating Count"})

Ratings = Quail_Springs_Ratings["Rating"]
Rating_count = Quail_Springs_Ratings["Rating Count"]

plt.style.use('classic')

plt.subplots(figsize=(6,8))
plt.barh(Ratings, Rating_count)
plt.ylabel("Ratings")
plt.xlabel("Rating Count")
plt.title("Boulder Problems in Quail Springs by Rating")
plt.yticks(Ratings)
plt.gca().invert_yaxis()

plt.show()
fig1, ax1 = plt.subplots(figsize=(15,15))

plt.style.use('classic')
plt.title("Boulder Problems in Quail Springs by Rating", y=1.05)
ax1.pie(Rating_count, labels=Ratings, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
print(Quail_Springs['Bouldering Sub-Area'].value_counts())
print(Quail_Springs.shape)
Quail_Springs