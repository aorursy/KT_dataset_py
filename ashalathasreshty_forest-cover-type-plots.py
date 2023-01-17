# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import rc
dtrain = pd.read_csv("../input/forest-cover-type-prediction/train.csv")
dtrain.head(10)
dtrain.columns
df = dtrain[['Id','Cover_Type','Elevation','Aspect','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']]
df
# Violin Plots



fig, ax = plt.subplots(figsize = (12, 12))



a1 = plt.subplot2grid((2,2),(0,0))

sns.violinplot( x=df["Cover_Type"], y=df["Horizontal_Distance_To_Hydrology"], width=1, linewidth = 1)

a1.set_title('Horizontal Distance To Hydrology')

a1.set_xlabel('Forest Cover Type')

a1.set_ylabel('Horizontal Distance')

a2 = plt.subplot2grid((2,2),(0,1))

sns.violinplot( x=df["Cover_Type"], y=df["Vertical_Distance_To_Hydrology"], width=1, linewidth = 1)

a2.set_title('Vertical Distance To Hydrology')

a2.set_xlabel('Forest Cover Type')

a2.set_ylabel('Vertical Distance')

a3 = plt.subplot2grid((2,2),(1,0))

sns.violinplot( x=df["Cover_Type"], y=df["Horizontal_Distance_To_Roadways"], width=1, linewidth = 1)

a3.set_title('Horizontal Distance To Roadways')

a3.set_xlabel('Forest Cover Type')

a3.set_ylabel('Horizontal Distance')

a4 = plt.subplot2grid((2,2),(1,1))

sns.violinplot( x=df["Cover_Type"], y=df["Horizontal_Distance_To_Fire_Points"], width=1, linewidth = 1)

a4.set_title('Horizontal_Distance_To_Fire_Points')

a4.set_xlabel('Forest Cover Type')

a4.set_ylabel('Horizontal Distance')
# Density Plots



fig, ax = plt.subplots(figsize = (12, 6))

a1 = plt.subplot2grid((1,2),(0,0))

p1 = sns.kdeplot(df['Horizontal_Distance_To_Hydrology'], shade=True, color="r")

p1.set_title('Distance To Hydrology')

p1.set_xlabel('Forest Cover Type')

p1.set_ylabel('Distance to Hydrology')

p1 = sns.kdeplot(df['Vertical_Distance_To_Hydrology'], shade=True, color="b")

a2 = plt.subplot2grid((1,2),(0,1))

p2=sns.kdeplot(df['Horizontal_Distance_To_Roadways'], shade=True, color="g")

p2=sns.kdeplot(df['Horizontal_Distance_To_Fire_Points'], shade=True, color="c")

p1.set_title('Distance to Roadways and Fire Points')

p1.set_xlabel('Forest Cover Type')

p1.set_ylabel('Distance')

# Box Plots

df2 = dtrain[['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology']]

sns.boxplot(data = df2, orient = "h")

plt.show()

df3 = dtrain[['Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']]

sns.boxplot(data = df3, orient = "h")

plt.show()



fig, ax = plt.subplots(figsize = (16, 20))



a1 = plt.subplot2grid((3,2),(0,0))

sns.boxplot( x=df["Cover_Type"], y=df["Elevation"], palette="Blues")

a1.set_title('Elevation')

a1.set_xlabel('Forest Cover Type')

a1.set_ylabel('Horizontal Distance')

a2 = plt.subplot2grid((3,2),(0,1))

sns.boxplot( x=df["Cover_Type"], y=df["Horizontal_Distance_To_Hydrology"],  palette="Greens" )

a2.set_title('Horizontal Distance To Hydrology')

a2.set_xlabel('Forest Cover Type')

a2.set_ylabel('Horizontal Distance')

a3 = plt.subplot2grid((3,2),(1,0))

sns.boxplot( x=df["Cover_Type"], y=df["Vertical_Distance_To_Hydrology"], palette="Reds")

a3.set_title('Vertical Distance To Hydrology')

a3.set_xlabel('Forest Cover Type')

a3.set_ylabel('Vertical Distance')

a4 = plt.subplot2grid((3,2),(1,1))

sns.boxplot( x=df["Cover_Type"], y=df["Horizontal_Distance_To_Roadways"], palette="Purples")

a4.set_title('Horizontal Distance To Roadways')

a4.set_xlabel('Forest Cover Type')

a4.set_ylabel('Horizontal Distance')

a5 = plt.subplot2grid((3,2),(2,0))

sns.boxplot( x=df["Cover_Type"], y=df["Horizontal_Distance_To_Fire_Points"], palette = "Oranges")

a5.set_title('Horizontal_Distance_To_Fire_Points')

a5.set_xlabel('Forest Cover Type')

a5.set_ylabel('Horizontal Distance')



# Hist plot

f, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True)

sns.distplot( df["Horizontal_Distance_To_Fire_Points"] , color="skyblue", ax=axes[0, 0])

sns.distplot( df["Horizontal_Distance_To_Hydrology"] , color="olive", ax=axes[0, 1])

sns.distplot( df["Vertical_Distance_To_Hydrology"] , color="gold", ax=axes[1, 0])

sns.distplot( df["Horizontal_Distance_To_Roadways"] , color="teal", ax=axes[1, 1])

df1 = dtrain[['Cover_Type', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',]]
#Creating barplot for Degree Variables

fig, a = plt.subplots(2, 2)

a[0][0].bar(df1['Cover_Type'], df1['Wilderness_Area1'], color = 'b')

a[0][1].bar(df1['Cover_Type'], df1['Wilderness_Area2'], color = 'r')

a[1][0].bar(df1['Cover_Type'], df1['Wilderness_Area3'], color = 'g')

a[1][1].bar(df1['Cover_Type'], df1['Wilderness_Area4'], color = 'y')
df21 = dtrain[['Cover_Type', 'Wilderness_Area1']]

df22 = dtrain[['Cover_Type', 'Wilderness_Area2']]

df23 = dtrain[['Cover_Type', 'Wilderness_Area3']]

df24 = dtrain[['Cover_Type', 'Wilderness_Area4']]
aa = df21.groupby('Wilderness_Area1').count()
aa['Wild_Area1'] = df21.groupby('Wilderness_Area1').count()

aa['Wild_Area2'] = df22.groupby('Wilderness_Area2').count()

aa['Wild_Area3'] = df23.groupby('Wilderness_Area3').count()

aa['Wild_Area4'] = df24.groupby('Wilderness_Area4').count()

aa
del aa['Cover_Type']
aa
# Stacked barplot



rc('font', weight='bold')



bars1 = aa.loc[0]

bars2 = aa.loc[1]



bars = np.add(bars1, bars2).tolist()

r = [0,1,2,3]



names = ['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4']

barWidth = 1



ax1 = plt.bar(r, bars1, color='#7f6d5f', edgecolor='white', width=barWidth)

ax2 = plt.bar(r, bars2, bottom=bars1, color='turquoise', edgecolor='white', width=barWidth)



plt.xticks(r, names, fontweight='bold')

plt.xticks(rotation=45, ha='right')



for r1, r2 in zip(ax1, ax2):

    h1 = r1.get_height()

    h2 = r2.get_height()

    plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., "%d" % h1, ha="center", va="center", color="white", fontsize=16, fontweight="bold")

    plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., "%d" % h2, ha="center", va="center", color="white", fontsize=16, fontweight="bold")



plt.legend()   

plt.ylabel("Count", fontsize=18)

plt.xlabel("Forest Wilderness Area", fontsize=18)



plt.show()



f, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True)



sns.scatterplot(dtrain['Hillshade_9am'],dtrain['Aspect'],hue=dtrain['Cover_Type'],palette='rainbow', ax=axes[0, 0])

sns.scatterplot(dtrain['Hillshade_Noon'],dtrain['Aspect'],hue=dtrain['Cover_Type'],palette='rainbow', ax=axes[0, 1])

sns.scatterplot(dtrain['Hillshade_3pm'],dtrain['Aspect'],hue=dtrain['Cover_Type'],palette='rainbow', ax=axes[0, 2])

sns.scatterplot(dtrain['Hillshade_9am'],dtrain['Elevation'],hue=dtrain['Cover_Type'],palette='rainbow', ax=axes[1, 0])

sns.scatterplot(dtrain['Hillshade_Noon'],dtrain['Elevation'],hue=dtrain['Cover_Type'],palette='rainbow', ax=axes[1, 1])

sns.scatterplot(dtrain['Hillshade_3pm'],dtrain['Elevation'],hue=dtrain['Cover_Type'],palette='rainbow', ax=axes[1, 2])

df4 = dtrain.iloc[:, 15:]
df4
df4['Elevation'] = dtrain[['Elevation']]
df4
df4_names = []

for i in range(len(df4.columns)):

#    print(df4.columns[i])

    df4_names.append(df4.columns[i])

print(df4_names)

    
sns.catplot(x="Cover_Type", y="Elevation", hue="Soil_Type1", kind="swarm", data=df4);
sns.catplot(x="Cover_Type", y="Elevation", hue="Soil_Type2", kind="violin", split=True, data=df4);