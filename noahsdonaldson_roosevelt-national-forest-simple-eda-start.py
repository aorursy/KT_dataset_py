import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/learn-together/train.csv")

test = pd.read_csv("/kaggle/input/learn-together/test.csv")
train.head()
test.head()
print(f"Missing values in train: {train.isnull().values.any()}")

print(f"Missing values in test: {test.isnull().values.any()}")
print(train.isnull().sum())

print(test.isnull().sum())
print(train.duplicated(subset=None, keep='first').sum())

print(test.duplicated(subset=None, keep='first').sum())
train.info()
test.info()
print("The number of unique values per feature in the train set:")

print(train.nunique())
print("The number of unique values per feature in the test set:")

print(test.nunique())
#new variable CoverType

def CoverType(row):

    if row.Cover_Type == 1:

        return 'Spruce/Fir'

    elif row.Cover_Type == 2:

        return 'Lodgepole Pine'

    elif row.Cover_Type == 3:

        return 'Ponderosa Pine'

    elif row.Cover_Type == 4:

        return 'Cottonwood/Willow'

    elif row.Cover_Type == 5:

        return 'Aspen'

    elif row.Cover_Type == 6:

        return 'Duglas-fir'

    else: 

        return 'Krummholz'

train['CoverTypeName'] = train.apply(CoverType, axis='columns')    
train.head()
# split into the 7 types

spruce = train[train.Cover_Type == 1]

lodgepole = train[train.Cover_Type == 2]

ponderosa = train[train.Cover_Type == 3]

cottonwood = train[train.Cover_Type == 4]

aspen = train[train.Cover_Type == 5]

douglas = train[train.Cover_Type == 6]

krummholz = train[train.Cover_Type == 7]
spruce.head()
# Get counts

train.CoverTypeName.value_counts()
# look a the first 10 features which are continous features

cont_feat = list(train.columns[:10])  #make a list of the columns with continous features

train[cont_feat].describe()
sns.distplot(a = spruce['Elevation'], label = "Spruce")

sns.distplot(a = lodgepole['Elevation'], label = "Lodgepole")

sns.distplot(a = ponderosa['Elevation'], label = "Ponderosa")

sns.distplot(a = cottonwood['Elevation'], label = "Cottonwood")

sns.distplot(a = aspen['Elevation'], label = "Aspen")

sns.distplot(a = douglas['Elevation'], label = "Douglas")

sns.distplot(a = krummholz['Elevation'], label = "Krummholz")



# Add title

plt.title("Histogram of Elevation, by cover type")



# Force legend to appear

plt.legend()
# scatter plots to show correlation to the Cover_Type?

num_columns = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am"

               , "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]



for i in num_columns:

    plt.figure()

    train.plot(kind="scatter", x="Cover_Type", y=i)

    plt.ylabel(i)

    plt.xlabel("Cover_Type")
i = "Wilderness_Area1"

sns.countplot(x=i, hue="CoverTypeName" ,data=train)

# Add title

plt.title("Counts of Wilderness area 1 by cover type")
i = "Wilderness_Area2"

sns.countplot(x=i, hue="CoverTypeName" ,data=train)

# Add title

plt.title("Counts of Wilderness area 2 by cover type")
i = "Wilderness_Area3"

sns.countplot(x=i, hue="CoverTypeName" ,data=train)

# Add title

plt.title("Counts of Wilderness area 3 by cover type")
i = "Wilderness_Area4"

sns.countplot(x=i, hue="CoverTypeName" ,data=train)

# Add title

plt.title("Counts of Wilderness area 4 by cover type")
soil = "Soil_Type1" #Soil_Type1 - 40

area = "Wilderness_Area1" #Wilderness_Area1 = 4

sns.catplot(x=area, hue = "CoverTypeName", col=soil, data=train, kind="count")
soil = "Soil_Type40" #Soil_Type1 - 40

area = "Wilderness_Area1" #Wilderness_Area1 = 4

sns.catplot(x=area, hue = "CoverTypeName", col=soil, data=train, kind="count")
spruce.groupby(['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4'])['CoverTypeName'].count()