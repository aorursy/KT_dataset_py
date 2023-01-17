import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sn



%matplotlib inline





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/train.csv")
df.head()
df.shape
df.columns
df.info()
df['Wilderness_Area'] = pd.get_dummies(df[df.columns[11:15]]).idxmax(axis=1)

df['Soil_Type'] = pd.get_dummies(df[df.columns[15:55]]).idxmax(axis=1)
data = df.drop(df[df.columns[11:55]], axis=1)

data = data.drop(columns='Id')
data.shape
def Cover_name(x):

    if x == 1:

        return 'Spruce/Fir'

    

    if x == 2:

        return 'Lodgepole Pine'

    

    if x == 3:

        return 'Ponderosa Pine'

    

    if x == 4:

        return 'Cottonwood/Willow'

    

    if x == 5:

        return 'Aspen'

    

    if x == 6:

        return 'Douglas-fir'

    

    if x == 7:

        return 'Krummholz'

    



data['Cover_Type'] = data['Cover_Type'].apply(Cover_name)

data['Cover_Type'].value_counts()
data['Wilderness_Area'] = data['Wilderness_Area'].replace({'Wilderness_Area1': 'Rawah',

                                                       'Wilderness_Area2': 'Neota',

                                                       'Wilderness_Area3': 'Comanche Peak',

                                                       'Wilderness_Area4': 'Cache la Poudre'})



data['Wilderness_Area'].value_counts()
data.head()
data.shape
data.info()
data.nunique()
data.describe()
sn.distplot(data['Elevation'], color='Blue')
sn.distplot(data['Aspect'], color='Green')
sn.distplot(data['Slope'], color='Red')
plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

plt.hist(data['Horizontal_Distance_To_Roadways'], color='cyan')

plt.hist(data['Horizontal_Distance_To_Fire_Points'], color='orange',alpha=0.7)

plt.legend('upper right' , labels = ['Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points'])

plt.xlabel("Distance")



plt.subplot(1,2,2)

plt.hist(data['Horizontal_Distance_To_Hydrology'], color='green')

plt.hist(data['Vertical_Distance_To_Hydrology'], color='maroon', alpha=0.7)

plt.legend('upper right' , labels = ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology'])

plt.xlabel("Distance")

plt.hist(data['Hillshade_9am'])

plt.hist(data['Hillshade_Noon'], alpha=0.7)

plt.hist(data['Hillshade_3pm'], alpha = 0.7)

plt.legend('upper left' , labels = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'])

plt.xlabel("Index value")
data['Soil_Type'].value_counts().plot(kind='barh', figsize = (10, 12), color='maroon')
plt.figure(figsize=(8, 5))

sn.countplot(x="Wilderness_Area", data=data)

plt.figure(figsize=(10, 5))

sn.countplot(x="Cover_Type", data=data, palette=sn.color_palette("cubehelix", 7))
plt.figure(figsize=(13,15))

plt.subplots_adjust(hspace=0.5)





plt.subplot(3,2,1)

a1 = sn.barplot(x='Cover_Type', y='Elevation', data=data)

a1.set_xticklabels(a1.get_xticklabels(), rotation=15)



plt.subplot(3,2,3)

a2 = sn.barplot(x='Cover_Type', y='Aspect', data=data)

a2.set_xticklabels(a2.get_xticklabels(), rotation=15);





plt.subplot(3,2,5)

a3 = sn.barplot(x='Cover_Type', y='Slope', data=data)

a3.set_xticklabels(a3.get_xticklabels(), rotation=15);





plt.subplot(3,2,2)

a4 = sn.boxplot(x='Cover_Type', y='Elevation', data=data)

a4.set_xticklabels(a4.get_xticklabels(), rotation=15);



plt.subplot(3,2,4)

a5 = sn.boxplot(x='Cover_Type', y='Aspect', data=data)

a5.set_xticklabels(a5.get_xticklabels(), rotation=15);



plt.subplot(3,2,6)

a6 = sn.boxplot(x='Cover_Type', y='Slope', data=data)

a6.set_xticklabels(a6.get_xticklabels(), rotation=15);
plt.figure(figsize=(15,10))

plt.subplots_adjust(wspace=0.2,hspace=0.5)





plt.subplot(2,2,1)

b1 = sn.violinplot(x='Cover_Type', y='Horizontal_Distance_To_Hydrology', data=data)

b1.set_xticklabels(b1.get_xticklabels(), rotation=15);





plt.subplot(2,2,2)

b2 = sn.violinplot(x='Cover_Type', y='Vertical_Distance_To_Hydrology', data=data)

b2.set_xticklabels(b2.get_xticklabels(), rotation=15);





plt.subplot(2,2,3)

b3 = sn.violinplot(x='Cover_Type', y='Horizontal_Distance_To_Roadways', data=data)

b3.set_xticklabels(b3.get_xticklabels(), rotation=15);





plt.subplot(2,2,4)

b4 = sn.violinplot(x='Cover_Type', y='Horizontal_Distance_To_Fire_Points', data=data)

b4.set_xticklabels(b4.get_xticklabels(), rotation=15);



plt.figure(figsize=(12, 7))

sn.countplot(x="Wilderness_Area", hue="Cover_Type", data=data)
plt.figure(figsize=(10,15))



plt.subplot(3,1,1)

sn.swarmplot(x='Cover_Type', y='Hillshade_9am', data=data)



plt.subplot(3,1,2)

sn.swarmplot(x='Cover_Type', y='Hillshade_Noon', data=data)



plt.subplot(3,1,3)

sn.swarmplot(x='Cover_Type', y='Hillshade_3pm', data=data)
plt.figure(figsize=(10,7))

sn.heatmap(data.corr(), annot=True, cmap="coolwarm", square=True );
plt.figure(figsize=(13,10))



plt.subplot(3,2,1)

sn.scatterplot(x='Aspect', y='Hillshade_9am', hue='Cover_Type', data=data, legend=False);



plt.subplot(3,2,2)

sn.scatterplot(x='Aspect', y='Hillshade_3pm', hue='Cover_Type', data=data, legend=False);



plt.subplot(3,2,3)

sn.scatterplot(x='Hillshade_3pm', y='Hillshade_9am', hue='Cover_Type', data=data, legend = 'brief');



plt.subplot(3,2,4)

sn.scatterplot(x='Hillshade_3pm', y='Hillshade_Noon', hue='Cover_Type', data=data,  legend=False);



plt.subplot(3,2,5)

sn.scatterplot(x='Slope', y='Hillshade_Noon', hue='Cover_Type', data=data, legend=False);



plt.subplot(3,2,6)

sn.scatterplot(x='Horizontal_Distance_To_Hydrology', y='Vertical_Distance_To_Hydrology', 

               hue='Cover_Type', data=data, legend=False);



sn.lmplot(x='Aspect', y='Hillshade_9am', hue='Cover_Type', data=data, scatter=False);



sn.lmplot(x='Hillshade_3pm', y='Hillshade_9am', hue='Cover_Type', data=data, scatter=False);



sn.lmplot(x='Slope', y='Hillshade_Noon', hue='Cover_Type', data=data, scatter=False);



sn.lmplot(x='Aspect', y='Hillshade_3pm', hue='Cover_Type', data=data, scatter=False);



sn.lmplot(x='Horizontal_Distance_To_Hydrology', y='Vertical_Distance_To_Hydrology', 

               hue='Cover_Type', data=data, scatter=False)



sn.lmplot(x='Hillshade_3pm', y='Hillshade_Noon', hue='Cover_Type', data=data,  scatter=False);
