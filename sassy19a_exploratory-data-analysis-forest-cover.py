import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
df = pd.read_csv('../input/train.csv')
df.columns
df2 = df.copy()
# add target names for readability
def forest(x):
    if x==1:
        return 'Spruce/Fir'
    elif x==2:
        return 'Lodgepole Pine'
    elif x==3:
        return 'Ponderosa Pine'
    elif x==4:
        return 'Cottonwood/Willow'
    elif x==5:
        return 'Aspen'
    elif x==6:
        return 'Douglas-fir'
    elif x==7:
        return 'Krummholz'
    
df2['Cover_Type'] = df2['Cover_Type'].apply(lambda x: forest(x))
df2[df.columns[1:11]].hist(figsize=(20,15),bins=50)
plt.tight_layout()
df2['Cover_Type'].value_counts()
cmap = sns.color_palette("Set2")

sns.countplot(x='Cover_Type', data=df2, palette=cmap);
plt.xticks(rotation=45);
# convert dummies into a single column
# convert wilderness
wild_dummies = df[['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4']]
wild = wild_dummies.idxmax(axis=1)
wild.name = 'Wilderness'

# convert soil
soil_dummies = df[df.columns[15:55]]
soil = soil_dummies.idxmax(axis=1)
soil.name = 'Soil'

# create new dataframe with only cover type, wilderness, soil
wild = pd.concat([df2['Cover_Type'],wild,soil], axis=1)
wild.head()
# name wilderness type for readability
def wild_convert(x):
    if x == 'Wilderness_Area1':
        return 'Rawah'
    elif x=='Wilderness_Area2':
        return 'Neota'
    elif x=='Wilderness_Area3':
        return 'Comanche Peak'
    elif x=='Wilderness_Area4':
        return 'Cache la Poudre'
    
wild['Wilderness'] = wild['Wilderness'].apply(lambda x: wild_convert(x))
sns.countplot(x='Wilderness',data=wild, palette=cmap);
plt.figure(figsize=(12,6))
sns.countplot(x='Cover_Type',data=wild, hue='Wilderness');
plt.xticks(rotation=45);
plt.figure(figsize=(12,5))
sns.countplot(x='Soil',data=wild);
plt.xticks(rotation=90);
# take a look at soil type summary statistics
df[df.columns[15:55]].describe().T.sort_values(by='mean')
# select features which are continuous
boxplots = df2[['Elevation', 'Aspect', 'Slope',
               'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
               'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
               'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points','Cover_Type']]
cmap = sns.color_palette("Set2")

fig, axes = plt.subplots(ncols=2, nrows=5, figsize=(10, 18))
a = [i for i in axes for i in i]
for i, ax in enumerate(a):
    sns.boxplot(x='Cover_Type', y=boxplots.columns[i], data=boxplots, palette=cmap, width=0.5, ax=ax);

# rotate x-axis for every single plot
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=45)

# set spacing for every subplot, else x-axis will be covered
plt.tight_layout()
# calculate pearson's correlation, exclude ID
corr = df[df.columns[1:]].corr()

corr['Cover_Type'].sort_values(ascending=False)
plt.figure(figsize=(15, 8))
sns.heatmap(corr, cmap=sns.color_palette("RdBu_r", 20));