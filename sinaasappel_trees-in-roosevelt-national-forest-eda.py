# load libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
# load data

train = pd.read_csv("../input/learn-together/train.csv", index_col = 'Id')

test = pd.read_csv("../input/learn-together/test.csv", index_col = 'Id')



# combine train and test data

train['train_test'] = 'train'

test['train_test']  = 'test'



alldata = pd.concat([train.drop(columns = ['Cover_Type']), test])
print("The train set has {0} rows and {1} columns.".format(str(train.shape[0]), str(train.shape[1])))

print("The test set has {0} rows and {1} columns.".format(str(test.shape[0]), str(test.shape[1])))
# set figure size

plt.figure(figsize=(14,6))



#add title

plt.title("Countplot of train versus test data")



# make countplot

sns.countplot(x = alldata.train_test)
print("The number of unique values per feature in the train set:")

print(train.nunique())
print("The number of unique values per feature in the test set:")

print(test.nunique())
# get number of missings per column in train and test set

print(train.isnull().sum().sum())

print(test.isnull().sum().sum())
print(train.duplicated(subset=None, keep='first').sum())

print(test.duplicated(subset=None, keep='first').sum())
# make new variable compass

def Compass(row):

    if row.Aspect < 45:

        return 'north'

    elif row.Aspect < 135:

        return 'east'

    elif row.Aspect < 225:

        return 'south'

    elif row.Aspect < 315:

        return 'west'

    else:

        return 'north'

    

train['Compass'] = train.apply(Compass, axis='columns')
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

        return 'Douglas-fir'

    else:

        return 'Krummholz'



train['CoverType'] = train.apply(CoverType, axis='columns')
# set figure size

plt.figure(figsize=(14,6))



#add title

plt.title("Countplot of Cover Types")



# make countplot

sns.countplot(x = train.CoverType)
myvars = list(train.columns[:10])

print("Describing the train set:")

train[myvars].describe().T
sns.distplot(a = alldata[alldata.train_test == 'test']['Elevation'], label = "test")

sns.distplot(a = alldata[alldata.train_test == 'train']['Elevation'], label = "train")



# Add title

plt.title("Histogram of Elevation, by train or test")



# Force legend to appear

plt.legend()
train['mean_Hillshade'] = (train.Hillshade_3pm + train.Hillshade_Noon + train.Hillshade_9am)/3



# make 7 new train sets, one for each cover type

spruce = train[train.Cover_Type == 1]

lodgepole = train[train.Cover_Type == 2]

ponderosa = train[train.Cover_Type == 3]

cottonwood = train[train.Cover_Type == 4]

aspen = train[train.Cover_Type == 5]

douglas = train[train.Cover_Type == 6]

krummholz = train[train.Cover_Type == 7]



# set figure size

plt.figure(figsize=(14,6))



# make the plots

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
# set figure size

plt.figure(figsize=(14,6))



sns.distplot(a = spruce['Aspect'], label = 'spruce')

sns.distplot(a = lodgepole['Aspect'], label = 'lodgepole')

sns.distplot(a = ponderosa['Aspect'], label = 'ponderasa')

sns.distplot(a = cottonwood['Aspect'], label = 'cottonwood')

sns.distplot(a = aspen['Aspect'], label = 'aspen')

sns.distplot(a = douglas['Aspect'], label = 'douglas')

sns.distplot(a = krummholz['Aspect'], label = 'krumholz')



# Add title

plt.title("Histogram of Aspect, by cover type")



# Force legend to appear

plt.legend()
# make new variable compass

def Compass(row):

    if row.Aspect < 22.5:

        return 'N'

    elif row.Aspect < 67.5:

        return 'NE'

    elif row.Aspect < 112.5:

        return 'E'

    elif row.Aspect < 157.5:

        return 'SE'

    elif row.Aspect < 202.5:

        return 'S'

    elif row.Aspect < 247.5:

        return 'SW'

    elif row.Aspect < 292.5:

        return 'W'

    elif row.Aspect < 337.5:

        return 'NW'

    else:

        return 'N'

    

train['Compass'] = train.apply(Compass, axis='columns')
df_plot = train.groupby(['CoverType', 'Compass']).size().reset_index().pivot(columns='CoverType', index='Compass', values=0)

df_plot.plot(kind='bar', stacked=True)



# Put the legend out of the figure

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# set figure size

plt.figure(figsize=(14,6))



sns.distplot(a = spruce['Slope'], label = 'spruce')

sns.distplot(a = lodgepole['Slope'], label = 'lodgepole')

sns.distplot(a = ponderosa['Slope'], label = 'ponderosa')

sns.distplot(a = cottonwood['Slope'], label = 'cottonwood')

sns.distplot(a = aspen['Slope'], label = 'aspen')

sns.distplot(a = douglas['Slope'], label = 'douglas')

sns.distplot(a = krummholz['Slope'], label = 'krummholz')



# Add title

plt.title("Histogram of Slope, by cover type")



# Force legend to appear

plt.legend()
# set figure size

plt.figure(figsize=(14,6))





sns.distplot(a = spruce['Horizontal_Distance_To_Hydrology'], label = 'spruce')

sns.distplot(a = lodgepole['Horizontal_Distance_To_Hydrology'], label = 'lodgepole')

sns.distplot(a = ponderosa['Horizontal_Distance_To_Hydrology'], label = 'ponderosa')

sns.distplot(a = cottonwood['Horizontal_Distance_To_Hydrology'], label = 'cottonwood')

sns.distplot(a = aspen['Horizontal_Distance_To_Hydrology'], label = 'aspen')

sns.distplot(a = douglas['Horizontal_Distance_To_Hydrology'], label = 'douglas')

sns.distplot(a = krummholz['Horizontal_Distance_To_Hydrology'], label = 'krummholz')



# Add title

plt.title("Histogram of Horizontal_Distance_To_Hydrology, by cover type")



# Force legend to appear

plt.legend()
# set figure size

plt.figure(figsize=(14,6))





sns.distplot(a = spruce['Vertical_Distance_To_Hydrology'])

sns.distplot(a = lodgepole['Vertical_Distance_To_Hydrology'])

sns.distplot(a = ponderosa['Vertical_Distance_To_Hydrology'])

sns.distplot(a = cottonwood['Vertical_Distance_To_Hydrology'])

sns.distplot(a = aspen['Vertical_Distance_To_Hydrology'])

sns.distplot(a = douglas['Vertical_Distance_To_Hydrology'])

sns.distplot(a = krummholz['Vertical_Distance_To_Hydrology'])



# Add title

plt.title("Histogram of Vertical_Distance_To_Hydrology, by cover type")



# Force legend to appear

plt.legend()
# set figure size

plt.figure(figsize=(14,6))



sns.boxplot(y="CoverType", x="Horizontal_Distance_To_Hydrology", data=train)
# set figure size

plt.figure(figsize=(14,6))



sns.boxplot(y="CoverType", x="Vertical_Distance_To_Hydrology", data=train)
sns.scatterplot(x = 'Vertical_Distance_To_Hydrology', 

                y = 'Horizontal_Distance_To_Hydrology', 

                hue = 'CoverType',

                data = train)
# set figure size

plt.figure(figsize=(14,6))





sns.distplot(a = spruce['Horizontal_Distance_To_Roadways'])

sns.distplot(a = lodgepole['Horizontal_Distance_To_Roadways'])

sns.distplot(a = ponderosa['Horizontal_Distance_To_Roadways'])

sns.distplot(a = cottonwood['Horizontal_Distance_To_Roadways'])

sns.distplot(a = aspen['Horizontal_Distance_To_Roadways'])

sns.distplot(a = douglas['Horizontal_Distance_To_Roadways'])

sns.distplot(a = krummholz['Horizontal_Distance_To_Roadways'])



# Add title

plt.title("Histogram of Horizontal_Distance_To_Roadways, by cover type")



# Force legend to appear

plt.legend()
# set figure size

plt.figure(figsize=(14,6))



sns.boxplot(y="CoverType", x="Horizontal_Distance_To_Roadways", data=train)
# set figure size

plt.figure(figsize=(14,6))





sns.distplot(a = spruce['Hillshade_9am'], label = "spruce")

sns.distplot(a = lodgepole['Hillshade_9am'], label = "lodgepole")

sns.distplot(a = ponderosa['Hillshade_9am'], label = "ponderosa")

sns.distplot(a = cottonwood['Hillshade_9am'], label = "cottonwood")

sns.distplot(a = aspen['Hillshade_9am'], label = "aspen")

sns.distplot(a = douglas['Hillshade_9am'], label = "douglas")

sns.distplot(a = krummholz['Hillshade_9am'], label = "krummholz")



# Add title

plt.title("Histogram of Hillshade_9am, by cover type")



# Force legend to appear

plt.legend()
# set figure size

plt.figure(figsize=(14,6))





sns.distplot(a = spruce['Hillshade_Noon'])

sns.distplot(a = lodgepole['Hillshade_Noon'])

sns.distplot(a = ponderosa['Hillshade_Noon'])

sns.distplot(a = cottonwood['Hillshade_Noon'])

sns.distplot(a = aspen['Hillshade_Noon'])

sns.distplot(a = douglas['Hillshade_Noon'])

sns.distplot(a = krummholz['Hillshade_Noon'])



# Add title

plt.title("Histogram of Hillshade_Noon, by cover type")



# Force legend to appear

plt.legend()
# set figure size

plt.figure(figsize=(14,6))





sns.distplot(a = spruce['Hillshade_3pm'])

sns.distplot(a = lodgepole['Hillshade_3pm'])

sns.distplot(a = ponderosa['Hillshade_3pm'])

sns.distplot(a = cottonwood['Hillshade_3pm'])

sns.distplot(a = aspen['Hillshade_3pm'])

sns.distplot(a = douglas['Hillshade_3pm'])

sns.distplot(a = krummholz['Hillshade_3pm'])



# Add title

plt.title("Histogram of Hillshade_3pm, by cover type")



# Force legend to appear

plt.legend()
train['mean_Hillshade'] = (train.Hillshade_3pm + train.Hillshade_Noon + train.Hillshade_9am)/3
# set figure size

plt.figure(figsize=(14,6))





sns.distplot(a = spruce['mean_Hillshade'])

sns.distplot(a = lodgepole['mean_Hillshade'])

sns.distplot(a = ponderosa['mean_Hillshade'])

sns.distplot(a = cottonwood['mean_Hillshade'])

sns.distplot(a = aspen['mean_Hillshade'])

sns.distplot(a = douglas['mean_Hillshade'])

sns.distplot(a = krummholz['mean_Hillshade'])



# Add title

plt.title("Histogram of mean_Hillshade, by cover type")



# Force legend to appear

plt.legend()
# set figure size

plt.figure(figsize=(10,10))



# make scatter plot of elevation and mean_Hillshade, color by CoverType

sns.scatterplot(y = train.Elevation, 

                x = train.mean_Hillshade,

                hue = train.CoverType)
# make scatter plot of elevation and mean_Hillshade, color by CoverType

sns.scatterplot(y = train.Elevation, 

                x = train.Hillshade_9am,

                hue = train.CoverType)
sns.scatterplot(y = train.Elevation, 

                x = train.Hillshade_Noon,

                hue = train.CoverType)
sns.scatterplot(y = train.Elevation, 

                x = train.Hillshade_3pm,

                hue = train.CoverType)
g = sns.FacetGrid(train, col="Compass", hue="CoverType", col_wrap=2)

g.map(plt.scatter, "Elevation", "Hillshade_9am", alpha=.5)

g.add_legend()
sns.distplot(a = spruce['Horizontal_Distance_To_Fire_Points'])

sns.distplot(a = lodgepole['Horizontal_Distance_To_Fire_Points'])

sns.distplot(a = ponderosa['Horizontal_Distance_To_Fire_Points'])

sns.distplot(a = cottonwood['Horizontal_Distance_To_Fire_Points'])

sns.distplot(a = aspen['Horizontal_Distance_To_Fire_Points'])

sns.distplot(a = douglas['Horizontal_Distance_To_Fire_Points'])

sns.distplot(a = krummholz['Horizontal_Distance_To_Fire_Points'])



# Add title

plt.title("Histogram of Horizontal_Distance_To_Fire_Points, by cover type")



# Force legend to appear

plt.legend()
# set figure size

plt.figure(figsize=(14,6))



sns.boxplot(y="CoverType", x="Horizontal_Distance_To_Fire_Points", data=train)
sns.scatterplot(x = 'Horizontal_Distance_To_Fire_Points',

               y = 'Horizontal_Distance_To_Roadways',

                hue = 'CoverType',

               data = train)
# make new variable Wilderness



def Wilderness(row):

    if row.Wilderness_Area1 == 1:

        return 'Rawah'

    elif row.Wilderness_Area2 == 1:

        return 'Neota'

    elif row.Wilderness_Area3 == 1:

        return 'Comanche Peak'

    elif row.Wilderness_Area4 == 1:

        return 'Cache la Poudre'

    else:

        return 0



train['Wilderness'] = train.apply(Wilderness, axis='columns')

test['Wilderness'] = test.apply(Wilderness, axis='columns')

alldata['Wilderness'] = alldata.apply(Wilderness, axis='columns')
plt.figure(figsize=(10,4))



plt.title("Countplot of Wilderness areas in train set")



sns.countplot(x = train.Wilderness)
plt.figure(figsize=(10,4))



plt.title("Countplot of Wilderness in test set")



sns.countplot(x = test.Wilderness)
# get the relevant data

r = [0,1,2,3]



df = train.groupby(['CoverType', 'Wilderness']).size().reset_index().pivot(columns='CoverType', index='Wilderness', values=0)

df = df.fillna(0)



totals = df.sum(axis = 1, skipna = True) 





aspen = [i / j * 100 for i,j in zip(df['Aspen'], totals)]

cottonwood = [i / j * 100 for i,j in zip(df['Cottonwood/Willow'], totals)]

douglas = [i / j * 100 for i,j in zip(df['Douglas-fir'], totals)]

krummholz = [i / j * 100 for i,j in zip(df['Krummholz'], totals)]

lodgepole = [i / j * 100 for i,j in zip(df['Lodgepole Pine'], totals)]

ponderosa = [i / j * 100 for i,j in zip(df['Ponderosa Pine'], totals)]

spruce = [i / j * 100 for i,j in zip(df['Spruce/Fir'], totals)]





# plot

barWidth = 0.85

names = ('Cache la Poudre','Comance Peak','Neota','Rawah')



# Create aspen bars

plt.bar(r, aspen, color='mediumblue', width=barWidth, label="Aspen")

# Create cottonwood bars

plt.bar(r, cottonwood, bottom=aspen, color='darkorange', width=barWidth, label="Cottonwood/Willow")

# Create douglas bars

plt.bar(r, douglas, bottom=[i+j for i,j in zip(aspen, 

                                               cottonwood)], color='forestgreen', width=barWidth, label = "Douglas-Fir")

# Create krummholz bars

plt.bar(r, krummholz, bottom=[i+j+k for i,j,k in zip(aspen, 

                                               cottonwood, 

                                               douglas)], color='red', width=barWidth, label = "Krummholz")



# Create lodgepole bars

plt.bar(r, lodgepole, bottom=[i+j+k+l for i,j,k,l in zip(aspen, 

                                               cottonwood, 

                                               douglas,

                                               krummholz)], color='darkviolet', width=barWidth, label = "Lodgepole Pine")

                            

# Create ponderosa bars

plt.bar(r, ponderosa, bottom=[i+j+k+l+m for i,j,k,l,m in zip(aspen, 

                                                cottonwood, 

                                               douglas,

                                               krummholz,

                                               lodgepole)], color='brown', width=barWidth, label = "Ponderosa Pine")



# Create spruce bars

plt.bar(r, spruce, bottom=[i+j+k+l+m+n for i,j,k,l,m,n in zip(aspen, 

                                               cottonwood, 

                                               douglas,

                                               krummholz,

                                               lodgepole,

                                               ponderosa)], color='violet', width=barWidth, label = "Spruce/Fir")



 

# Custom x axis and y axis

plt.xticks(r, names)

plt.xlabel("Wilderness area")

plt.ylabel("Percentage")



# give title to chart

plt.title("Forest types by Wilderness Area")





# Put the legend out of the figure

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)



# Show graphic

plt.show()
# this is really slooooooow, is there a different to programm this, to make it faster??



def SoilType(row):

    if row.Soil_Type1 == 1:

        return 1

    elif row.Soil_Type2 == 1 :

        return 2

    elif row.Soil_Type3 == 1 :

        return 3

    elif row.Soil_Type4 == 1:

        return 4

    elif row.Soil_Type5 == 1:

        return 5

    elif row.Soil_Type6 == 1:

        return 6

    elif row.Soil_Type7 == 1:

        return 7

    elif row.Soil_Type8 == 1:

        return 8

    elif row.Soil_Type9 == 1:

        return 9

    elif row.Soil_Type10 == 1:

        return 10

    elif row.Soil_Type11 == 1:

        return 11

    elif row.Soil_Type12 == 1:

        return 12

    elif row.Soil_Type13 == 1:

        return 13

    elif row.Soil_Type14 == 1:

        return 14

    elif row.Soil_Type15 == 1:

        return 15

    elif row.Soil_Type16 == 1 :

        return 16

    elif row.Soil_Type17 == 1 :

        return 17

    elif row.Soil_Type18 == 1:

        return 18

    elif row.Soil_Type19 == 1:

        return 19

    elif row.Soil_Type20 == 1:

        return 20

    elif row.Soil_Type21 == 1:

        return 21

    elif row.Soil_Type22 == 1:

        return 22

    elif row.Soil_Type23 == 1:

        return 23

    elif row.Soil_Type24 == 1:

        return 24

    elif row.Soil_Type25 == 1:

        return 25

    elif row.Soil_Type26 == 1:

        return 26

    elif row.Soil_Type27 == 1:

        return 27

    elif row.Soil_Type28 == 1:

        return 28

    elif row.Soil_Type29 == 1:

        return 29

    elif row.Soil_Type30 == 1:

        return 30

    elif row.Soil_Type31 == 1:

        return 31

    elif row.Soil_Type32 == 1:

        return 32

    elif row.Soil_Type33 == 1:

        return 33

    elif row.Soil_Type34 == 1:

        return 34

    elif row.Soil_Type35 == 1:

        return 35

    elif row.Soil_Type36 == 1:

        return 36

    elif row.Soil_Type37 == 1:

        return 37

    elif row.Soil_Type38 == 1:

        return 38

    elif row.Soil_Type39 == 1:

        return 39

    elif row.Soil_Type40 == 1:

        return 40

    else:

        return 0

    

train['SoilType'] = train.apply(SoilType, axis='columns')

test['SoilType'] = test.apply(SoilType, axis='columns')

#alldata['SoilType'] = alldata.apply(SoilType, axis='columns')
# set figure size

plt.figure(figsize=(14,6))



#add title

plt.title("Countplot of SoilType")



sns.countplot(x = train.SoilType)
# set figure size

plt.figure(figsize=(14,6))



#add title

plt.title("Countplot of SoilType")



sns.countplot(x = test.SoilType)


# get the relevant data

r = list(range(1,39))



df = train.groupby(['CoverType', 'SoilType']).size().reset_index().pivot(columns='CoverType', index='SoilType', values=0)

df = df.fillna(0)



totals = df.sum(axis = 1, skipna = True) 





aspen = [i / j * 100 for i,j in zip(df['Aspen'], totals)]

cottonwood = [i / j * 100 for i,j in zip(df['Cottonwood/Willow'], totals)]

douglas = [i / j * 100 for i,j in zip(df['Douglas-fir'], totals)]

krummholz = [i / j * 100 for i,j in zip(df['Krummholz'], totals)]

lodgepole = [i / j * 100 for i,j in zip(df['Lodgepole Pine'], totals)]

ponderosa = [i / j * 100 for i,j in zip(df['Ponderosa Pine'], totals)]

spruce = [i / j * 100 for i,j in zip(df['Spruce/Fir'], totals)]





# plot

barWidth = 0.3

names = list(range(1,39)) # is not correct!!!!



# set figure size

plt.figure(figsize=(12,6))



# Create aspen bars

plt.bar(r, aspen, color='mediumblue', width=barWidth, label="Aspen")



# Create cottonwood bars

plt.bar(r, cottonwood, bottom=aspen, color='darkorange', width=barWidth, label="Cottonwood/Willow")



# Create douglas bars

plt.bar(r, douglas, bottom=[i+j for i,j in zip(aspen, 

                                               cottonwood)], color='forestgreen', width=barWidth, label = "Douglas-Fir")

# Create krummholz bars

plt.bar(r, krummholz, bottom=[i+j+k for i,j,k in zip(aspen, 

                                               cottonwood, 

                                               douglas)], color='red', width=barWidth, label = "Krummholz")



# Create lodgepole bars

plt.bar(r, lodgepole, bottom=[i+j+k+l for i,j,k,l in zip(aspen, 

                                               cottonwood, 

                                               douglas,

                                               krummholz)], color='darkviolet', width=barWidth, label = "Lodgepole Pine")

                            

# Create ponderosa bars

plt.bar(r, ponderosa, bottom=[i+j+k+l+m for i,j,k,l,m in zip(aspen, 

                                                cottonwood, 

                                               douglas,

                                               krummholz,

                                               lodgepole)], color='brown', width=barWidth, label = "Ponderosa Pine")



# Create spruce bars

plt.bar(r, spruce, bottom=[i+j+k+l+m+n for i,j,k,l,m,n in zip(aspen, 

                                               cottonwood, 

                                               douglas,

                                               krummholz,

                                               lodgepole,

                                               ponderosa)], color='violet', width=barWidth, label = "Spruce/Fir")



 

# Custom x axis and y axis

plt.xticks(r, names)

plt.xlabel("Soil type")

plt.ylabel("Percentage")



# give title to chart

plt.title("Forest types by soil type")





# Put the legend out of the figure

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)



# Show graphic

plt.show()
