# import the needed libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/covtype.csv')
df.describe()
df.columns
wild1 = df['Wilderness_Area1'].groupby(df['Cover_Type'])

totals = []

for value in wild1.sum():

    totals.append(value)

print(totals)

total_sum = sum(totals)

print("Total Trees in Area: %d" % total_sum)

percentages = [ (total*100 / total_sum) for total in totals]

print(percentages)
trees = ['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow',

         'Aspen','Douglas-fir', 'Krummholz']

xs = [i + 0.1 for i, _ in enumerate(trees)]

plt.bar(xs, totals)

plt.ylabel("# of Trees")

plt.title("Cover Types in Wilderness Area 1")

plt.xticks([i + 0.5 for i, _ in enumerate(trees)], trees, rotation='vertical');
wild2 = df['Wilderness_Area2'].groupby(df['Cover_Type'])

totals = []

for value in wild2.sum():

    totals.append(value)

print(totals)

total_sum = sum(totals)

print("Total Trees in Area: %d" % total_sum)

percentages = [ (total*100 / total_sum) for total in totals]

print(percentages)
trees = ['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow',

         'Aspen','Douglas-fir', 'Krummholz']

xs = [i + 0.1 for i, _ in enumerate(trees)]

plt.bar(xs, totals)

plt.ylabel("# of Trees")

plt.title("Cover Types in Wilderness Area 2")

plt.xticks([i + 0.5 for i, _ in enumerate(trees)], trees, rotation='vertical');
wild3 = df['Wilderness_Area3'].groupby(df['Cover_Type'])

totals = []

for value in wild3.sum():

    totals.append(value)

print(totals)

total_sum = sum(totals)

print("Total Trees in Area: %d" % total_sum)

percentages = [ (total*100 / total_sum) for total in totals]

print(percentages)
trees = ['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow',

         'Aspen','Douglas-fir', 'Krummholz']

xs = [i + 0.1 for i, _ in enumerate(trees)]

plt.bar(xs, totals)

plt.ylabel("# of Trees")

plt.title("Cover Types in Wilderness Area 3")

plt.xticks([i + 0.5 for i, _ in enumerate(trees)], trees, rotation='vertical');
wild4 = df['Wilderness_Area4'].groupby(df['Cover_Type'])

totals = []

for value in wild4.sum():

    totals.append(value)

print(totals)

total_sum = sum(totals)

print("Total Trees in Area: %d" % total_sum)

percentages = [ (total*100 / total_sum) for total in totals]

print(percentages)
trees = ['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow',

         'Aspen','Douglas-fir', 'Krummholz']

xs = [i + 0.1 for i, _ in enumerate(trees)]

plt.bar(xs, totals)

plt.ylabel("# of Trees")

plt.title("Cover Types in Wilderness Area 4")

plt.xticks([i + 0.5 for i, _ in enumerate(trees)], trees, rotation='vertical');
# Group elevation data by cover type

coverSets = df['Wilderness_Area1'].groupby(df['Cover_Type'])
totals = []

counter = 0

for total in coverSets.count():

    #counter = counter + 1

    totals.append(total)

totals_zip = zip(totals, [1, 2, 3, 4, 5, 6, 7])

print(list(totals_zip))
total_trees = sum(totals)

percentages = [(total*100 / total_trees) for total in totals]

print(percentages)
trees = ['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow',

         'Aspen','Douglas-fir', 'Krummholz']

xs = [i + 0.1 for i, _ in enumerate(trees)]

plt.bar(xs, totals)

plt.ylabel("# of Trees")

plt.title("Tree types in Roosevelt National Forest, Colorado")

plt.xticks([i + 0.5 for i, _ in enumerate(trees)], trees, rotation='vertical');
#print(elevations.describe())

spruce = df[df.Cover_Type == 1]

lodgepole = df[df.Cover_Type == 2]

ponderosa = df[df.Cover_Type == 3]

willow = df[df.Cover_Type == 4]

aspen = df[df.Cover_Type == 5]

douglas = df[df.Cover_Type == 6]

krummholz = df[df.Cover_Type == 7]

plt.figure()

plt.title('Elevation of Cover Types')

plt.ylabel('Elevation (in meters)')

data = [spruce.Elevation, lodgepole.Elevation, ponderosa.Elevation, willow.Elevation,

aspen.Elevation, douglas.Elevation, krummholz.Elevation]

plt.xticks([1, 2, 3, 4, 5, 6, 7])

plt.boxplot(data)

plt.show() 
plt.figure()

plt.title('Horizontal_Distance_To_Hydrology of Cover Types')

plt.ylabel('Distance (in meters)')

data = [spruce.Horizontal_Distance_To_Hydrology, lodgepole.Horizontal_Distance_To_Hydrology,

        ponderosa.Horizontal_Distance_To_Hydrology, willow.Horizontal_Distance_To_Hydrology,

        aspen.Horizontal_Distance_To_Hydrology, douglas.Horizontal_Distance_To_Hydrology, 

        krummholz.Horizontal_Distance_To_Hydrology]

plt.xticks([1, 2, 3, 4, 5, 6, 7])

plt.boxplot(data)

plt.show() 
plt.figure()

plt.title('Vertical_Distance_To_Hydrology of Cover Types')

plt.ylabel('Distance (in meters)')

data = [spruce.Vertical_Distance_To_Hydrology, lodgepole.Vertical_Distance_To_Hydrology,

        ponderosa.Vertical_Distance_To_Hydrology, willow.Vertical_Distance_To_Hydrology,

        aspen.Vertical_Distance_To_Hydrology, douglas.Vertical_Distance_To_Hydrology, 

        krummholz.Vertical_Distance_To_Hydrology]

plt.xticks([1, 2, 3, 4, 5, 6, 7])

plt.boxplot(data)

plt.show() 
plt.figure()

plt.title('Slope of Cover Types')

plt.ylabel('Angle (in degrees)')

data = [spruce.Slope, lodgepole.Slope,

        ponderosa.Slope, willow.Slope,

        aspen.Slope, douglas.Slope, 

        krummholz.Slope]

plt.xticks([1, 2, 3, 4, 5, 6, 7])

plt.boxplot(data)

plt.show() 
plt.figure()

plt.title('Horizontal Distance to Roadways')

plt.ylabel('Distance (in meters)')

roadway_data = [spruce.Horizontal_Distance_To_Roadways, lodgepole.Horizontal_Distance_To_Roadways,

        ponderosa.Horizontal_Distance_To_Roadways, willow.Horizontal_Distance_To_Roadways,

        aspen.Horizontal_Distance_To_Roadways, douglas.Horizontal_Distance_To_Roadways, 

        krummholz.Horizontal_Distance_To_Roadways]

plt.xticks([1, 2, 3, 4, 5, 6, 7])

plt.boxplot(roadway_data)

plt.show() 
plt.figure()

plt.title('Horizontal Distance to Fire Points')

plt.ylabel('Distance (in meters)')

firepoint_data = [spruce.Horizontal_Distance_To_Fire_Points, lodgepole.Horizontal_Distance_To_Fire_Points,

        ponderosa.Horizontal_Distance_To_Fire_Points, willow.Horizontal_Distance_To_Fire_Points,

        aspen.Horizontal_Distance_To_Fire_Points, douglas.Horizontal_Distance_To_Fire_Points, 

        krummholz.Horizontal_Distance_To_Fire_Points]

plt.xticks([1, 2, 3, 4, 5, 6, 7])

plt.boxplot(firepoint_data)

plt.show()
road_fire = df[['Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']]

road_fire.corr()
x = [spruce.Horizontal_Distance_To_Roadways.median(), lodgepole.Horizontal_Distance_To_Roadways.median(),

        ponderosa.Horizontal_Distance_To_Roadways.median(), willow.Horizontal_Distance_To_Roadways.median(),

        aspen.Horizontal_Distance_To_Roadways.median(), douglas.Horizontal_Distance_To_Roadways.median(), 

        krummholz.Horizontal_Distance_To_Roadways.median()]

y = [spruce.Horizontal_Distance_To_Fire_Points.median(), lodgepole.Horizontal_Distance_To_Fire_Points.median(),

        ponderosa.Horizontal_Distance_To_Fire_Points.median(), willow.Horizontal_Distance_To_Fire_Points.median(),

        aspen.Horizontal_Distance_To_Fire_Points.median(), douglas.Horizontal_Distance_To_Fire_Points.median(), 

        krummholz.Horizontal_Distance_To_Fire_Points.median()]

print(x)

print(y)

plt.figure()

plt.title("Roadway Dist Medians vs. Fire Point Dist Medians")

plt.xlabel("Distance to Roadways (in meters)")

plt.ylabel("Distance to Fire Points (in meters)")

plt.scatter(x, y)

plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

plt.show()
print(np.corrcoef(x,y))
soil_counts = []

for num in range(1,41):

    col = ('Soil_Type' + str(num))

    this_soil = df[col].groupby(df['Cover_Type'])

    totals = []

    for value in this_soil.sum():

        totals.append(value)

    total_sum = sum(totals)

    soil_counts.append(total_sum)

    print("Total Trees in Soil Type {0}: {1}".format(num, total_sum))

    percentages = [ (total*100 / total_sum) for total in totals]

    print("{0}\n".format(percentages))

print("Number of trees in each soil type:\n{0}".format(soil_counts))

        
soil_types = range(1,41)

xs = [i + 0.2 for i, _ in enumerate(soil_types)]

plt.bar(xs, soil_counts)

plt.ylabel("# of Trees")

plt.title("Soil Types in Roosevelt National Forest, Colorado")

plt.xticks([i + 0.75 for i, _ in enumerate(soil_types)], soil_types, rotation='vertical');




soil_counts = []

for num in range(1,41):

    col = ('Soil_Type' + str(num))

    this_soil = df[col].groupby(df['Cover_Type'])

    totals = []

    for value in this_soil.sum():

        totals.append(value)

    total_sum = sum(totals)

    soil_counts.append(total_sum)

    print("Total Trees in Soil Type {0}: {1}".format(num, total_sum))

    percentages = [ (total*100 / total_sum) for total in totals]

    print("{0}\n".format(percentages))

print("Number of trees in each soil type:\n{0}".format(soil_counts))

        