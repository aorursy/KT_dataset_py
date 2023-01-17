import pandas as pd
import matplotlib.pyplot as plt

# Get path to data sets in notebook
"""
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""   
plt.style.use('ggplot')
df = pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
# Count of houses in neighborhood to determine what which neighborhoods are the most popular or prevalent in the data set.
neighborhood = dict()

for house in df['Neighborhood']:
    if house not in neighborhood:
        neighborhood[house] = 1
    else:
        neighborhood[house] += 1

plt.xticks(rotation=90)
plt.bar(neighborhood.keys(), neighborhood.values())
plt.xlabel("Neighborhood")
plt.ylabel("# Of Homes")
plt.title("Count of Homes in Neighborhood")
plt.show()
# Average house value by neighborhood in dataset

neighborhood_values = dict()

for house in df['Neighborhood']:
    if house not in neighborhood_values:
        neighborhood_values[house] = 0

for house in neighborhood_values:
    neighborhood_values[house] = df['SalePrice'].where(df['Neighborhood'] == house).mean()

title = "Average House Price by Neighborhood"
plt.style.use('seaborn')
plt.xlabel('Neighborhood')
plt.ylabel('Average House Value')
plt.title("Avg. Home Value by Neighborhood")
plt.xticks(rotation=90)
plt.bar(neighborhood_values.keys(), neighborhood_values.values())
plt.show()
# This graph helps visualize if their is a linear correlation between when the house was built, remodeled, and sold.

# Average house value by year built

year_built = dict()

for year in df['YearBuilt']:
    if year not in year_built:
        year_built[year] = 0

for year in year_built:
    year_built[year] = df['SalePrice'].where(df['YearBuilt'] == year).mean()

# Average house value by year remodeled

year_renovated = dict()

for row in df['YearRemodAdd']:
    if row not in year_renovated:
        year_renovated[row] = 0

for year in year_renovated:
    year_renovated[year] = df['SalePrice'].where(df['YearRemodAdd'] == year).mean()
# Average house value by year sold

year_sold = dict()

for row in df['YrSold']:
    if row not in year_sold:
        year_sold[row] = 0

for year in year_sold:
    year_sold[year] = df['SalePrice'].where(df['YrSold'] == year).mean()

# Average price by most recent renovation date of the house. I imagine the price value will be graphed in a more pure linear fasion based on the latest year renovated.
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(year_built.keys(), year_built.values(), c='r', label="Year Built")
ax1.scatter(year_renovated.keys(), year_renovated.values(), c='b', label='Year Renovated')
ax1.scatter(year_sold.keys(), year_sold.values(), c='g', label="Year Sold")
plt.xticks(rotation=90)
plt.xlabel('Year Built')
plt.ylabel('Average Price')
plt.title("Home Value Comparison by Year")
plt.legend(loc="upper center")
plt.show()
# Is there a relation between the overall quality and value of the house finish? and overall condition of house in general?
# Turns out the overall quality has a more linear relationship to the value of the house than overall finish. Making overall quality more reliable in determining house value by these labels.

quality = dict()
for house in df['OverallQual']:
    if house not in quality:
        quality[house] = 0

for rating in quality:
    quality[rating] = df['SalePrice'].where(df['OverallQual'] == rating).mean()


condition = dict()
for house in df['OverallCond']:
    if house not in condition:
        condition[house] = 0

for rating in condition:
    condition[rating] = df['SalePrice'].where(df['OverallCond'] == rating).mean()


fig = plt.figure()
ax1 = fig.add_subplot()

ax1.scatter(condition.keys(), condition.values(), c='b', label='Condition of Finish')
ax1.scatter(quality.keys(), quality.values(), c='r', label='General Quality')
plt.legend(loc='upper left')
plt.xlabel('Rating of Home')
plt.ylabel('Avg. Value of Home')
plt.title("Comparison of Rating and Quality of Home")
plt.show()

# Value of home based on zoning type

zone = dict()

for house in df['MSZoning']:
    if house not in zone:
        zone[house] = 0

for prop_zone in zone:
    zone[prop_zone] = df['SalePrice'].where(df['MSZoning'] == prop_zone).mean()

plt.title("Avg. House Values by Zoning")
plt.xlabel("Zoning Type")
plt.ylabel("House Value")
plt.bar(zone.keys(), zone.values(), color='r')
plt.show()


"""
Scatter Plot analysis of value of home compared to : BldgType, HouseStyle, and MSSubClass
"""


BldgType = dict()
HouseStyle = dict()
MSSubClass = dict()

# fill dictionaries with values

for prop in df['BldgType']:
    if prop not in BldgType:
        BldgType[prop] = 0

for prop in df['HouseStyle']:
    if prop not in HouseStyle:
        HouseStyle[prop] = 0

for prop in df['MSSubClass']:
    if prop not in MSSubClass:
        MSSubClass[prop] = 0

# Fill dictionaries with values

for key in HouseStyle:
    HouseStyle[key] = df['SalePrice'].where(df['HouseStyle'] == key).mean()

for key in BldgType:
    BldgType[key] = df['SalePrice'].where(df['BldgType'] == key).mean()

for key in MSSubClass:
    MSSubClass[key] = df['SalePrice'].where(df['MSSubClass'] == key).mean()


plt.figure(figsize=(16, 4))
plt.subplot(131)
plt.xlabel("Building Type")
plt.ylabel("House Value Avg.")
plt.title("Building Type Comparison")
plt.scatter(BldgType.keys(), BldgType.values(), c='b')
plt.subplot(132)
plt.xlabel("House Style")
plt.title("House Style Data Comparison")
plt.scatter(HouseStyle.keys(), HouseStyle.values(), c='r')
plt.subplot(133)
plt.xlabel("MS Sub Class")
plt.title("MS Sub Class Comparison")
plt.scatter(MSSubClass.keys(), MSSubClass.values(), c='g')
plt.show()


# This graph is going to visualize the number of houses sold in each year and month respectively to get a better understanding of the distribution of when homes are sold by year and by month.

year_sold = dict()

for row in df['YrSold']:
    if row not in year_sold:
        year_sold[row] = 1
    else:
        year_sold[row] += 1


month_sold = dict()

for row in df['MoSold']:
    if row not in month_sold:
        month_sold[row] = 1
    else:
        month_sold[row] += 1

plt.figure(figsize=(16,4))
plt.subplot(131)
plt.xticks(rotation=45)
plt.xlabel("Year Sold")
plt.ylabel("Count of Homes Sold")
plt.title("No. of Homes Sold Each Year")
plt.bar(year_sold.keys(), year_sold.values())
plt.subplot(132)
plt.xticks(rotation=45)
plt.xlabel("Month Sold")
plt.ylabel("No. of Homes Sold")
plt.title("No. of Homes Sold by Month")
plt.bar(month_sold.keys(), month_sold.values(), color='blue')
plt.show()
