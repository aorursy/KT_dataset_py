%matplotlib inline

import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# Store start time to check script's runtime
scriptStartTime = time.time()

# Read file
df = pd.read_csv("../input/train.csv")

df["Dates"] = pd.to_datetime(df["Dates"])
# What are the columns in this dataset
print(df.columns)
# Let's see what the Categories are
print(df["Category"].unique())
# Check amount of unique values
print("Uniques:")
for column in df.columns:
    print("Unique in '" + column + "': " + str(df[column].nunique()))
# Amount of crimes per category
groups = df.groupby("Category")["Category"].count()
groups = groups.sort_values(ascending=0)
plt.figure()
groups.plot(kind='bar', title="Category Count")
print(groups)
# Largest category is LARCENY/THEFT, let's investigate it further
dfTheft = df[df["Category"] == "LARCENY/THEFT"]
groups = dfTheft.groupby("DayOfWeek")["Category"].count()
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
groups = groups[weekdays]
plt.figure()
groups.plot(kind="bar", title="LARCENY/THEFT per weekday")
print(groups)
# Find the crime group with highest per-day-Coefficient Of Variation
dayOfWeekVars = pd.DataFrame(columns=["Category", "CoefficientOfVariation"])
rows = []
for c in df["Category"].unique():
    dfSubset = df[df["Category"] == c]
    dfSubsetGrouped = dfSubset.groupby("DayOfWeek")["Category"].count()
    std = dfSubsetGrouped.std()
    mean = dfSubsetGrouped.mean()
    cv = std / mean
    
    # Only consider category, if there are enough samples
    if (len(dfSubset) > 300):
        rows.append({'Category': c, 'CoefficientOfVariation': cv})

categoryDayCV = pd.DataFrame(rows).sort_values(by="CoefficientOfVariation", ascending=0)
#plt.figure()
categoryDayCV.plot(x="Category", kind="bar", title="Category Day Coefficient Of Variation")
plt.show()

print("Top 5 Coefficient Of Variation by day:")
print(categoryDayCV["Category"][:5])
print("Bottom 5 Coefficient Of Variation by day:")
print(categoryDayCV["Category"][-5:])
for category in categoryDayCV["Category"][:5]:
    dfCategory = df[df["Category"] == category]
    groups = dfCategory.groupby("DayOfWeek")["Category"].count()
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    groups = groups[weekdays]
    plt.figure()
    groups.plot(kind="bar", title=category + " count by day")
# Function to plot data grouped by Dates and Category
def plotTimeGroup(dfGroup, ncols=10, area=False, title=None):
    categoryCV = pd.DataFrame(columns=["Category", "CV"])
    rows = []

    for column in dfGroup.columns:
        col = dfGroup[column]
        # Only consider category, if there are enough samples
        if (col.sum() > 500):
            rows.append({'Category': column, 'CV': col.std() / col.mean()})

    categoryCV = pd.DataFrame(rows).sort_values(by="CV", ascending=0)
    #The graph with all categories is unreadable. Therefore, columns with a
    # high coefficient of variation are extracted:
    topCVCategories = categoryCV[:ncols]["Category"].tolist()


    f = plt.figure(figsize=(13,8))
    ax = f.gca()
    if area:
        dfGroup[topCVCategories].plot.area(ax=ax, title=title, colormap="jet")
    else:
        dfGroup[topCVCategories].plot(ax=ax, title=title, colormap="jet")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=1, fontsize=11)

# Crime category count per year
dfGroup = df[["Dates", "Category"]]
# Drop year 2015 because it does not contain all months
dfGroup = dfGroup[dfGroup["Dates"].map(lambda x: x.year < 2015)]
dfGroup = dfGroup.groupby([dfGroup["Dates"].map(lambda x: x.year), "Category"])
dfGroup = dfGroup.size().unstack()

plotTimeGroup(dfGroup, title="Crime Categories History")
plotTimeGroup(dfGroup, title="Crime Categories History", area=True)
# Crime category count per year
dfGroup = df[["Dates", "Category"]]
# Drop year 2015 because it does not contain all months
dfGroup = dfGroup[dfGroup["Dates"].map(lambda x: x.year < 2015)]
dfGroup = dfGroup.groupby([dfGroup["Dates"].map(lambda x: x.month), "Category"])
dfGroup = dfGroup.size().unstack()

plotTimeGroup(dfGroup, ncols=15, title="Crime Categories Per Month")
plotTimeGroup(dfGroup, ncols=15, title="Crime Categories Per Month", area=True)