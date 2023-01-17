import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv("../input/voted-kaggle-kernels.csv")
df.head()
# Need to first separate out all of the versions in each set.
df["Version History"] = df["Version History"].str.split("|")

# Since not all kernels have revisions, we'll need to replace nan with empty lists
df.loc[df["Version History"].isnull(), "Version History"] = \
    df.loc[df["Version History"].isnull(), "Version History"].apply(lambda x: [])
    
df["Revisions"] = [len(row) for row in df["Version History"]]
df.head()
tmp = {"Kernel ID":[], "Revision":[]}
times = [] # Will be used for indexing
for index, row in df.iterrows():
    for versions in list(row["Version History"]):
        versionNum, versionDate = versions.split(',')
        tmp["Kernel ID"].append(index)
        tmp["Revision"].append(versionNum[versionNum.rfind(" ")+1:])
        # 
        times.append(versionDate)

revDF = pd.DataFrame(tmp, index = pd.DatetimeIndex(times))
revDF.sort_index(inplace=True)
revDF.index.name = "Date"
revDF.head()
def nextAxis(maxRow, maxCol):
    curRow, curCol = 0, 0
    while maxRow > curRow:
        yield (curRow, curCol)
        curCol += 1
        if maxCol == curCol:
            curCol = 0
            curRow += 1
            
yearIter = range(2015, 2019)

fig, axes = plt.subplots(nrows=2, ncols=2)

axisGen = nextAxis(2, 2)

for year, color, axisPoints in zip(yearIter, ['b', 'g', 'r', 'y'], axisGen):
    valueDF = revDF['{}'.format(year)].groupby(["Date"])["Kernel ID"].count()
    
    print("{} has {} revisions".format(year, valueDF.sum()))

    # Get the next axis points
    points = [point for point in axisPoints]
    
    # Plot the graph
    valueDF.plot(kind = "line", figsize=(16,16), label=year, ax = axes[points[0], points[1]], color=[color])
tags = df['Tags'].str.strip(',') # Remove unnecessary ','
tags = tags.str.split(',') # Split each tag into a separate element

tmp = {"Tag":[]}
indexes = [] # Will be used for indexing
# Since not all kernels have revisions, we'll need to replace nan with empty lists
index = 0
for row in tags:
    if type(row) != list:
        index += 1
        continue
    for tag in list(row):
        tmp["Tag"].append(tag)
        indexes.append(index)
    index += 1
        
# Now create the tag DataFrame
tagDF = pd.DataFrame(tmp, index = indexes)
tagDF.index.name = "Kernel ID"
tagDF.head()
tagDF.loc[1]
tagThreshold = 5

tagValueCount = tagDF['Tag'].value_counts()
tagValueCount[tagValueCount > tagThreshold].plot(kind='pie',figsize=(15,15))
mostKernels = df['Owner'].value_counts()
mkUser = mostKernels.keys()[0]
mkUser
mkuDF = df[df['Owner'] == mkUser]
mkuDF
print("{:.2f}%".format(mkuDF['Tags'].isnull().sum() / len(mkuDF) * 100))
tagDF.loc[mkuDF.index, 'Tag'].value_counts().plot(kind = 'pie', figsize=(10, 10), fontsize=14)
revOwn = df.groupby('Owner')['Revisions'].sum().sort_values(ascending=False)
mostRevUser = revOwn.keys()[0]
print("{} with {} revisions".format(mostRevUser, revOwn[mostRevUser]))
votesOwner = df.groupby('Owner')['Votes'].sum().sort_values(ascending=False)
mostVotesUser = votesOwner.keys()[0]
print("{} with {} votes".format(mostVotesUser, votesOwner[mostVotesUser]))
viewsOwner = df.groupby('Owner')['Views'].sum().sort_values(ascending=False)
mostViewsUser = viewsOwner.keys()[0]
print("{} with {} votes".format(mostViewsUser, viewsOwner[mostViewsUser]))
commentsOwner = df.groupby('Owner')['Comments'].sum().sort_values(ascending=False)
mostCommentsUser = commentsOwner.keys()[0]
print("{} with {} comments".format(mostCommentsUser, commentsOwner[mostCommentsUser]))
forksOwner = df.groupby('Owner')['Forks'].sum().sort_values(ascending=False)
mostForksUser = forksOwner.keys()[0]
print("{} with {} forks".format(mostForksUser, forksOwner[mostForksUser]))
df['Language'].value_counts().plot(kind = 'bar')