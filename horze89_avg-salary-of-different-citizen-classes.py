# Graphic Settings
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.size'] = 18
# Loading data
filename1 = "../input/pums/ss13pusa.csv"
filename2 = "../input/pums/ss13pusb.csv"

# CIT:  Citizenship status
# SCHL: Educational attainment
# WAGP: Wages or salary income past 12 months
cols = ["CIT","SCHL","WAGP"]
print("Reading " + filename1 + "...")
dfA = pd.read_csv(filename1,usecols = cols)
print(str(dfA.shape[0]) + " rows were readed.")
print("Reading " + filename2 + "...")
dfB = pd.read_csv(filename2,usecols = cols)
print(str(dfB.shape[0]) + " rows were readed.")

# Cutting Frames into a single dataframe (ignore_index to avoid doubling indexes)
print("Catting frames into a single dataframe...")
frames = [dfA, dfB]
df = pd.concat(frames,ignore_index="True")
print("Dataframe was loaded containing " + str(df.shape[0]) + " tuples")
# Visualization (by means of PieChart) the Citizenship status presence in the USA (2013)
# Citizenship status
# 1 .Born in the U.S.
# 2 .Born in Puerto Rico, Guam, the U.S. Virgin Islands,.or the Northern Marianas
# 3 .Born abroad of American parent(s)
# 4 .U.S. citizen by naturalization
# 5 .Not a citizen of the U.S.

# Comparison between (1) US Citizens and Others (2,3,4,5) 
tmpDF = df.copy()
fig = plt.figure(figsize=(18,8))
fig.suptitle('US Citizens vs Foreign Citizens', fontsize=25)

mySeries = pd.Series(tmpDF.CIT)
mySeries[mySeries > 1] = 2 # Classes 2,3,4 and 5 are generalized as the same class in a dataframe copy

myCounts = mySeries.value_counts()
myDict   = dict(myCounts)

# Create 1st PieChart
labels = ["1) US Native Citizens","Others"]
sizes = [float(f) for f in myDict.values()]
colors = ['green', 'lightskyblue']
plt.subplot(1,2,1)
patches, texts = plt.pie(sizes,colors=colors, explode = (0, 0.2), shadow=True,startangle=90)
plt.legend(patches,labels,loc="best")

# Visualization of the remaining classes in a 2nd PieChart
tmpDF = df.copy()
mySeries = pd.Series(tmpDF.CIT)
mySeries  = mySeries[mySeries > 1]

myCounts = mySeries.value_counts()
myDict   = dict(myCounts)

labels = ["2) P.R., Guam, V.I. or N.M.I.",
          "3) American parent(s)","4) Citizen by naturalization","5) Not a citizen of the U.S."]
sizes = [float(f) for f in myDict.values()]
colors = ['gold','purple','cyan','green']

plt.subplot(1,2,2)
patches, texts = plt.pie(sizes,colors=colors, shadow=True, startangle=45)

plt.legend(patches,labels,loc="best")
plt.show()
### Comparison between AVG Salary (2013) of the 5 classes of citizens


# Get the sorted list of the unique value representing the 5 classes of citizens
ind = np.sort(pd.Series.unique(pd.Series(df.CIT)))

# Convert SCHL level as follow:
# 1-20: < Bachelor's degree -> 1
# 21 .Bachelor's degree     -> 2
# 22 .Master's degree       -> 3
# 23 .Professional degree   -> 4
# 24 .Doctorate degree      -> 5
tmpDF = df.copy()
tmpDF.ix[tmpDF.SCHL<21, "SCHL"] = 1
tmpDF.ix[tmpDF.SCHL==21,"SCHL"] = 2
tmpDF.ix[tmpDF.SCHL==22,"SCHL"] = 3
tmpDF.ix[tmpDF.SCHL==23,"SCHL"] = 4
tmpDF.ix[tmpDF.SCHL==24,"SCHL"] = 5
schls = [1,2,3,4,5]
nSchls = len(schls)
# Group by (Class of Cirizen ans Scholar Level) and get the mean value
grouped = tmpDF.groupby(["CIT","SCHL"]).mean()

# Create the figure
plt.close('all')
fig, ax = plt.subplots() 
fig.set_size_inches(20,10)
fig.suptitle('Average Salary', fontsize=25)

colors = ["r","g","b","y","c"]
width  = 0.1
rIdx   = 0
rects  = []

# For each School Level get and draw the average salary for all class of citiz.
for i in range(nSchls):
    schl = schls[i]
    groupedRI = grouped.reset_index() # Dataframe normalization
    meansWAGP = tuple(groupedRI.loc[groupedRI["SCHL"] == schl, "WAGP"])    
    g = ax.bar(ind + i*width, meansWAGP, width, color=colors[i])
    rects.append(g)
    ax.set_xticks(ind + i*width - 1.5*width)
    rIdx = rIdx + 1

ax.legend(
    (rects[0],rects[1],rects[2],rects[3],rects[4]),
    ("No Degree","Bachelor's","Master's","Professional","PhD"))
ax.set_ylabel("2013 AVG Salary")
ax.set_xticklabels(("US Native","P.R., Guam, V.I. or N.M.I.","American parent(s)","Naturalization","Not a US citizen"))

plt.show()

