# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import ttest_ind
import matplotlib.pyplot as plot
import seaborn as sns
sns.set()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/StudentsPerformance.csv")
display(data.head())
#male vs female
fdata = data[data["gender"]=="female"]["math score"]
mdata = data[data["gender"]=="male"]["math score"]

print(fdata.shape)
print(mdata.shape)

plot.ylabel("Average math score")
plot.bar(["F", "M"], [fdata.mean(), mdata.mean()])
plot.show()

t,p = ttest_ind(mdata, fdata)
print(p/2)
#male vs female
fdata = data[data["gender"]=="female"]["reading score"]
mdata = data[data["gender"]=="male"]["reading score"]

plot.ylabel("Average reading score")
plot.bar(["F", "M"], [fdata.mean(), mdata.mean()])
plot.show()

t,p = ttest_ind(mdata, fdata)
print(p/2)
#male vs female
fdata = data[data["gender"]=="female"]["writing score"]
mdata = data[data["gender"]=="male"]["writing score"]

plot.ylabel("Average writing score")
plot.bar(["F", "M"], [fdata.mean(), mdata.mean()])
plot.show()

t,p = ttest_ind(mdata, fdata)
print(p/2)
avgDat = data[["math score", "reading score", "writing score"]].mean(axis=1).rename("Score")
data = pd.concat([data, avgDat], 1)
display(data.head())
slunch = data[data["lunch"]=="standard"]["Score"]
flunch = data[data["lunch"]=="free/reduced"]["Score"]

print(slunch.shape)
print(flunch.shape)

plot.ylabel("Average score")
plot.bar(["standard lunch", "free/reduced lunch"], [slunch.mean(), flunch.mean()])
plot.show()

t,p = ttest_ind(slunch, flunch)
print(p/2)
preps = data["test preparation course"].unique()
print(preps)
noprep = data[data["test preparation course"]=="none"]["Score"]
prep = data[data["test preparation course"]=="completed"]["Score"]

print(noprep.shape)
print(prep.shape)

plot.ylabel("Average score")
plot.bar(["no preparation course", "completed course"], [noprep.mean(), prep.mean()])
plot.show()

t,p = ttest_ind(noprep, prep)
print(p/2)
eth = data["race/ethnicity"].unique()
print(eth)

scores = []

for e in eth:
    scores.append(data[data["race/ethnicity"]==e]["Score"].mean())

srt = np.argsort(scores)

scores = np.array(scores)
scores = scores[srt]
eth = eth[srt]
    
plot.ylabel("Average score")
plot.bar(eth, scores)
plot.show()

for e1 in eth:
    for e2 in eth:
        if e1 == e2:
            continue
        
        s1 = data[data["race/ethnicity"]==e1]["Score"]
        s2 = data[data["race/ethnicity"]==e2]["Score"]
        
        if s1.mean() > s2.mean():
            t,p = ttest_ind(s1, s2)

            if p/2 < 0.05:
                print("\""+e1+"\" scores significantly higher than \""+e2+"\"\tp/2="+str(p/2))

edu = data["parental level of education"].unique()
print(edu)

scores = []

for e in edu:
    scores.append(data[data["parental level of education"]==e]["Score"].mean())

srt = np.argsort(scores)

scores = np.array(scores)
scores = scores[srt]
edu = edu[srt]

plot.ylabel("Average score")
plot.bar(edu, scores)
plot.xticks(rotation=45)
plot.show()

for e1 in edu:
    for e2 in edu:
        if e1 == e2:
            continue
        
        s1 = data[data["parental level of education"]==e1]["Score"]
        s2 = data[data["parental level of education"]==e2]["Score"]
        
        if s1.mean() > s2.mean():
            t,p = ttest_ind(s1, s2)

            if p/2 < 0.05:
                print("\""+e1+"\" scores significantly higher than \""+e2+"\"\tp/2="+str(p/2))

