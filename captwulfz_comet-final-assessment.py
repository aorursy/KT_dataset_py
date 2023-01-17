# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
mainDataSet = pd.read_csv("../input/train.csv")
mainDataSet
# Moran, Mr. James - 27 Years Old
# Williams, Mr. Charles Eugene -  24 Years Old
# Emir, Mr. Farred Chehab - 26 Years Old
# O'Dwyer, Miss. Ellen "Nellie" - 24 Years Old
# Todoroff, Mr. Lalio - 23 Years Old
# Giles, Mr. Frederick Edward - 21 Years Old
# Sage, Miss. Dorothy Edith "Dolly" - 14 months
# van Melkebeke, Mr. Philemon - 23 Years Old
# Laleff, Mr. Kristo - 23 Years Old
# Johnston, Miss. Catherine Helen "Carrie" - 7 Years Old
#People without Ages
noAgeSet = mainDataSet[mainDataSet.Age != mainDataSet.Age]
noAgeSet

#People who Survived
survivedSet = noAgeSet[noAgeSet.Survived == 1]
survivedSet
#People who Survived and Male
maleSurvivedSet = survivedSet[survivedSet.Sex == "male"]
maleSurvivedSet
#People who Survived and Female
femaleSurvivedSet = survivedSet[survivedSet.Sex == "female"]
femaleSurvivedSet

#People who Died
deadSet = noAgeSet[noAgeSet.Survived == 0]
deadSet
#People who Died and Male
maleDeadSet = deadSet[deadSet.Sex == "male"]
#People who Died and Female
femaleDeadSet = deadSet[deadSet.Sex == "female"]

noAgeSet.groupby('PassengerId').count()


EaS = mainDataSet[["Survived", "Embarked"]]
EaS = EaS.sort_values("Embarked")
ECount2 = EaS.groupby("Embarked").count()
ECount2
SCount = EaS.groupby("Survived").count()
SCount
numDA = SCount["Embarked"]
labels = "Dead", "Survived"

fig2, ax2 = plt.subplots()
ax2.pie(numDA, labels = labels, autopct = '%1.1f%%', startangle=90)
ax2.axis('equal')
plt.show()
EaSCount = EaS[EaS.Survived == 1]
EaSCount = EaSCount.groupby("Embarked").count()
labels2 = "Cherbourg", "Queenstown", "Southampton"
numS = EaSCount["Survived"]
fig1, ax1 = plt.subplots()
ax1.pie(numS, labels = labels2, autopct = '%1.1f%%', startangle=90)
ax1.axis('equal')
plt.show()
PaE = mainDataSet[["Embarked", "Pclass", "Survived"]]
PaE = PaE.sort_values("Embarked")

ECount = PaE["Embarked"]
ClCount = PaE["Pclass"]

data = pd.DataFrame({'Embarked': ECount, 
                     'PassengerClass': ClCount})

pd.crosstab(data.Embarked, data.PassengerClass).plot.barh(stacked=True)


numE = ECount2["Survived"]
numEPercent = pd.Series(numS/numE * 100)
numEPercent
