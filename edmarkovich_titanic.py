import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# For feature engineering, let's combine test and train data sets so we

# can have a fuller picture of the passengers.



train = pd.read_csv("../input/train.csv")

train.drop("Survived",1,inplace=True)

test = pd.read_csv("../input/test.csv")

train = train.append(test)
# Let's break appart the "Name" into "Family Name"

train['FamilyName'] = train.Name.apply(lambda x: x.split(",")[0])

train.drop("Name", 1, inplace= True)
# Let's figure out if any missing cabins have last names or ticket numbers that do?

cabins = train#[["FamilyName","Cabin"]]

nanNames = cabins[cabins.Cabin.isnull()]

cabinNames = cabins[-cabins.Cabin.isnull()].FamilyName

candates = nanNames[nanNames.FamilyName.isin(cabinNames)].FamilyName.unique()



cabins[FamilyName.isin(candidates)]