import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
deaths = pd.read_csv('../input/DeathRecords.csv')
manners = pd.read_csv('../input/MannerOfDeath.csv')
icd10 = pd.read_csv('../input/Icd10Code.csv')
age = pd.read_csv('../input/AgeType.csv')
race = pd.read_csv('../input/Race.csv')
loc = pd.read_csv('../input/PlaceOfDeathAndDecedentsStatus.csv')
pla = pd.read_csv('../input/PlaceOfInjury.csv')
mar = pd.read_csv('../input/MaritalStatus.csv')
disp = pd.read_csv('../input/MethodOfDisposition.csv')
edu = pd.read_csv('../input/Education2003Revision.csv')
res = pd.read_csv('../input/ResidentStatus.csv')
deaths.drop(["Education1989Revision",
             "EducationReportingFlag",
             "AgeSubstitutionFlag",
             "AgeRecode52",
             "AgeRecode27",
             "AgeRecode12",
             "InfantAgeRecode22",
             "CauseRecode358",
             "CauseRecode113",
             "InfantCauseRecode130",
             "CauseRecode39",
             "NumberOfEntityAxisConditions",
             "NumberOfRecordAxisConditions",
             "BridgedRaceFlag",
             "RaceImputationFlag",
             "RaceRecode3",
             "RaceRecode5",
             "HispanicOrigin",
             "HispanicOriginRaceRecode",
             "CurrentDataYear"], inplace=True, axis=1)
print(deaths.columns)
features_to_drop = ["Id", "MaritalStatus", "ResidentStatus", "MethodOfDisposition", "InjuryAtWork"]
deaths.drop(features_to_drop, inplace=True, axis=1)
#odrzucamy dziwne obserwacje
deaths = deaths[deaths.Age < 200]
#male dzieci
x = list(deaths["AgeType"])
y = list(deaths["Age"])
z = [b if a == 1 else 0 for a, b in zip(x, y)]
deaths["Age"] = z
deaths.drop("AgeType", inplace=True, axis=1)
#chcemy znac edukacje
deaths = deaths[deaths.Education2003Revision < 9]
deaths = deaths[deaths.Education2003Revision > 0]
from sklearn.cross_validation import train_test_split
from sklearn import tree
clf = tree.DecisionTreeClassifier()
train, test = train_test_split(deaths, test_size = 0.2)

X = train[list(train.columns)[1:]]
Y = train["Education2003Revision"]
from sklearn import preprocessing

def change_to_one_hot(colname):
    lb = preprocessing.LabelBinarizer()
    OH = pd.DataFrame(lb.fit_transform(deaths[colname]))
    deaths.drop([colname], inplace=True, axis=1)
    
len(set(X["Icd10Code"]))
from sklearn import metrics
from sklearn import cross_validation
scores = cross_validation.cross_val_score(clf, X, Y, cv=5, scoring='f1_weighted')
print(scores)
