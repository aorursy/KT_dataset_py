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
             "HispanicOriginRaceRecode"], inplace=True, axis=1)
print(deaths.columns)
suicidals = deaths.loc[deaths["MannerOfDeath"] == 2]
import re
from collections import Counter

def make_fancy_plot(colname, funk=lambda x:x, values=None, rotation=None):
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    f.suptitle(re.sub("([a-z])([A-Z])","\g<1> \g<2>", colname))
    if values is None:
        my_list = list(deaths[colname])
        values = [
            e
            for i, e in enumerate(my_list)
            if my_list.index(e) == i
        ]
    x = [funk(a) for a in values]
    x = np.array(x)
    
    l = list(deaths[colname])
    c = Counter(l)
    y1 = [c[a] for a in values]
    
    sns.barplot(x, y1, palette="BuGn_d", ax=ax1)
    ax1.set_ylabel("All deaths")
    
    l = list(suicidals[colname])
    c = Counter(l)
    y2 = [c[a] for a in values] 
    
    g = sns.barplot(x, y2, palette="BuGn_d", ax=ax2)
    ax2.set_ylabel("Suicides")
    if rotation is not None:
        plt.xticks(rotation=rotation)
    sns.despine(bottom=True)
def unique_list(my_list):
    return([
        e
        for i, e in enumerate(my_list)
        if my_list.index(e) == i
    ])
make_fancy_plot("Sex",values=["F","M"], funk=(lambda x: "Female" if x=="F" else "Male"))
def unique_two_lists(x, y):
    for a, b in zip(x, y):
        if a in d.keys():
            d[a] += b
        else:
            d[a] = b
    x = unique_list(x)
    y = [d[a] for a in x]
    return (x,y)
days = {1: "Sunday",
        2: "Monday",
        3: "Tuesday",
        4: "Wednesday",
        5: "Thursday",
        6: "Friday",
        7: "Saturday"}

make_fancy_plot("DayOfWeekOfDeath", 
                values=[2, 3, 4, 5, 6, 7, 1], 
                funk=(lambda x: days[x]))
months = {1: "January",
          2: "February",
          3: "March",
          4: "April",
          5: "May",
          6: "June",
          7: "July",
          8: "August",
          9: "September",
          10: "October",
          11: "November",
          12: "December"}

make_fancy_plot("MonthOfDeath", 
                values=range(1,13), 
                funk=(lambda x: months[x]),
                rotation=60)
make_fancy_plot("ResidentStatus", 
                values=range(1,5), 
                funk=(lambda x: "Resident" if x == 1 else "Non-resident"))
f = lambda x: list(mar.loc[mar["Code"] == x]["Description"])[0]

make_fancy_plot("MaritalStatus", 
                values=list(mar["Code"]), 
                funk=f,
                rotation=60)
f = lambda x: list(edu.loc[edu["Code"] == x]["Description"])[0]

make_fancy_plot("Education2003Revision", 
                values=list(edu["Code"]), 
                funk=f,
                rotation=80)
# Draw a nested barplot to show survival for class and sex
g = sns.factorplot(x="MaritalStatus", y="Age", hue="Sex", data=deaths,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)

