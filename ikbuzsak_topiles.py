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
top10 = deaths[['Icd10Code', 'Id']].groupby(['Icd10Code']).count().sort_values(['Id'], ascending=False).head(10)
top10 = pd.merge(top10, icd10, left_index=True, right_on=['Code'])
top10.plot(kind='bar', x='Description')
suicidals = deaths.loc[deaths["MannerOfDeath"] == 2]
top10 = suicidals[['Icd10Code', 'Id']].groupby(['Icd10Code']).count().sort_values(['Id'], ascending=False).head(10)
top10 = pd.merge(top10, icd10, left_index=True, right_on=['Code'])
top10.plot(kind='bar', x='Description')
top10 = suicidals[['Icd10Code', 'Id']].where(suicidals["Sex"]=="F").groupby(['Icd10Code']).count().sort_values(['Id'], ascending=False).head(10)
top10 = pd.merge(top10, icd10, left_index=True, right_on=['Code'])
top10.plot(kind='bar', x='Description')
top10 = suicidals[['Icd10Code', 'Id']].where(suicidals["Sex"]=="M").groupby(['Icd10Code']).count().sort_values(['Id'], ascending=False).head(10)
top10 = pd.merge(top10, icd10, left_index=True, right_on=['Code'])
top10.plot(kind='bar', x='Description')
top10 = deaths[['Icd10Code', 'Id']].where(deaths["Race"]==2).where(deaths["MannerOfDeath"]!=7).groupby(['Icd10Code']).count().sort_values(['Id'], ascending=False).head(10)
top10 = pd.merge(top10, icd10, left_index=True, right_on=['Code'])
top10.plot(kind='bar', x='Description')
top10 = deaths[['Icd10Code', 'Id']].where(deaths["Race"]==1).where(deaths["MannerOfDeath"]!=7).groupby(['Icd10Code']).count().sort_values(['Id'], ascending=False).head(10)
top10 = pd.merge(top10, icd10, left_index=True, right_on=['Code'])
top10.plot(kind='bar', x='Description')
top10 = deaths[['Icd10Code', 'Id']].where(deaths["AgeType"]!=1).groupby(['Icd10Code']).count().sort_values(['Id'], ascending=False).head(10)
top10 = pd.merge(top10, icd10, left_index=True, right_on=['Code'])
top10.plot(kind='bar', x='Description')
top10 = deaths[['Icd10Code', 'Id']].where(deaths["AgeType"]==1).where(deaths["Age"]<=12).where(deaths["Age"]>=8).groupby(['Icd10Code']).count().sort_values(['Id'], ascending=False).head(10)
top10 = pd.merge(top10, icd10, left_index=True, right_on=['Code'])
top10.plot(kind='bar', x='Description')
from collections import Counter
import re

def unique_list(my_list):
    return([
        e
        for i, e in enumerate(my_list)
        if my_list.index(e) == i
    ])
    

def unique_two_lists(x, y):
    d = {}
    for a, b in zip(x, y):
        if a in d.keys():
            d[a] += b
        else:
            d[a] = b
    x = unique_list(x)
    y = [d[a] for a in x]
    return (x,y)

def make_fancy_plot(colname, 
                    funk=lambda x:x, 
                    values=None, 
                    rotation=None, 
                    first=deaths, 
                    second=suicidals,
                    first_title="All Deaths",
                    second_title="Suicides"):
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    f.suptitle(re.sub("([a-z])([A-Z])","\g<1> \g<2>", colname))
    if values is None:
        values = unique_list(list(first[colname]))
        
    x = [funk(a) for a in values]
    
    
    
    l = list(first[colname])
    c = Counter(l)
    y1 = [c[a] for a in values]
    x1, y1 = unique_two_lists(x, y1)
    x1 = np.array(x1)
    
    sns.barplot(x1, y1, palette="BuGn_d", ax=ax1)
    ax1.set_ylabel(first_title)
    
    l = list(second[colname])
    c = Counter(l)
    y2 = [c[a] for a in values] 
    x2, y2 = unique_two_lists(x, y2)
    x2 = np.array(x2)
    
    g = sns.barplot(x2, y2, palette="BuGn_d", ax=ax2)
    ax2.set_ylabel(second_title)
    
    
    if rotation is not None:
        plt.xticks(rotation=rotation)
    sns.despine(bottom=True)
def make_fancier_plot(colname, 
                      datasets,
                      funk=lambda x:x, 
                      values=None, 
                      rotation=None, 
                      titles=None):
    
    f, tup = plt.subplots(len(datasets), 1, figsize=(8, 6), sharex=True)
    f.suptitle(re.sub("([a-z])([A-Z])","\g<1> \g<2>", colname))
    if values is None:
        values = unique_list(list(datasets[0][colname]))
        
    x = [funk(a) for a in values]
    
    for dataset, title, axx in zip(datasets, titles, tup):
        l = list(dataset[colname])
        c = Counter(l)
        y1 = [c[a] for a in values]
        x1, y1 = unique_two_lists(x, y1)
        x1 = np.array(x1)
        sns.barplot(x1, y1, palette="BuGn_d", ax=axx)
        axx.set_ylabel(title)
    
    if rotation is not None:
        plt.xticks(rotation=rotation)
    sns.despine(bottom=True)
f = lambda x: "Other" if x in ['R', 'E', 'D'] else list(disp.loc[disp["Code"] == x]["Description"])[0]
make_fancier_plot(colname="MethodOfDisposition", 
                  funk=f, 
                  rotation=45, 
                  datasets=[deaths, suicidals], 
                  titles=["All deaths", "suicides"])
f = lambda x: "Other" if x in ['R', 'E', 'D'] else list(disp.loc[disp["Code"] == x]["Description"])[0]
make_fancy_plot("MethodOfDisposition", funk=f, rotation=45)
strings = ["gun", "firearm"]

x = list(icd10["Code"])
y = list(icd10["Description"])
t = [a for a, b in zip(x, y) if b.lower().find("gun") != -1 or b.lower().find("firearm") != -1]
t = t[1:] #Chikungunya virus disease
x = deaths["Icd10Code"]
new_x = [True if a in t else False for a in x]
deaths["GUNS"] = new_x
deaths_with_guns = deaths.loc[deaths["GUNS"] == True]
f = lambda x: "Asian" if x in [7, 6, 28, 38, 48, 58, 68, 78, 4, 5, 8] else list(race.loc[race["Code"] == x]["Description"])[0]

make_fancy_plot("Race", 
                values=list(race["Code"]), 
                funk=f,
                rotation=80,
                first=deaths,
                second=deaths_with_guns,
                second_title="Deaths with guns")
top10 = deaths_with_guns[['Icd10Code', 'Id']].where(deaths_with_guns["Race"]==1).groupby(['Icd10Code']).count().sort_values(['Id'], ascending=False).head(10)
top10 = pd.merge(top10, icd10, left_index=True, right_on=['Code'])
top10.plot(kind='bar', x='Description')
top10 = deaths_with_guns[['Icd10Code', 'Id']].where(deaths_with_guns["Race"]==2).groupby(['Icd10Code']).count().sort_values(['Id'], ascending=False).head(10)
top10 = pd.merge(top10, icd10, left_index=True, right_on=['Code'])
top10.plot(kind='bar', x='Description')
f = lambda x: list(loc[loc["Code"] == x]["Description"])[0]

make_fancy_plot("PlaceOfDeathAndDecedentsStatus", 
                values=list(loc["Code"]), 
                funk=f,
                rotation=89,
                first=deaths,
                second=deaths_with_guns,
                second_title="Deaths with guns")
f = lambda x: list(pla[pla["Code"] == x]["Description"])[0]

make_fancier_plot(colname="PlaceOfInjury", 
                  values=range(10), 
                  funk=f, 
                  rotation=90, 
                  datasets=[deaths.loc[deaths["PlaceOfInjury"] != 99], 
                            suicidals.loc[suicidals["PlaceOfInjury"] != 99], 
                            deaths_with_guns.loc[deaths_with_guns["PlaceOfInjury"] != 99]], 
                  titles=["All deaths", "Suicides", "Deaths with guns"])
deaths