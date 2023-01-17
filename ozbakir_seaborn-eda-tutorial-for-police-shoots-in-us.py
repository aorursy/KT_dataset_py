# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
household_income = pd.read_csv('../input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv', encoding='windows-1252')

povertyRate = pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv', encoding='windows-1252')

highSchoolCompRate_for25old = pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv', encoding='windows-1252')

raceDistByCity = pd.read_csv('../input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv', encoding='windows-1252')

kill = pd.read_csv('../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', encoding='windows-1252')
povertyRate.head()
# is_NaN = povertyRate.isnull()

# row_has_NaN = is_NaN.any(axis=1)

# povertyRate[row_has_NaN]
povertyRate.info()
# povertyRate[povertyRate.poverty_rate == "-"]

povertyRate.poverty_rate.value_counts()
# povertyRate.poverty_rate = povertyRate.poverty_rate.astype(float)

# povertyRate.info()

unknownValues = []

indis = povertyRate.poverty_rate.index.values

values = povertyRate.poverty_rate.values

for i,j in zip(indis, values):

    try:

        float(j)

    except:

        unknownValues.append([i,j])

print(len(unknownValues), 'unknown values', unknownValues)
# povertyRate.poverty_rate = [i.replace('-','0') for i in povertyRate.poverty_rate]

unknownValues = []

index = povertyRate.poverty_rate.index.values

values = povertyRate.poverty_rate.values

for i,j in zip(index, values):

    try:

        float(j) # burada ilgili veriyi float yapmaya çalışıyoruz. Yapamaz ise uyumsuz veridir. Dolayısıyla listeye ekliyoruz.

    except:

        unknownValues.append([i,j])

unknownValues = [int(i[0]) for i in unknownValues] # only index numbers

# print(unknownValues)

# povertyRate.poverty_rate.replace('-',0,inplace=True)

povertyRate.drop(unknownValues, inplace=True)

povertyRate.poverty_rate = povertyRate.poverty_rate.astype('float')

povertyRate.poverty_rate

areaList = povertyRate['Geographic Area'].unique() # numpy object



area_pov_ratio = []

for i in areaList:

    x = povertyRate[povertyRate['Geographic Area'] == i]

    area_pov_ratio.append(x.poverty_rate.sum()/len(x))

#     area_pov_ratio.append(sum(x.poverty_rate)/len(x))



data = pd.DataFrame({'areaList': areaList, 'areaPovRatio':area_pov_ratio})

data.sort_values(by='areaPovRatio', ascending=False, inplace=True)

data.reset_index(drop=True, inplace=True) # dataFrame of 'areaList' and 'areaPovRatio'



plt.figure(figsize=(15,10))

sns.barplot(data.areaList,data.areaPovRatio)

plt.xticks(rotation=45)

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.show()
kill.name.value_counts()
separate = kill.name[kill.name != 'TK TK'].str.split()

separate2 = pd.Series([i.split() for i in kill.name[kill.name != 'TK TK']]) # alternatif

print(type(separate) == type(separate2))



name = []

surname = []

for i in separate2:

    name.append(i[0])

    surname.append(i[1])



liste = name + surname

nameCount = Counter(liste)

mostCommonName = nameCount.most_common(15)

x,y = zip(*mostCommonName)

x,y = list(x), list(y)



plt.figure(figsize=(15,10))

sns.barplot(x=list(x),y=list(y), palette=sns.cubehelix_palette(len(x),reverse=True))

plt.xlabel('Name or surname of killed people')

plt.xticks(rotation=45)

plt.ylabel('Frequency')

plt.show()
# High school graduation rate

highSchoolCompRate_for25old.percent_completed_hs.value_counts()
highSchoolCompRate_for25old.info()
# highSchoolCompRate_for25old.percent_completed_hs.replace('-','0.0', inplace=True)

# highSchoolCompRate_for25old.percent_completed_hs = highSchoolCompRate_for25old.percent_completed_hs.astype(float)



# areaList = highSchoolCompRate_for25old['Geographic Area'].unique()

# areaHighSchoolRatio = []

# for i in areaList:

#     x = highSchoolCompRate_for25old[highSchoolCompRate_for25old['Geographic Area'] == i]

#     areaHighSchoolRatio.append(x.percent_completed_hs.sum()/len(x))



# data2 = pd.DataFrame({'Area':areaList, 'HighSchoolRatio':areaHighSchoolRatio}) 

# data2.sort_values(by='HighSchoolRatio', ascending=False, inplace=True)

# data2.reset_index(drop=True, inplace=True)

# # data2.plot(kind='bar')



# plt.figure(figsize=(15,10))

# sns.barplot(data2.Area,data2.HighSchoolRatio)

# plt.xticks(rotation=45)

# plt.xlabel('States')

# plt.ylabel('Frequency')

# plt.title('High school graduation rate of over 25 old')

# plt.show()



# highSchoolCompRate_for25old.percent_completed_hs.replace('-','0.0', inplace=True)

unknownValues = []

index = highSchoolCompRate_for25old.percent_completed_hs.index.values

values = highSchoolCompRate_for25old.percent_completed_hs.values

for i,j in zip(index, values):

    try:

        float(j) # burada ilgili veriyi float yapmaya çalışıyoruz. Yapamaz ise uyumsuz veridir. Dolayısıyla listeye ekliyoruz.

    except:

        unknownValues.append([i,j])

unknownValues = [int(indis[0]) for indis in unknownValues] # only index numbers

# print(unknownValues)

# povertyRate.poverty_rate.replace('-',0,inplace=True)

highSchoolCompRate_for25old.drop(unknownValues, inplace=True)

highSchoolCompRate_for25old.percent_completed_hs = highSchoolCompRate_for25old.percent_completed_hs.astype(float)



areaList = highSchoolCompRate_for25old['Geographic Area'].unique()

areaHighSchoolRatio = []

for i in areaList:

    x = highSchoolCompRate_for25old[highSchoolCompRate_for25old['Geographic Area'] == i]

    areaHighSchoolRatio.append(x.percent_completed_hs.sum()/len(x))



data2 = pd.DataFrame({'Area':areaList, 'HighSchoolRatio':areaHighSchoolRatio}) # dataFrame of 'areaList' and 'HighSchoolRatio'

data2.sort_values(by='HighSchoolRatio', ascending=False, inplace=True)

data2.reset_index(drop=True, inplace=True)

# data2.plot(kind='bar')



plt.figure(figsize=(15,10))

sns.barplot(data2.Area,data2.HighSchoolRatio)

plt.xticks(rotation=45)

plt.xlabel('States')

plt.ylabel('Frequency')

plt.title('High school graduation rate of over 25 old')

plt.show()
highSchoolCompRate_for25old.info()
raceDistByCity.head()
raceDistByCity.info()
# def filterValues(dataframe):

#     cols = dataframe.columns.values



# #     for col in ['', None, '11.5', '-', 'XL', '1', 1, 1.5]:

#     for col in cols:

#         index = list(dataframe[col].index.values)

#         values = list(dataframe[col].values)

    

#         compatibleValues = []

#         incompatibleValues = []

        

#         for i,v in zip(index, values):

            

#             try:

#                 int(v) and float(v)

#                 compatibleValues.append(v)

#             except:

                

#                 if v == None or v == '' or type(v) != int and type(v) != float and type(v) == str and len(list(v)) <= 1 and not v.isalpha():

# #                     print(type(v))

#                     incompatibleValues.append(v)

# #                     print(incompatibleValues)

#                 else:



#                     try:

#                         float(v)

#                         compatibleValues.append(v)



#                     except:

#                         compatibleValues.append(v)



# #     print(compatibleValues)

#         print(incompatibleValues)
# deneme

def filterStr_or_FloatCols(pandasSeries):

    "return: index of undefined values"

    index = pandasSeries.index.values

    values = pandasSeries.values

    

    known_index_values = []

    unknown_index_values = []

    unknown_index = []

    

    for i,v in zip(index, values):

        try:

            float(v)

            known_index_values.append([i,v])

                        

        except:

            if len(v) > 5:

                

                known_index_values.append([i,v])



            else:

                unknown_index_values.append([i,v])

                unknown_index.append(int(i))

    

    return np.array(unknown_index)



#     print(len(unknown_index_values), "incompatible values", unknown_index_values)
# filterStr_or_FloatCols(raceDistByCity['Geographic area'])

# filterStr_or_FloatCols(raceDistByCity['City'])

print(filterStr_or_FloatCols(raceDistByCity['share_white']))

# filterStr_or_FloatCols(raceDistByCity['share_black'])

# filterStr_or_FloatCols(raceDistByCity['share_native_american'])

# filterStr_or_FloatCols(raceDistByCity['share_asian'])

# filterStr_or_FloatCols(raceDistByCity['share_hispanic'])

raceDistByCity.info()
undefined_index = filterStr_or_FloatCols(raceDistByCity['share_white'])

raceDistByCity.drop(undefined_index, axis=0, inplace=True)

raceDistByCity.iloc[:,2:] = raceDistByCity.iloc[:,2:].astype(float) # doing one line conversion instead of the following codes

# raceDistByCity.share_white = raceDistByCity.share_white.astype(float)

# raceDistByCity.share_black = raceDistByCity.share_black.astype(float)

# raceDistByCity.share_native_american = raceDistByCity.share_native_american.astype(float)

# raceDistByCity.share_asian = raceDistByCity.share_asian.astype(float)

# raceDistByCity.share_hispanic = raceDistByCity.share_hispanic.astype(float)

raceDistByCity.info()
area_list = raceDistByCity['Geographic area'].unique()

white_ratio = []

black_ratio = []

native_american_ratio = []

asian_ratio = []

hispan_ratio = []

ratio_lists = ['white_ratio', 'black_ratio', 'native_american_ratio', 'asian_ratio', 'hispan_ratio']

list_colors = ['green', 'blue', 'cyan', 'yellow', 'red']

for i in area_list:

    x = raceDistByCity[raceDistByCity['Geographic area'] == i]

    white_ratio.append(x.share_white.sum() / len(x))

    black_ratio.append(x.share_black.sum() / len(x))

    native_american_ratio.append(x.share_native_american.sum() / len(x))

    asian_ratio.append(x.share_asian.sum() / len(x))

    hispan_ratio.append(x.share_hispanic.sum() / len(x))



# Visualization



f,ax = plt.subplots(figsize=(15,20))

for l, color in zip(ratio_lists, list_colors):

#     print(eval(l))

    sns.barplot(eval(l), area_list, color=color, alpha=0.4, label=l[:-6]) # HERE, I used eval() fuction. Becaue, i want to use list names in labels



    ax.legend(loc='lower right', frameon=True)

ax.set(xlabel='Percentage of Races', ylabel='States', title="Percentage of State's population")

plt.show()
data.head()
data2.head()
data.sort_values(by='areaList', inplace=True)

data.reset_index(drop=True, inplace=True)

data2.sort_values(by='Area', inplace=True)

data2.reset_index(drop=True, inplace=True)
data.areaPovRatio = data.areaPovRatio / data.areaPovRatio.max()

data2.HighSchoolRatio = data2.HighSchoolRatio / data2.HighSchoolRatio.max()

dataConcat = pd.concat([data,data2.HighSchoolRatio], axis=1)

dataConcat.sort_values(by='areaPovRatio', inplace=True)

dataConcat.reset_index(drop=True, inplace=True)

dataConcat.head()


# Visualize

f, ax = plt.subplots(figsize=(20,10))

# sns.pointplot(x='areaList', y='areaPovRatio', data=dataConcat, color='lime', alpha=0.8)

# sns.pointplot(x='areaList', y='HighSchoolRatio', data=dataConcat, color='red', alpha=0.8)

plt.plot_date(x=dataConcat.areaList, y=dataConcat.areaPovRatio, color='lime', alpha=0.8, ls='-', label="High School Graduate Ratio")

plt.plot_date(x=dataConcat.areaList, y=dataConcat.HighSchoolRatio, color='red', alpha=0.8, ls='-', label="Poverty Ratio")



plt.legend(frameon=True, fontsize=15)

plt.xlabel('States', fontsize=15, color='b')

plt.ylabel('Values', fontsize=15, color='b')

plt.title('High School Graduate VS Poverty Rate', fontsize=20, color='b')

plt.grid()

plt.show()

dataConcat.head()
data.head()
data2.head()
dataConcat.head()
# Joint Plot

# import scipy.stats as stats

# g = sns.jointplot(dataConcat.areaPovRatio, dataConcat.HighSchoolRatio, height=7, kind='kde').annotate(stats.pearsonr)

# g = sns.jointplot(dataConcat.areaPovRatio, dataConcat.HighSchoolRatio, height=7, kind='kde', stat_func=stats.pearsonr)

g = sns.jointplot(dataConcat.areaPovRatio, dataConcat.HighSchoolRatio, height=7, kind='kde', ratio=5) # ratio=5 is default



plt.show()
g = sns.jointplot(dataConcat.areaPovRatio, dataConcat.HighSchoolRatio, height=7, kind='scatter', ratio=5) # ratio=5 is default



plt.show()
# Pie Plot

kill.race.value_counts(dropna=False)
kill.race.dropna(inplace=True)

kill.race.value_counts(dropna=False)
kill[kill.isnull().any(axis=1)] # shows the nan value rows

index_of_nans = kill[kill.isnull().any(axis=1)].index.values # shows their index

# kill.columns[kill.isnull().any()]

kill.drop(index_of_nans, inplace=True)
kill.race.unique()

# kill.race.value_counts().index.values

# labels = kill.race.value_counts().index.values

labels = kill.race.unique()



colors = ['grey','blue','red','yellow','green','brown']

explode = [0.02,0.02,0.02,0.02,0.02,0.02] # to split pie slices

sizes = kill.race.value_counts().values



# Visualize

plt.figure(figsize=(7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Killed People According to Race', color='b', fontsize=15)

plt.show()
sns.lmplot(x='areaPovRatio', y='HighSchoolRatio', data=dataConcat)

# sns.regplot(x='areaPovRatio', y='HighSchoolRatio', data=dataConcat)



plt.show()
sns.kdeplot(dataConcat.areaPovRatio, dataConcat.HighSchoolRatio, shade=True, cut=3)

plt.show()
palette = sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data=dataConcat, palette=palette, inner='points')
kill.head()
sns.boxplot(x='gender', y='age', hue='manner_of_death', data=kill, palette='PRGn')
sns.swarmplot(x='gender', y='age', hue='manner_of_death', data=kill)
sns.pairplot(dataConcat)
armed = kill.armed.value_counts()



plt.figure(figsize=(10,7))

sns.barplot(x=armed[:7].index,y=armed[:7].values)
# plt.figure(figsize=(10,7))



kill['age_limit'] = ['above25' if i >= 25 else 'below25' for i in kill.age]

sns.countplot(kill.age_limit)
sns.countplot(data=kill, x='age_limit')
city = kill.city.value_counts()

plt.figure(figsize=(15,7))

sns.barplot(city[:10].index, city[:10].values)

plt.title('Most Dangerous Cities')