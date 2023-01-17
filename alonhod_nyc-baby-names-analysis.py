# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load file and rename columns

names = pd.read_csv('/kaggle/input/nyc-baby-names/NYC Baby Name.csv')

names.rename(columns = {"Child's First Name" : 'Name', 'Year of Birth' : 'Year'}, inplace=True)
print(len(names))

print(len(names.drop_duplicates()))

names.head()
names.drop_duplicates(inplace=True)
# Rename columns

names.columns = ['Year','Gender','Ethnicity','Name','Count','Rank']

# Capitalization of letters

names['Name'] = names['Name'].str.title()

# Top 5 names

names.pivot_table(index='Name', values='Count', aggfunc='sum').sort_values('Count', ascending=False).head(5)
genders = names.groupby('Gender')
for gender, df in genders:

    print('\n', gender)

    print(df.pivot_table(index='Name', values='Count', aggfunc='sum').sort_values('Count', ascending=False).head(3))
GenderNames = pd.DataFrame()



for gender, df in genders:

    top3 = df.pivot_table(index = 'Name', values='Count', aggfunc='sum').sort_values('Count', ascending = False).head(3)

    top3['Gender'] = gender

    GenderNames = GenderNames.append(top3)

GenderNames.reset_index(inplace=True)

GenderNames = GenderNames[['Gender','Name','Count']]
GenderNames
# Option B - using pivot_table()

GenderDF = names.pivot_table(values='Count', index=['Gender','Name'],aggfunc='sum').sort_values(['Gender','Count'], ascending=False)

print(GenderDF.loc['MALE'].head(3))

print(GenderDF.loc['FEMALE'].head(3))
years = names.drop(columns='Ethnicity')

years = years.pivot_table(values=['Count'], index=['Name', 'Year'], aggfunc='sum')

years = years.groupby('Year')
YearNames = pd.DataFrame()

for year, data in years:

    YearNames = YearNames.append(years.get_group(year).sort_values('Count', ascending=False).head(1))



YearNames
yearspv = names.pivot_table(index=['Year','Name'], values='Count', aggfunc='sum')

for year in yearspv.index.get_level_values(0).unique():

    print(yearspv.loc[year].sort_values('Count', ascending=False).head(1))
YearGender = names.drop(columns='Ethnicity')

YearGender = YearGender.pivot_table(values=['Count'], index=['Gender', 'Year','Name'], aggfunc='sum')

for idx, df_select in YearGender.groupby(level=[0,1]):

        print(df_select.sort_values('Count', ascending=False).head(1))

GYNames = pd.DataFrame()

GYPivot = names.groupby(['Gender','Year'])

for gy, data in GYPivot:

    Popular = data.pivot_table(index='Name', values='Count', aggfunc=sum).sort_values('Count', ascending=False).head(1)

    Popular['Gender'] = gy[0]

    Popular['Year'] = gy[1]

    GYNames = GYNames.append(Popular)



GYNames
IsabellaDF = names.groupby('Name').get_group('Isabella')

IsabellaDF.head(3)
IsabellaByYear = pd.DataFrame(columns=['Year','Count'])

for year in IsabellaDF['Year'].drop_duplicates():

    mask = IsabellaDF['Year'] == year

    count = IsabellaDF[mask]['Count'].sum()

    IsabellaByYear = IsabellaByYear.append([{'Year':year, 'Count':count}])

    

IsabellaByYear.set_index('Year', inplace=True)
IsabellaByYear