# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

# load 2016 data

pssa_df = pd.read_excel('../input/2016 PSSA School Level Perfomance Results (1).xlsx', header=5)

print(pssa_df.columns)

# "Growth" column is missing compared to 2015
pssa_2015 = pd.read_excel('../input/2015 PSSA School Level Data.xlsx', header=6)

print(pssa_2015.columns)

# "County" is missing. Fill that in from the 2016 data

missing_counties = {109426003:'MCKEAN', 127040001:'BEAVER', 171510293:'PHILADELPHIA'}

counties = []

for i,x in enumerate(pssa_2015['AUN']):

    try:

        counties.append(pssa_df[pssa_df['AUN']==x].iloc[0]['County'])

    except IndexError:

        if x in missing_counties:

            counties.append(missing_counties[x])

        else:

            print('Missing County!')

            print('--------------')

            print(pssa_2015.iloc[i])

            counties.append('')

pssa_2015['County'] = counties

# Change "Growth**" to Growth

pssa_2015.rename(columns={'Growth**':'Growth'},inplace=True)

# combine

pssa_df = pssa_df.append(pssa_2015,ignore_index=True)
pssa_2014 = pd.read_excel('../input/School_Level_Assessment_File_SN_20132014_2015_0519_v03 (3).xlsx', header=6)

print(pssa_2014.columns)

# Schl = School Number, District Name = District, School Name = School, Student Group = Group, N Scored = Number Scored, Pct. Advanced = % Advanced, Pct. Proficient = % Proficient, Pct. Basic = % Basic, Pct. Below Basic = % Below Basic

col_names = {

    "Schl": "School Number", "District Name": "District", "School Name": "School",

    "Student Group": "Group", "N Scored": "Number Scored", "Pct. Advanced": "% Advanced",

    "Pct. Proficient": "% Proficient", "Pct. Basic": "% Basic", "Pct. Below Basic": "% Below Basic"

}

pssa_2014.rename(columns=col_names,inplace=True)

print(pssa_2014.columns)