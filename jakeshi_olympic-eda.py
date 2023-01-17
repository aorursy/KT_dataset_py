# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
PATH = '../input/athlete_events.csv'
data = pd.read_csv(PATH)
data.head()
male_youngest=data[(data['Sex']=='M')& (data['Year']==1996)]['Age'].min()
female_youngest=data[(data['Sex']=='F')& (data['Year']==1996)]['Age'].min()
int(male_youngest),int(female_youngest)
male_2000 = data.query('Sex =="M" & Year==2000')
male_2000_gym = male_2000.query('Sex =="M" & Year==2000 & Sport=="Gymnastics"')
male_2000.Name.unique().size
male_2000_by_sport = male_2000.groupby('Sport')['Name'].nunique()
male_2000_by_sport
male_2000_by_sport.describe()
97/male_2000_by_sport.sum()
female_basketball_2000=data.query('Sex =="F" & Year==2000 & Sport=="Basketball"').drop_duplicates()
female_basketball_2000.Height.describe()
data.query('Year==2002').Weight.max()
data.query('Year==2002 & Weight==123')
data.query('Name == "Pawe Abratkiewicz"').Year.unique()
data.query('Year == 2000 & Team=="Australia" & Sport=="Tennis"')
# data.query('Year == 2000 & Team=="Australia" & Sport=="Tennis" & Medal.isnan()')

x = data.query('Year == 2000 & Team=="Australia" & Sport=="Tennis"').loc[data['Medal'].notnull(), 
                                                                        ['Sport', 'Event','Medal' ]]
x
x = data.query('Year == 2016 & Team=="Serbia"').loc[data['Medal'].notnull(), 
                                                             ['Sport', 'Event','Medal' ]].drop_duplicates()
x
x = data.query('Year == 2016 & Team=="Switzerland"').loc[data['Medal'].notnull(), 
                                                             ['Sport', 'Event','Medal' ]].drop_duplicates()
x
x = data.query('Year == 2014')[['Name','Age']].drop_duplicates()['Age']
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
plt.hist(x,range=(5,105))
data.query('City == "Lake Placid"').Games.drop_duplicates()
data.query('City == "Sankt Moritz"').Games.drop_duplicates()
data.query('Year == 2016').Sport.drop_duplicates().size
data.query('Year == 1995').Sport.drop_duplicates().size
