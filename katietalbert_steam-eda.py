# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.chdir('/kaggle/input/buildingdatagenomeproject2')

os.listdir()
steam = pd.read_csv('steam_cleaned.csv')

print(steam.shape)

steam.head()
steam.columns
steam.dtypes
steam.describe()
steam.info()
print(steam.isnull().sum().sort_values(ascending=False).head(20))
# dropping completely null columns

drop_cols = ['Cockatoo_lodging_Eric', 'Cockatoo_industrial_Nathaniel', 'Cockatoo_education_Sheryl',

             'Cockatoo_education_Gussie', 'Cockatoo_office_Paige', 'Cockatoo_education_Mayra', 'Cockatoo_lodging_Cletus',

             'Cockatoo_lodging_Albert', 'Cockatoo_lodging_Aimee', 'Cockatoo_lodging_Ana', 'Cockatoo_lodging_Elvia',

             'Cockatoo_education_Joel', 'Cockatoo_office_Roxanna', 'Cockatoo_religion_Diedre', 'Peacock_office_Naomi',

             'Peacock_office_Dara', 'Cockatoo_lodging_Tessie', 'Cockatoo_public_Leah', 'Cockatoo_education_Maynard']

steam = steam.drop(drop_cols, axis=1)

steam.shape
columns = steam.columns.tolist()

zeros = steam.copy()

zeros = zeros.replace(0, np.nan)

drop = [];

ii = 1;

while ii<len(columns):

    if zeros[columns[ii]].isnull().sum() == 17544:

        drop.append(columns[ii])

    ii = ii + 1

    

drop
steam = steam.drop(drop, axis = 1)
# changing timestamp to datetime

steam['timestamp'] = steam['timestamp'].astype('datetime64')

steam.dtypes
import missingno as msno

msno.matrix(steam);
#looks like the data can be broken down by site id

peacock = steam.loc[:,'Peacock_lodging_Terrie':'Peacock_education_Robbie']

moose = steam.loc[:,'Moose_education_Florence':'Moose_education_Ricardo']

bull = steam.loc[:,'Bull_education_Magaret':'Bull_education_Luke']

hog = steam.loc[:,'Hog_other_Noma':'Hog_office_Denita']

eagle = steam.loc[:,'Eagle_office_Lamont':'Eagle_education_Shana']

cockatoo = steam.loc[:,'Cockatoo_public_Chiquita':'Cockatoo_public_Shad']
msno.matrix(peacock);
peacock = peacock.interpolate(method="slinear")

peacock.isnull().sum()
msno.matrix(peacock);
peacock.columns
drop_cols = ['Peacock_lodging_Francesca', 'Peacock_public_Kelvin', 'Peacock_lodging_Sergio', 'Peacock_lodging_Mathew',

             'Peacock_assembly_Dena', 'Peacock_education_Lucie', 'Peacock_lodging_Chloe',

             'Peacock_lodging_Wes']

steam = steam.drop(drop_cols, axis=1)

steam.shape
msno.matrix(moose);
msno.matrix(bull);
msno.matrix(hog);
msno.matrix(eagle);
msno.matrix(cockatoo);
cockatoo.isnull().sum().sort_values(ascending=False).head(10)
drop_cols = ['Cockatoo_lodging_Fritz', 'Cockatoo_education_Shawn', 'Cockatoo_education_Doreen', 'Cockatoo_office_Alton',

             'Cockatoo_lodging_Johnathan', 'Cockatoo_assembly_Griselda']

steam = steam.drop(drop_cols, axis=1)

steam.shape
steam.head(10)
steam = steam.interpolate(method="slinear")

steam.isnull().sum().sort_values(ascending=False).head(10)
msno.matrix(steam)
# drop columns with significant missing data (top 4 with most null values) because there's too much missing for bfill to be 

# accurate



drop_cols = ['Cockatoo_education_Julio', 'Peacock_assembly_Mamie', 'Cockatoo_lodging_Alicia', 'Cockatoo_education_Charity']

steam = steam.drop(drop_cols, axis = 1)
steam = steam.fillna(method='ffill')

steam.isnull().sum()
steam.isnull().sum().sort_values(ascending=False).head(10)
steam = steam.fillna(method='bfill')

steam.isnull().sum().sort_values(ascending=False).head(10)
steam.to_csv('/kaggle/working/steam_cleaned2.csv')