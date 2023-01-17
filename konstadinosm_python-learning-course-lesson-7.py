import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
fao_db = pd.read_csv("../input/FAO.csv", encoding='latin1') # read csv file from directory and input data into a dataframe
fao_db.head(3) # return the first 3 entries to ensure file was imported
fao_db.shape # get information about the dataframe shape
fao_db.index # get information regarding the indexing of the dataframe
fao_db.columns # get information about the dataframe columns
fao_db.info() # get information about the value types each columns contains
fao_db.count() # get information about the number of data in each column - notice different amounts in the years' columns
               # indicates countries leaving or entering production. also indicates new countries or merging of countries.
fao_db[['Area', 'Y2000']].head(5) # get only the Area and Year 2000 columns
new_db = fao_db[['Area', 'Y2000']].copy()
new_db.head(5)
fao_db.iloc[:,2:].head(7) # get all columns after Area column
fao_db.iloc[4:7,4:8]
fao_db[fao_db.columns[10:]].head(5) # get only Years
fao_db[['Area']].head(5) # get only the Area column
fao_db['Area'].head(5)
fao_db['Area'].unique() # get unique countries
len(fao_db['Area'].unique()) # get number of unique countries
fao_db['Item'].unique()
len(fao_db['Item'].unique())
fao_db.loc[fao_db['Area'] == 'Greece']
fao_db.loc[fao_db['Area'] == 'Greece'].describe()
fao_db.loc[fao_db['Area'] == 'Greece'].Y1961.describe()
fao_db.Y1961
fao_db.Y1961.describe()