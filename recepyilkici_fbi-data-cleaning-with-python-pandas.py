import pandas as pd
import numpy as np
import datetime as dt
fbi = pd.read_excel('../input/fbi-crime-in-the-united-states/fbi.xlsx')
fbi.head()
pd.set_option('display.max.columns', None)
fbi.head()
# The first 2 rows and the last 8 rows are trash, clean it!
fbi.drop(fbi.index[0:2], inplace=True)
fbi.drop(fbi.index[-11:], inplace=True)
fbi.reset_index(inplace=True, drop=True)
fbi.head()
fbi.iloc[0] # column names
cols = []

for i in range(0, 4):
    cols.append(fbi.iloc[0][i])

for i in range(4,23):
    if i%2 == 1:
        cols.append(fbi.iloc[0][i])
    
cols
# Update column names

fbi.columns = fbi.iloc[0]
fbi.drop(fbi.index[0:2], inplace=True)
fbi.reset_index(inplace=True, drop=True)
fbi.head()
# Remove Rate Per columns

cols = fbi.columns[fbi.columns.notnull()]
fbi = fbi[cols].copy()
fbi.head()
# Remove Percent Change rows

for index, row in fbi.iterrows():
    if row['Year'] == 'Percent change':
        fbi.drop(index, inplace=True)
fbi.reset_index(inplace=True, drop=True)
fbi.head()
# The first 2 rows are trash, clean it!

fbi.drop(fbi.index[0:2], inplace=True)
fbi.reset_index(inplace=True, drop=True)
fbi.head()
# Fill the blanks in Area column

for i in range(fbi.shape[0]):
    if str(fbi['Area'][i]) == 'nan':
        fbi['Area'][i] = fbi['Area'][i-1]

fbi.head(10)
# Look at this here! If any row has '5' space, then the row must be State!

for i in fbi['Area'].unique():
    print(i)
# Clear unnecessary numbers of States

for i in range(fbi.shape[0]):
    # Specific solution for South and South Atlantic
    if fbi['Area'][i] == 'South6, 7 ,8':
        fbi['Area'][i] = 'South'
    if fbi['Area'][i] == 'South Atlantic6, 7,8':
        fbi['Area'][i] = 'South Atlantic'
    for j in fbi['Area'][i]:
        try:
            a = int(j)
        except:
            a = None
        
        if isinstance(a, int):
            fbi['Area'][i] = fbi['Area'][i].replace(str(a)+', ', '')
            fbi['Area'][i] = fbi['Area'][i].replace(str(a), '')
            
fbi.head(10)
# Clear unnecessary numbers of Column names

cols = fbi.columns
new_cols = []

for i in cols:
    for j in i:
        try:
            a = int(j)

        except:
            a = None

    if isinstance(a, int):
        b = i.replace(j, '')
    
    else:
        b = i
    
    new_cols.append(b)
    
fbi.columns = new_cols
fbi.head()
# Split data by regions

north_east = fbi.loc[fbi['Area']=='Northeast'].index.values.astype(int)[0]
mid_west = fbi.loc[fbi['Area']=='Midwest'].index.values.astype(int)[0]
south = fbi.loc[fbi['Area']=='South'].index.values.astype(int)[0]
west = fbi.loc[fbi['Area']=='West'].index.values.astype(int)[0]

fbi_north = fbi[north_east:mid_west].copy()
fbi_mid = fbi[mid_west:south].copy()
fbi_south = fbi[south:west].copy()
fbi_west = fbi[west:].copy()
# Add 'Region' columns

fbi_north['Region'] = 'Northeast'
fbi_mid['Region'] = 'Midwest'
fbi_south['Region'] = 'South'
fbi_west['Region'] = 'West'
# Take the Region data to another dataframe. Maybe one day it will be needed.

fbi_region = pd.concat([fbi_north[0:2], fbi_mid[0:2], fbi_south[0:2], fbi_west[0:2]])
fbi_region.reset_index(inplace=True, drop=True)
fbi_region
# Clear first 2 rows of split data
fbi_north.drop(fbi_north.index[0:2], inplace=True)
fbi_north.reset_index(inplace=True, drop=True)
fbi_mid.drop(fbi_mid.index[0:2], inplace=True)
fbi_mid.reset_index(inplace=True, drop=True)
fbi_south.drop(fbi_south.index[0:2], inplace=True)
fbi_south.reset_index(inplace=True, drop=True)
fbi_west.drop(fbi_west.index[0:2], inplace=True)
fbi_west.reset_index(inplace=True, drop=True)
# I told that, If any row has '5' space, then the row must be State. So, right here, I detect the states with this information. 

# I could do this work from another way, like enter the states manually, but I chose this way.

fbi_north['Geographic_Division'] = None
for i in range(fbi_north.shape[0]):
    if fbi_north['Area'][i][0:5] != '     ':
        fbi_north['Geographic_Division'][i] = fbi_north['Area'][i]
    
    else:
        fbi_north['Geographic_Division'][i] = fbi_north['Geographic_Division'][i-1]

# ----------------------------------------------------------------------------------
        
fbi_mid['Geographic_Division'] = None
for i in range(fbi_mid.shape[0]):
    if fbi_mid['Area'][i][0:5] != '     ':
        fbi_mid['Geographic_Division'][i] = fbi_mid['Area'][i]
    
    else:
        fbi_mid['Geographic_Division'][i] = fbi_mid['Geographic_Division'][i-1]

# ----------------------------------------------------------------------------------

fbi_south['Geographic_Division'] = None
for i in range(fbi_south.shape[0]):
    if fbi_south['Area'][i][0:5] != '     ':
        fbi_south['Geographic_Division'][i] = fbi_south['Area'][i]
    
    else:
        fbi_south['Geographic_Division'][i] = fbi_south['Geographic_Division'][i-1]

# ----------------------------------------------------------------------------------

fbi_west['Geographic_Division'] = None
for i in range(fbi_west.shape[0]):
    if fbi_west['Area'][i][0:5] != '     ':
        fbi_west['Geographic_Division'][i] = fbi_west['Area'][i]
    
    else:
        fbi_west['Geographic_Division'][i] = fbi_west['Geographic_Division'][i-1]
# Sample

fbi_west.head()
# Merge the split data

fbi = pd.concat([fbi_north, fbi_mid, fbi_south, fbi_west], ignore_index=True)

fbi['State'] = fbi['Area'].copy()
fbi.drop('Area', axis=1, inplace=True)
fbi.head()
# Change column order for visual pleasure lol

cols = ['Region', 'Geographic_Division', 'State']
cols += list(fbi.columns)
del cols[-3:]
cols
fbi = fbi[cols]
fbi.head()
# Now I can delete '5' spaces of states :(

for i in range(fbi.shape[0]):
    if fbi['State'][i][0:5] == "     ":
        fbi['State'][i] = fbi['State'][i].replace("     ", "")
# Take the Division data to another dataframe. Maybe one day it will be needed like Region data.

fbi_division = fbi.loc[fbi['Geographic_Division'] == fbi['State']]
# Drop divisions from main dataframe

fbi.drop(fbi.loc[fbi['Geographic_Division'] == fbi['State']].index, inplace=True)
fbi.reset_index(inplace=True, drop=True)
# Update column names again

cols = list(fbi.columns)

for i in range(len(cols)):
    cols[i] = cols[i].replace("\n", "")

fbi.columns = cols
# Ta da!

fbi.head()