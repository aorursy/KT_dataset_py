import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import patsy
# Load our data.

data = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1')
# These are the columns that seem of interest and contain sufficient data to be valuable

# for exploratory and possibly predictive values.



data_columns = [

    

    ## Spatio-Temporal Variables:

                'iyear', 'imonth', 'iday', 'latitude', 'longitude',

    

    ## Binary Variables: 

                'extended', 'vicinity', 'crit1', 'crit2', 'crit3', 'doubtterr',

                'multiple', 'success', 'suicide', 'guncertain1', ## check back guncertain

                'claimed', 'property', 'ishostkid',

    

    ## Continuous Variables:

                'nkill', 'nwound',               

    

    ## Categorical variables (textual): 

                'country_txt', 'region_txt', 'alternative_txt', 'attacktype1_txt', 'targtype1_txt',

                'natlty1_txt', 'weaptype1_txt', 

    

    ## Descriptive Variables: 

                'target1', 'gname', 'summary',    

    

                                            ]



data = data.loc[:, data_columns] # Only keep described columns.



# Random acts of violence and other outliers should not be part of the data.

# Thus, restrict the set the only attacks where the terrorism motive is certain.

data = data[(data.crit1 == 1) & (data.crit2 == 1) & (data.crit3 == 1) & (data.doubtterr == 0)]



# Weapontype column contains very long name for vehicle property -> shorten.

data.weaptype1_txt.replace(

    'Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)',

    'Vehicle', inplace = True)



# Replace -9 (unknown) values with 0 (no). -9 values are much more likely to be false than true.

data.iloc[:,[6, 15, 16, 17]] = data.iloc[:,[6, 15, 16, 17]].replace(-9,0)



# Some values in the claimed category are 2 (should be 0 or 1).

# Assume these were input mistakes and set 2 to 1.

data.claimed.replace(2,1, inplace = True)



# Ensure consistent values and make everything lowercase.

data.target1 = data.target1.str.lower()

data.gname = data.gname.str.lower()

data.summary = data.summary.str.lower()    

data.target1 = data.target1.fillna('unknown').replace('unk','unknown')



# Some nwound and nkill are NaN. Replace them with median.

data.nkill = np.round(data.nkill.fillna(data.nkill.median())).astype(int) 

data.nwound = np.round(data.nwound.fillna(data.nwound.median())).astype(int) 



# Database only reports victims as nkill and nwound. Combine these into ncasualties column.

# Also add has_casualties column.

data['ncasualties'] = data['nkill'] + data['nwound']

data['has_casualties'] = data['ncasualties'].apply(lambda x: 0 if x == 0 else 1)



print("Data prepared.")
# Missing data in these columns is acceptable:

data_to_drop = ['latitude','longitude','summary', 'alternative_txt', 'natlty1_txt']

missing_data = data.drop(data_to_drop, axis = 1)

print(missing_data.isnull().sum().sort_values(ascending = False).head(4))
# Set NaN to 0...

data.guncertain1.fillna(0, inplace = True)

data.ishostkid.fillna(0, inplace = True)



# Drop columns that are not of use for prediction.

data = data.drop(['latitude','longitude','summary'], axis =1)



data.shape
y_temp = data.claimed

y_temp.shape
# Features that will be used for modeling. Split into text and numerical



categorical = ['country_txt', 'alternative_txt', 'attacktype1_txt',

               'targtype1_txt', 'weaptype1_txt', 'gname', 'target1']



numerical = ['extended', 'vicinity', 'multiple', 'success',

             'suicide', 'guncertain1', 'ncasualties', 'property', 'ishostkid']
formula =  ' + '.join(numerical)+ ' + ' + ' + '.join(['C('+i+')' for i in categorical]) + ' -1' 

formula
X_temp = patsy.dmatrix(formula, data = data, return_type= 'dataframe')

print(X_temp.shape, y_temp.shape)