import pandas as pd
import numpy as np
from calendar import monthrange
pd.set_option('display.max_columns', None)
path = "../input/globalterrorism-old.csv"
df = pd.read_csv(path, low_memory= False)
df.info(memory_usage='deep')
# =============================================================================
# SUMMARY - BEFORE
# dtypes: datetime64[ns](1), float64(55), int64(22), object(57)
# memory usage: 590.0 MB
# =============================================================================
df.head(2)
# Deleting unnecessary columns
cols =  ['eventid','approxdate', 'resolution', 'country','region', 'provstate', 'vicinity', 'multiple', 'alternative','attacktype1', 'attacktype2',
         'attacktype3','weaptype1','weapsubtype1','weaptype2','weapsubtype2', 'targtype1', 'targsubtype1','natlty1','targtype2','targsubtype2',
         'natlty2','targtype3', 'targsubtype3', 'natlty3','claimmode','propextent','hostkidoutcome','weaptype3','weaptype3_txt', 'weapsubtype3',
         'weapsubtype3_txt','weaptype4','weaptype4_txt','weapsubtype4','weapsubtype4_txt', 'guncertain1', 'guncertain2', 'guncertain3', 'gname3', 
         'propcomment', 'ishostkid', 'nhostkid', 'nhostkidus', 'nhours', 'ndays', 'divert', 'kidhijcountry', 'ransom', 'ransomamt', 'ransomamtus', 
         'ransompaid', 'ransompaidus', 'ransomnote', 'hostkidoutcome_txt', 'attacktype3_txt', 'targtype2_txt', 'targsubtype2_txt', 'corp2', 'target2', 
         'natlty2_txt', 'targtype3_txt', 'targsubtype3_txt', 'corp3', 'target3', 'natlty3_txt', 'claim2', 'claimmode2', 'claimmode2_txt', 'claim3', 
         'claimmode3', 'claimmode3_txt', 'compclaim', 'motive', 'gsubname', 'gname2', 'gsubname2', 'gsubname3', 'nperps', 'nperpcap', 'attacktype2_txt', 
         'weapsubtype1_txt', 'weaptype2_txt', 'weapsubtype2_txt', 'weapdetail', 'nkillus', 'nwoundus', 'nwoundte', 'nreleased', 'addnotes', 'scite1', 
         'scite2', 'scite3', 'dbsource', 'INT_LOG', 'INT_IDEO', 'INT_MISC', 'INT_ANY', 'related']

# Dropping columns function
def deleting_columns(col):
    df.drop(col, axis=1, inplace=True)

# Dropping columns in the cols list
for col in cols:
    try:
        deleting_columns(col)  
    except:
        print("Error in deleting columns:  {}".format(col))
# Calculating memory usage function
def mem_usage(pandas_obj):
    
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    
    return "{:03.2f} MB".format(usage_mb)
# Test type int columns
df_int = df.select_dtypes(include=['int'])
converted_int = df_int.apply(pd.to_numeric,downcast='unsigned')

print('Memory usage of columns type INT before optimization is: ', mem_usage(df_int)) #15.99 MB
print('Memory usage of columns type INT after optimization is: ', mem_usage(converted_int)) #3.29 MB

df_int = converted_int

# Re-run the cell - text below is not correct
# Test type float columns
df_float = df.select_dtypes(include=['float'])
converted_float = df_float.apply(pd.to_numeric,downcast='float')

print('Memory usage of columns type FLOAT before optimization is: ', mem_usage(df_float)) #37.69 MB
print('Memory usage of columns type FLOAT after optimization is: ', mem_usage(converted_float)) #18.85 MB

df_float = converted_float
# Optimize all object columns by converting them to type category if the unique values < 0.5
df_obj_col = df.select_dtypes(include=['object'])

for col in df_obj_col:
    try:
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5:
            df.loc[:,col] = df[col].astype('category')
        else:
            df.loc[:,col] = df[col]
    except:
        print("{} , Error converting object columns to type category".format(col))
# Check if null value is more than 90% in all columns
for col in df.columns:
    if df[col].isnull().sum() / len(df[col]) > .90:
        print("Columns more than 90% :  {}".format(col))
# Check if the -9 value is more than 90% in INT columns
x = df.select_dtypes(include=['int'])

for col in x:
    if len(df.loc[df[col] == -9, col]) / len(df[col]) > .90:
        print("Columns with -9 more than 90% :  {}".format(col))
len(df.loc[df['location'] == -9, 'location']) / len(df['location'])
for col in df.columns:
    if df[col].isnull().sum() / len(df[col]) > .70:
        print("Columns more than 70% :  {}".format(col))
df.drop(['alternative_txt', 'claimmode_txt', 'propvalue'], axis= 1, inplace= True)
# Renaming rest of columns
new_cols_names = {  'country_txt' : 'country_name',
                    'region_txt' : 'region_name',
                    'provstate' : 'province_state',
                    'crit1' : 'criteria_pol_eco_rel_goals',
                    'crit2' : 'criteria_coerce',
                    'crit3' : 'criteria_outside_hum_law',
                    'doubtterr' : 'criteria_unsure',
                    'attacktype1_txt' : 'attack_type',
                    'success' : 'attack_status',
                    'suicide' : 'attack_suicide',
                    'weaptype1_txt' : 'weapon_first_type_general',
                    'weapsubtype1_txt' : 'weapon_first_type_specefic',
                    'targtype1_txt' : 'target_victim_type',
                    'targsubtype1_txt' : 'target_victim_subtype',
                    'corp1' : 'name_of_targeted_entity',
                    'target1' : 'name_of_targeted_entity_specific',
                    'natlty1_txt' : 'nationality_of_target',
                    'gname' : 'group_name',
                    'individual' : 'individual_not_groups',
                    'claimed' : 'claim_of_responsibility',
                    'nkill' : 'total_number_killed',
                    'nkillter' : 'total_number_terrorists_killed',
                    'property' : 'property_damage',
                    'propextent_txt' : 'property_damage_category',
                    'nwound': 'total_number_injured',
                    'iyear': 'year', # make it ready for the to_datetime function
                    'imonth': 'month', # make it ready for the to_datetime function
                    'iday': 'day' # make it ready for the to_datetime function 
                 }

df.rename(columns = new_cols_names, inplace=True)
# Solution: Convert 0 to np.nan and then use ffill method to get numbers in previous cells
df.loc[df.month == 0, 'month'] = np.nan
df.loc[df.day == 0, 'day'] = np.nan

df.loc[:, ['month', 'day']] = df.loc[:, ['month', 'day']].fillna(method= 'ffill')  # inplace = True argument doesnt work here. I had to assign it to the "optimized_df.loc[:, ['month', 'day']]"
# Check if the day exceed the month range
for i in df.index.tolist():
    
    prev_month = 1
    y, m, d = df.year[i].astype(int), df.month[i].astype(int), df.day[i].astype(int)
    
    if d > 28:        
        if prev_month != m:
            d = 1
            df.loc[i, 'day'] = d
        elif m == 2:
            d = monthrange(y,m)[1]
            df.loc[i, 'day'] = d

    if (d == 30) | (d == 31):
        if d > monthrange(y,m)[1]:
            d = monthrange(y,m)[1]
            df.loc[i, 'day'] = d
        else:
            d = 1
            df.loc[i, 'day'] = d
# Creating datetime column
df['dates'] = pd.to_datetime(df[['year', 'month', 'day']].astype(int))
# Drop year, month, day columns and re arrange all columns starting with date column
df.drop(['year', 'month', 'day'], axis= 1, inplace= True)

df = df [[ 'dates', 'extended', 'country_name', 'region_name', 'city', 'latitude', 'longitude', 'specificity', 'location',
                               'criteria_pol_eco_rel_goals', 'criteria_coerce', 'criteria_outside_hum_law', 'criteria_unsure', 'attack_status',
                               'attack_suicide', 'attack_type', 'target_victim_type', 'target_victim_subtype', 'name_of_targeted_entity',
                               'name_of_targeted_entity_specific', 'nationality_of_target', 'group_name', 'individual_not_groups', 
                               'claim_of_responsibility', 'weapon_first_type_general', 'total_number_killed', 'total_number_terrorists_killed', 
                               'total_number_injured', 'property_damage', 'property_damage_category', 'summary']]
# Check memory usage
df.info(memory_usage='deep')
# =============================================================================
# SUMMARY - AFTER
# dtypes: category(13), datetime64[ns](1), float64(8), int64(8), object(1)
# memory usage: 87.6 MB
# =============================================================================


# =============================================================================
# SUMMARY - BEFORE
# dtypes: datetime64[ns](1), float64(55), int64(22), object(57)
# memory usage: 590.0 MB
# =============================================================================