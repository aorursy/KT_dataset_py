# Import required libraries
import numpy as np 
import pandas as pd 
#Load raw data

honeyraw9802 = pd.read_csv("../input/honeyraw_1998to2002.csv", sep='delimiter', header = None, skip_blank_lines=True, engine='python')
honeyraw9802 = honeyraw9802[9:]

honeyraw0307 = pd.read_csv("../input/honeyraw_2003to2007.csv", sep='delimiter', header = None, skip_blank_lines=True, engine='python')
honeyraw0307 = honeyraw0307[81:]

honeyraw0812 = pd.read_csv("../input/honeyraw_2008to2012.csv", sep='delimiter', header = None, skip_blank_lines=True, engine='python')
honeyraw0812 = honeyraw0812[72:]
#create an empty dataframe

honeyDF = pd.DataFrame()

# Loop over the dataframes, drop the unwanted columns

for df in (honeyraw9802, honeyraw0307, honeyraw0812):
    test = df[0].apply(lambda x: pd.Series(str(x).split(',')))
    test.columns = ["X1", "X2", "state", "numcol", "yieldpercol", "totalprod", "stocks", "priceperlb", "prodvalue", "X3", "X4", "X5", "X6"]
    test = test.drop(['X1', 'X2', 'X3', 'X4', 'X5', 'X6'], axis=1)
    
    #remove doublequotes
    test['state'] = test['state'].str.replace('"', '')
    test = test[~test['state'].str.contains('/')]
    
    # perform the mathemetaical changes
    test[['numcol']] = (test[['numcol']].apply(pd.to_numeric, errors='coerce')*1000)
    test[['yieldpercol']] = test[['yieldpercol']].apply(pd.to_numeric, errors='coerce')
    test[['totalprod']] = (test[['totalprod']].apply(pd.to_numeric, errors='coerce'))*1000
    test[['stocks']] = (test[['stocks']].apply(pd.to_numeric, errors='coerce')*1000)
    test[['priceperlb']] = (test[['priceperlb']].apply(pd.to_numeric, errors='coerce'))/100
    test[['prodvalue']] = (test[['prodvalue']].apply(pd.to_numeric, errors='coerce'))*1000
    
    #drop rows with NA
    test = test.dropna(axis=0, how='any')
    #print(test.head())
    
    #append the dataframes
    honeyDF = honeyDF.append(test, ignore_index = True)

print(honeyDF.info())
# load the US states dictionary(used for full form to abbreviated form conversion)
# 'import us' doesnt work as the package is not added to kaggle kernel yet
states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}
#Extract the keys and values
lst_of_abbrv = states.keys()
lst_of_states = states.values()
#changes the full form of states into abbreviation
pd.options.mode.chained_assignment = None  # default='warn'
for x in honeyDF['state']:
    for abbr, name in states.items():
        if name == x:
            honeyDF['state']= honeyDF['state'].replace(x, abbr)
# Capture the index to add the year column
start_index = np.array(honeyDF.state[honeyDF.state == 'AL'].index.tolist())
stop_index = np.array(honeyDF.state[honeyDF.state == 'WY'].index.tolist())
diff_index = (stop_index - start_index) + 1
#create an emty array
allArrays = np.array([], dtype='int')
# Based on the index repeat the "year" values
i = 0
for j in range(1998,2013):
    temp_array = np.array(j).repeat(diff_index[i])
    i = i + 1
    allArrays = np.concatenate((allArrays, temp_array), axis=0)   
# append the array containing year values to the main dataframe
honeyDF['year'] = allArrays
# Note that the "year" has been added to the main data frame honeyDF
print(honeyDF.info())
print(honeyDF.head())
print(honeyDF.tail())
#Export the dataframe to the csv file
#honeyDF.to_csv('honeyproduction.csv', index=False)