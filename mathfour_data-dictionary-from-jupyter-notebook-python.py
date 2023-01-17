import pandas as pd
import numpy as np
pd.options.display.max_columns = 60
pd.options.display.max_rows = 35
import warnings
warnings.filterwarnings("ignore")
def count_blanks(column):
    count = 0
    for thing in column:
        if not thing:
            count += 1
    return count
data = pd.read_csv('../input/PlayData.csv')
data.head()
data_strings = ['identifier', 'type', 'city', 'code', 'name', 'damage', 'number_of_engines'
                'make', 'serial_number', 'purpose', 'phase_of_flight', 'status']
data_bools = ['yes_no_bool']
data_numbers = ['integer_one', 'integer_two', 'cost']
data_dates = ['date', 'another_date']
data_latlong = ['latitude', 'longitude']
data_dictionary = pd.DataFrame(index=data.columns, columns=['missing_vals', 'blanks', 'type', 'description'])
for name in data.columns:
    data_dictionary.loc[name,'missing_vals'] = data[name].isna().sum()
    data_dictionary.loc[name,'blanks'] = count_blanks(data[name])
    data_dictionary.loc[name,'type'] = 'string'
data_dictionary
data_dictionary.loc[data_dates,'type'] = 'date'
data_dictionary.loc[data_bools,'type'] = 'bool'
data_dictionary.loc[data_numbers,'type'] = 'int'
data_dictionary.loc[data_numbers,'type'] = 'float'
data_dictionary.loc['identifier','description'] = 'Words_To_Describe_It'
data_dictionary.loc['type','description'] = 'Words_To_Describe_It'
data_dictionary.loc['city','description'] = 'Words_To_Describe_It'
data_dictionary.loc['code','description'] = 'Words_To_Describe_It'
data_dictionary.loc['name','description'] = 'Words_To_Describe_It'
data_dictionary.loc['damage','description'] = 'Words_To_Describe_It'
data_dictionary.loc['make','description'] = 'Words_To_Describe_It'
data_dictionary.loc['serial_number','description'] = 'Words_To_Describe_It'
data_dictionary.loc['purpose','description'] = 'Words_To_Describe_It'
data_dictionary.loc['phase_of_flight','description'] = 'Words_To_Describe_It'
data_dictionary.loc['status','description'] = 'Words_To_Describe_It'
data_dictionary.loc['yes_no_bool','description'] = 'Words_To_Describe_It'
data_dictionary.loc['number_of_engines','description'] = 'Words_To_Describe_It'
data_dictionary.loc['integer_one','description'] = 'Words_To_Describe_It'
data_dictionary.loc['integer_two','description'] = 'Words_To_Describe_It'
data_dictionary.loc['cost','description'] = 'Words_To_Describe_It'
data_dictionary.loc['date','description'] = 'Words_To_Describe_It'
data_dictionary.loc['another_date','description'] = 'Words_To_Describe_It'
data_dictionary.loc['latitude','description'] = 'Words_To_Describe_It'
data_dictionary.loc['longitude','description'] = 'Words_To_Describe_It'
# take a look
data_dictionary
data_dictionary.to_csv('DataDictionary.csv')
data_dictionary.sort_values(by='type') # or missing_vals or description
for column in data_numbers:
    data[column] = pd.to_numeric(data[column])
for column in data_bools:
    data[column] = data[column].astype(bool)
for column in data_dates:
    data[column] = pd.to_datetime(data[column])
data.info()