import pandas  as pd

#load the nobel laureates data
laureates_data = pd.read_csv('/kaggle/input/nobel-laureates/archive.csv')

# get the columns of the data
print(laureates_data.columns)

# look for missing values
print(laureates_data[laureates_data.isnull().any(axis=1)])
# I want more insights into what sort of data to expect
print(laureates_data.head())
print(laureates_data.dtypes)
# Let's see what values other than 'Individual' the Laureate type can take
print(laureates_data[laureates_data['Laureate Type'] != 'Individual']['Laureate Type'])
# Let's see what are the fields where organizations have been awarded a Nobel prize
print(laureates_data[laureates_data['Laureate Type'] != 'Individual']['Category'].unique())
print('The number of entries: %d \n\n' % laureates_data[laureates_data['Full Name'].str.contains('Marie Curie')].shape[0])
print(laureates_data[laureates_data['Full Name'].str.contains('Marie Curie')])
name_counts = laureates_data['Full Name'].value_counts()
multi_name = list(name_counts[name_counts > 1].index)

for name in multi_name:
    temp = laureates_data[laureates_data['Full Name']==name].Year.unique()
    if len(temp) > 1:
        print(name, ' ', temp, '\n')