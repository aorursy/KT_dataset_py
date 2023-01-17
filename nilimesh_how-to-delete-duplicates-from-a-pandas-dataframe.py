# ========================================================================================

# Applied Data Science Recipes @ https://wacamlds.podia.com

# Western Australian Center for Applied Machine Learning and Data Science (WACAMLDS)

# ========================================================================================



print()

print(format('How to delete duplicates from a Pandas DataFrame','*^82'))    

import warnings

warnings.filterwarnings("ignore")



# load libraries

import pandas as pd



# Create dataframe with duplicates

raw_data = {'first_name': ['Jason', 'Jason', 'Jason','Tina', 'Jake', 'Amy'], 

            'last_name': ['Miller', 'Miller', 'Miller','Ali', 'Milner', 'Cooze'], 

            'age': [42, 42, 1111111, 36, 24, 73], 

            'preTestScore': [4, 4, 4, 31, 2, 3],

            'postTestScore': [25, 25, 25, 57, 62, 70]}



df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 

                                        'preTestScore', 'postTestScore'])

print(); print(df)



# Identify which observations are duplicates

print(); print(df.duplicated())



print(); print(df.drop_duplicates(keep='first'))



# Drop duplicates in the first name column, but take the last obs in the duplicated set

print(); print(df.drop_duplicates(['first_name'], keep='last'))