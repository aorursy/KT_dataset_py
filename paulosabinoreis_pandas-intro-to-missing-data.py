import pandas as pd 

import numpy  as np



#    From numpy We'll just use np.nan
'''

    1 - Creating a small dataset to work with.

    2 - Turning it into a Data Frame.

    3 - Trying to understand it.

'''



people_data = {

    'first': ['Corey', 'Jane', 'John', 'Chris', np.nan, None, 'NA'], 

    'last': ['Schafer', 'Doe', 'Doe', 'Schafer', np.nan, np.nan, 'Missing'], 

    'email': ['CoreyMSchafer@gmail.com', 'JaneDoe@email.com', 'JohnDoe@email.com', None, np.nan, 'Anonymous@email.com', 'NA'],

    'age': ['33', '55', '63', '36', None, None, 'Missing']

}



data_frame = pd.DataFrame(people_data)
''' 

    As We can see, there are a lot of missing values. But this is a small dataset,

    if there were thousands of rows, this aproach wouldn't really help us to detect

    missing values:

'''

data_frame
'''

    Therefore We should use some functions to better understand the data.

    In this case, to find out more about the missing values.

'''

data_frame.isna()
# Sum of how many missing data Pandas can easily detect

data_frame.isna().sum() 
'''

    If You look closer, you'll notice that the ['age'] column has 3 missing values (at rows: 4, 5 and 6)

    But the previous function tell us there are only 2. Then, for you to see what's happening, let's drop

    the rows with 'na' values. You'll notice there will still be missing values, but these Pandas won't

    recognize as 'NaN'.

    

'''

data_frame.dropna() 
#    .dropna(how='any') vs .dropna(how='all')

#    First one drops the row if it finds any 'na' value in that row.

#    The second one  only drops the row if all the row has missing values.

#    how='any'   is the default setting.

data_frame.dropna(how='all')  # Fourth row is full of missing values.
'''

    If just a column need to has no missing values (let's say you'll work on ['age'] column, then 

    it doesn't really matter if ['email'] column has or not any email) you can specify where you 

    want to look for missing values.

'''

data_frame.dropna(subset=['age'])  

# data_frame.dropna(subset=['email','age'])  # And You can specify multiple columns too.
'''

    Well, continuing... in order to make Pandas know that 'Missing' and 'NA' values must be treated as 'na' values,

    We could do this:

'''

#   I'll make use of a new data frame to actually modify it.

new_data_frame = pd.DataFrame(people_data)

#   And now We can just replace the "not deteceted missing" values for an actual 'NaN'

new_data_frame.replace(['NA','Missing'], np.nan, inplace=True)



new_data_frame
'''

    Now, It'll detect all of the 'na' existing values

'''

new_data_frame.isna()
new_data_frame.isna().sum()
#  First data frame sum of 'na' values

data_frame.isna().sum()
'''

        Possible ways to handle them:

    1 - You can drop them.

    or

    2 - You can Replace them using .fillna()

'''

new_data_frame.dropna()
new_data_frame.fillna('MISSING')
#  Notice That all modifications are made in a copy of the Data Frame.

#  Then, if We try to print new_data_frame, We'll see that the Data Frame is intact 

new_data_frame
#  To really alter it, You must use inplace=True.

new_data_frame.fillna('MISSING', inplace=True)

new_data_frame
new_data_frame.replace('MISSING', np.nan, inplace=True)

new_data_frame
new_data_frame.dropna(inplace=True)

new_data_frame
print("Remember: Your choice of how to handle the missing data truly depends on what questions You are answering.")

print("There's much more to learn about this topic. I hope this was helpful to You.")