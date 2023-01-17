import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
print('Reading data...')

key_1 = pd.read_csv('../input/key_1.csv')

train_1 = pd.read_csv('../input/train_1.csv')

ss_1 = pd.read_csv('../input/sample_submission_1.csv')



print ('Data has been read')


print("Train".ljust(15), train_1.shape)

print("Key".ljust(15), key_1.shape)



# print ("Train head".ljust(15), train_1.head())

# print ("Key head".ljust(15), key_1.head())
# Each article name has the following format: 'name_project_access_agent' 

# Take the Page column, split it, throw it in a DataFrame

# Leave off the name with [-3:]

print ("Exploring page names")

page_details = pd.DataFrame([i.split("_")[-3:] for i in train_1["Page"]])

# Rename the columns 

page_details.columns = ["project", "access", "agent"]

print(page_details.describe())



# Filter to unique values and take a look

print("The unique values in a list: ")

project_columns = page_details['project'].unique()

access_columns = page_details['access'].unique()

agents_columns = page_details['agent'].unique()

print(list(project_columns))

print(list(access_columns))

print(list(agents_columns))
#np.nan_to_num(np.round(np.nanmedian(train_1.drop('Page', axis=1).values[:, -56:], axis=1)))
# This scores 64.8 according to the Simple Model

# visits = np.round(np.mean(train_1.drop('Page', axis=1).values, axis=1))



# This scores one point worse than -56:, with a score of 46.7 instead of 45.7

# With -80, we get 46.3.

visits = np.nan_to_num(np.round(np.nanmedian(train_1.drop('Page', axis=1).values[:, -100:], axis=1)))
print ('Now we build our submission')

# for some reason we drop the last 11 letters of the pages

ids = key_1.Id.values

pages = key_1.Page.values

d_pages = {}

for id, page in zip(ids, pages):

    d_pages[id] = page[:-11]



# Now we put our predicted values into our new dictionary

d_visits = {}

for page, visits_number in zip(pages, visits):

    d_visits[page] = visits_number



print('Modifying sample submission...')

# Take the values of the Id and the Visits columns

ss_ids = ss_1.Id.values

ss_visits = ss_1.Visits.values



# enumerate is a Python method that loops over the index and item

for i, ss_id in enumerate(ss_ids):

    ss_visits[i] = d_visits[d_pages[ss_id]]



print('Saving submission...')

# Put in a DataFrame again to use the to_csv method

subm = pd.DataFrame({'Id': ss_ids, 'Visits': ss_visits})

subm.to_csv('submission.csv', index=False)