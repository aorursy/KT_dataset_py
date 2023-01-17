import numpy as np 

import pandas as pd 
# Read files

file = open('/kaggle/input/melanoma-skin-lesion-id-ph2-data/PH2_dataset.txt')

data1 = file.read()
# Fix header

head = data1.split('\n')[0].split('|')

print(f'Original: len({len(head)})\n',head)





head = [val.strip() for val in head if len(val) > 2]

print(f'\nClean: len({len(head)})\n',head)
# Extract data 

df = pd.DataFrame(columns=head) # make new dataframe



data_lines = data1.split('\n')[1:-25] # remove head and remove "Legends for data" in the end



for line in data_lines:

    # extract data

    line = line.split('|')

    row = [val.strip() for val in line if len(val) > 2]

    

    # make a new row for dataframe

    new_row = dict(zip(head, row))

    

    # add row to dataframe

    df = df.append(new_row, ignore_index=True)

df
df.to_csv('PH2_metainfo_1.csv')
# read xlsx file

data2 = pd.read_excel('/kaggle/input/melanoma-skin-lesion-id-ph2-data/PH2_dataset.xlsx')



# extract head row

head = data2.iloc[11].values



# drop "Legends for data"

broken_df = data2.drop(np.arange(12))

broken_df.columns = head



# make new dataframe

df = pd.DataFrame(columns=head)

df = pd.concat([df, broken_df], ignore_index=True)

df
df.to_csv('PH2_metainfo_2.csv')
print('Legends for metadata-1:\n')

data1.split('\n')[-25:]
print('Legends for metadata-2:')

print('not really interpretable =(')

data2[:11]