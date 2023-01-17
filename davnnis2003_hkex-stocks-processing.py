# check all the available files



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd



columns = ['Time', 'Closed Price', 'Volumne', 'Stock Code']

df = pd.DataFrame()



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        if df.empty:

            print('creating new DataFrame... ')

            print('')

            df = pd.read_csv(os.path.join(dirname, filename))

            df['Stock Code'] = filename.split('.')[0]

            print('New DataFrame:')

            print(df.head())

            print('')

        else:

            df_new = pd.read_csv(os.path.join(dirname, filename))

            df_new['Stock Code'] = filename.split('.')[0]

            print('New DataFrame:')

            print(df_new.head())

            df = pd.concat([df, df_new], axis = 0)

            print('')



df = df[['Stock Code', 'Time', 'Closed Price', 'Volume']]

print('Output:')

print(df)
# Export the combined data

df.to_csv('combined_data.csv', header = True, index = False)