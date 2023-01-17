# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.set_option('display.max_columns', None) # Display all columns in the DataFrame as default

pd.set_option('display.max_rows', None) # Display all rows in the DataFrame as default

pd.options.display.float_format = '{:.2f}'.format # Suppresses scientific notation
df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx', sheet_name='All') # Gathering Data: Use Pandas read_excel function to load the DataFrame

df.head(n=10) # Show the first 10 rows
print('DataFrame Shape: ', df.shape) # Show the DataFrame Shape

df.describe(include='all') # Show data descriptive summary
df.replace(to_replace = ["negative","not_detected","absent","normal"], value = 0, inplace = True) # Replace to binary

df.replace(to_replace = ["positive","detected","present"], value = 1, inplace = True) # Replace to binary

df.replace(to_replace = ["not_done", "Não Realizado"], value = pd.np.nan, inplace = True) # Replace 'not_done' to NaN



# Leukocytes < 10.000 is considered normal while greater or equal this value is abnormal

df[['Urine - Leukocytes']].replace(to_replace = "<1000", value = 0, inplace = True) # Replace to zero if equal to <1000

df['Urine - Leukocytes'] = pd.to_numeric(df['Urine - Leukocytes'], errors = 'coerce') # Convert to numeric

df.loc[df['Urine - Leukocytes'] < 10000, 'Urine - Leukocytes'] = 0 # Replace to zero if < 10000

df.loc[df['Urine - Leukocytes'] >= 10000, 'Urine - Leukocytes'] = 1 # Replace to 1 if >= 10000

df['Urine - pH'] = pd.to_numeric(df['Urine - pH'], errors = 'coerce') # Convert to numeric

df[['Urine - pH']].replace(to_replace = "5.0", value = 5, inplace = True) # Adjust number

df[['Urine - pH']].replace(to_replace = "5.5", value = 5.5, inplace = True) # Adjust number

df[['Urine - pH']].replace(to_replace = "6.0", value = 6, inplace = True) # Adjust number

df[['Urine - pH']].replace(to_replace = "6.5", value = 6.5, inplace = True) # Adjust number

df[['Urine - pH']].replace(to_replace = "7.0", value = 7, inplace = True) # Adjust number

df[['Urine - pH']].replace(to_replace = "7.5", value = 7.5, inplace = True) # Adjust number

df[['Urine - pH']].replace(to_replace = "8.0", value = 8, inplace = True) # Adjust number

df2 = df.dropna(axis = 'columns', how = 'all') # Exclude columns with NULL values

df3 = df2.drop(['Parainfluenza 2', 'Fio2 (venous blood gas analysis)', 'Myeloblasts', 'Urine - Esterase', 'Urine - Bile pigments',

               'Urine - Ketone Bodies', 'Urine - Urobilinogen', 'Urine - Protein', 'Urine - Hyaline cylinders', 'Urine - Granular cylinders', 

               'Urine - Yeasts'], axis = 'columns') # Remove columns that mean and std are zero or nan

df4 = df3.drop(['Urine - Aspect', 'Urine - Crystals', 'Urine - Color'], axis = 'columns') # Remove categorical features with little clinical relevance

print('DataFrame Shape: ', df4.shape) # Show the DataFrame Shape
df4.describe(include='all')
df2 = df.dropna(axis=0, subset=['Hemoglobin'])

df2.head()
df2.count()