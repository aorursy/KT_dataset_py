# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
file = "../input/austin_311_service_requests.csv"

df = pd.read_csv(file)

df.columns
# drop rows with all na values

print('len before {}'.format(len(df)))

df=df.dropna(axis=0, how='all')

print('len after {}'.format(len(df)))
# null values

len(df) - df.count()
import re



def clean_city(string):

    if pd.isnull(string):

        return 'AUSTIN'

    string = string.upper()

    

    p = re.compile(r'(?i)aus')

    if p.search(string):

        return 'AUSTIN'

    

    p = re.compile(r'(?i)DRIPPING')

    if p.search(string):

        return 'DRIPPING SPRINGS'

    

    p = re.compile(r'(?i)WEST LAKE')

    if p.search(string):

        return 'WEST LAKE HILLS'

    

    p = re.compile(r'(?i)PFLUGERVILL')

    if p.search(string):

        return 'PFLUGERVILLE'

    

    return string



df['city'] = df['city'].apply(clean_city)



df['city'].value_counts()
df['county'].value_counts()
df['incident_zip'].value_counts()
df_address=df[['incident_address','incident_zip','street_name','street_number']]

null_address_df=df_address[df_address.isnull().all(axis=1)]

df_address[(df_address['incident_zip'].isnull())&(df_address['incident_address'].notnull())]
df['location'].value_counts()
# df['owning_department'].value_counts()

df[df['owning_department'].isnull()]
df['unique_key'].value_counts()