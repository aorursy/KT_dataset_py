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
df = pd.read_csv("../input/Test_file.csv")
df
## Getting rid of columns and rows that are all null (empty)

df = df.dropna(axis=1, how= 'all')

df = df.dropna(axis=0, how='all')
df['Last Name'] = df['Last Name'].str.lower()

df['First Name'] = df['First Name'].str.lower()
df1 = df[df['First Name'].isnull()]

df2 = df[df['First Name'].notnull()]
df1
df1_name_list = []

for e in df1['Last Name'].str.split(','):

    df1_name_list.append((e[1] + " " + e[0]).strip())
df1['Name'] = df1_name_list
df2
df2['Name'] = df2['First Name'] + " " + df2['Last Name']
df_new = pd.concat([df1,df2], axis=0, sort=False)
df_new
master_list = ['Lam Totien', 'Guo Wei Zhang','Yicai Samuel','Michelle Phan','John Rogers','James Lee',

              'Minh Kim','Phan Le','Larry Kavagna', 'Josh Kim','Huy Handsome','Phuong Beautiful']
## Make master list names all lowercase

new_master_list = []

for i in master_list:

    new_master_list.append(i.lower())
## New column that indicates whether the employee name is in the master list or not

in_master_list_bool = []

for a in df_new.Name:

    if a in new_master_list:

        in_master_list_bool.append(1)

    else:

        in_master_list_bool.append(0)



df_new['in_master_list_bool'] = in_master_list_bool        
print("Names and email of employees who are in the master list: ")

df_new[df_new.in_master_list_bool==1].loc[:,['Name','Email Address']]