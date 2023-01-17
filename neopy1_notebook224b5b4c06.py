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
df = pd.read_csv('../input/AguaH.csv')

df.info()
#renaming columns to english

col = {'ENE':'JAN', 'ABR':'APR', 'AGO':'AUG','DIC':'DEC'}

temp =[]

for i in df.columns[5:]:

    i = i.split('_')

    if i[1] in col.keys():

        temp.append(col[i[1]]+i[2])

    else:

        temp.append(i[1]+i[2])

column = ['LANDUSE_TYPE','USER','PIPE DIAM','VENDOR','JAN16']+temp

df.columns=column

df.head()
#counting NA values

new_df = df.ix[:,5:]

new_df['NA_count'] = new_df.count(axis=1,numeric_only='True')

new_df['NA_count'] = 84 - new_df['NA_count']

new_df['NA_count'].plot(kind='pie')