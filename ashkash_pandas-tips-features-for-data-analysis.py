# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns # visualization lib



import random 



# from IPython.core.interactiveshell import InteractiveShell  

# InteractiveShell.ast_node_interactivity = "all" # printing all the line of the cell





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Import  Data 





df =sns.load_dataset('tips')

print(f"{df.shape[0]}rows and {df.shape[1]} columns")

df.head()
#most common way to do Filter in pandas is

df.loc[(df['tip']>6)  & (df['total_bill'] >=30)] 



#elegant method

df.query("tip >6 & total_bill>=30")



# reference global variable name with @

median_tip = df['tip'].median()

display(df.query("tip>@median_tip").head())



# wrap column name containing . with backtick: `

df.rename(columns={'total_bill':'total.bill'}, inplace=True)

display(df.query("`total.bill`<20").head())

df.rename(columns={'total.bill':'total_bill'}, inplace=True)



# wrap string condition with single quotes (this is what I like)

display(df.query("day=='Sat'").head())

# could also do it the other way around (i.e. 'day=="Sat"')
display(df.head())

display(df.tail())



# In the last line, display() is redundant but it is there for consistency. 

# It works the same way if we take out display() from the last line:



display(df.head())

df.tail()
display(df.sort_values(by=['total_bill', 'tip'], ascending=[True, False]).head())

df.sort_values(by=['total_bill', 'tip'],ascending =[1,0]).head()
display(df.nsmallest(5, 'total_bill'))



#the above code is equivalent to 

df.sort_values(by='total_bill').head()
display(df.nlargest(5, 'total_bill'))

display(df.sort_values(by='total_bill', ascending=False).head())
#let’s check out the column types:

df.info()
df.describe(include='all')
display(df.describe(include=['category'])) # categorical types

display(df.describe(include=['number'])) # numerical types
display(df.describe(exclude=['number']))
print(f"{pd.options.display.max_columns} columns")

print(f"{pd.options.display.max_rows} rows")
pd.options.display.max_columns = None

pd.options.display.max_rows = None
# This may or may not be a good idea depending on how big your dataframe is.

# We can also set these options to a number of our choice:

pd.options.display.max_columns = 50

pd.options.display.max_rows = 100
pd.options.display.float_format = '{:.4f}'.format # 4 decimal places
import random



[[random.randint(0, 10) for _ in range(random.randint(3, 5))] for _ in range(10)]
[random.randint(0, 10) for _ in range(random.randint(3, 5))]
n = 10

df = pd.DataFrame(

    {

        "list_col": [[random.randint(0, 10) for _ in range(random.randint(3, 5))] for _ in range(10)],

    }

)

display(df)

df.shape # output
df = df.explode("list_col")

display(df)

df.shape #output