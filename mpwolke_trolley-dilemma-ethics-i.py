#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQL4k93lwMMZ-PfeoigYMt6uF7SOeKXlytAh4hS_RN_xyRDWiCnBA&s',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQuXPgew8sfq7194uFPS5ZdDAYUGBfZaJk1hOC6VtcFAWzB3M2kRw&s',width=400,height=400)
nRowsRead = 1000 # specify 'None' if want to read whole file

# trolley.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df = pd.read_csv('/kaggle/input/trolley-dilemma/trolley.csv', delimiter=',', nrows = nRowsRead)

df.dataframeName = 'trolley.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcShQu7hCZP0srGjBZwoq3db0eY8Z32XQ04xw_ysfitGJOhPL0yX2g&s',width=400,height=400)
df.dtypes
sns.distplot(df["killer"].apply(lambda x: x**4))

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTLiIu5YQVOUQ5nN0xI4RzUN4xZnMDBlQKwOnauBfjXYLq7HnU17Q&s',width=400,height=400)
sns.distplot(df["health"].apply(lambda x: x**4))

plt.show()
#codes from PSVishnu @psvishnu

num = df.select_dtypes ( include = "number" )
#codes from PSVishnu @psvishnu

counter = 1

plt.figure(figsize=(15,18))

for col in num.columns:

    if np.abs(df [col].skew ( )) > 1 and df[col].nunique() < 10:

        plt.subplot(5,3,counter)

        counter += 1

        df [col].value_counts().plot.bar()

        plt.xticks(rotation = 45)

        plt.title(f'{col}\n(skewness {round(df [ col ].skew ( ),2)})')



plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

plt.show ( )
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRWktnkiuUhANkHwQFvXrc7HutgvrQCbsh7V9DRSV23Phka6v0ESA&s',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRYq-la4F7cxYT_wS3tYz6oxRH9PZLHxE8-Sp1V3D3eNJSEM_Un&s',width=400,height=400)
%%time

# > 10 sec

counter = 1

truly_conti = []

plt.figure(figsize=(18,40))

for i in num.columns:

    if np.abs(df [ i ].skew ( )) > 1 and df[i].nunique() > 10:

        plt.subplot(20,3,counter)

        counter += 1

        truly_conti.append(i)

        sns.distplot ( df [ i ] )

        plt.title(f'{i} (skewness {round(df [ i ].skew ( ),1)})')

        plt.xticks(rotation = 45)

plt.tight_layout()

plt.show ( )
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQRRj-tfuSzPt7o3-94o2wQ27_miKovtxRB8SU_iEbkwGH0f70iLQ&s',width=400,height=400)
plt.figure(figsize=(18,3))

sns.boxenplot(data=df.loc[:,truly_conti[0]],orient='h')

plt.title(truly_conti[0])

plt.show()
ethics = [

    'id','health','relative_bais','age','expected_years_left','total years', 'psychopath', 'IQ'

]
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = '',width=400,height=400)
sns.pairplot(data=df,diag_kind='kde',vars=ethics,hue='age')

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ9wU1N3BRNpBZSPwPEAb3udvDkZD-uYlBc5Kv__MUJXwnThqE&s',width=400,height=400)