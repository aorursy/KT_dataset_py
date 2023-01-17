# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df =  pd.read_csv('../input/Pokemon.csv')  #read the csv file and save it into a variable
df.head(n=10) 
import pandas as pd
df =  pd.read_csv('../input/Pokemon.csv') 
df['Type 2'].value_counts()

import pandas as pd
df =  pd.read_csv('../input/Pokemon.csv') 
df['Type 1'].value_counts()
import pandas as pd
df =  pd.read_csv('../input/Pokemon.csv') 
df['Type 2'].value_counts().plot(kind='barh') # creates a horizontal bar graph

import pandas as pd
import seaborn as sns
df =  pd.read_csv('../input/Pokemon.csv') 

df = df.drop(['Generation', 'Legendary'], 1)

sns.jointplot(x = "HP", y  = "Attack", data= df)


import pandas as pd
import seaborn as sns
df =  pd.read_csv('../input/Pokemon.csv', index_col='#')

pkmn = df.groupby(['Legendary', 'Generation']).mean()[['Attack', 'Defense']]
pkmn.plot.bar(stacked=True)


import pandas as pd
import seaborn as sns
df =  pd.read_csv('../input/Pokemon.csv', index_col='#')

sns.jointplot( x='Attack', y='Defense', data=df, kind='hex', gridsize=20)




import pandas as pd
import seaborn as sns
df =  pd.read_csv('../input/Pokemon.csv', index_col='#')

ax  = df['Type 1'].value_counts().plot.bar(
        figsize=(12,6),
        fontsize=12
)
ax.set_title('Pokemon by Primary Type', fontsize = 25)
sns.despine(bottom = True, left= True)


