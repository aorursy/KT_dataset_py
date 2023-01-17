# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/80-cereals/cereal.csv')
df.isnull().sum()
plt.rcParams['figure.figsize']=(20,10)
ax = sns.barplot(y = 'name', x = 'rating', data = df.sort_values('rating', ascending = False)[:15])

ax.set(title = 'Top 15 cereals', ylabel = 'Name', xlabel = 'Rating')
ax = sns.lmplot(y = 'calories', x = 'rating', data = df)

ax.set(title = 'Relationship between Calories and Rating')
candidate_cols = [col for col in df.columns if df[col].dtype == 'int64' or df[col].dtype == 'float64' if col not in ['rating']]        
labels = []

values = []



for col in candidate_cols:

    labels.append(col)

    values.append(np.corrcoef(df.rating, df[col])[0,1])





corr_df = pd.DataFrame({'label':labels, 'coeff-value': values})

corr_df.sort_values('coeff-value', inplace = True)



plt.barh(corr_df.label, corr_df['coeff-value'], color = 'salmon')
sns.pairplot(data = df[['fiber','protein','potass']], kind = 'reg', height = 4)
df['protein/calorie'] = df['protein']/df['calories']
df.rating.describe()
dfBestRating = df[df.rating > 50]
sns.set(style="darkgrid")

ax = sns.lmplot(x = 'protein/calorie', y = 'rating', data = dfBestRating, height = 15, line_kws = {'color' : 'orange'})



ax.set(xlim=(0.015, 0.085), ylim=(45, 95))

plt.title('Protein/calories vs Rating')

plt.xlabel('Protein/calories ratio')

plt.ylabel('Rating')



df1 = dfBestRating[['protein/calorie','rating','name']]



for idx,row in df1.iterrows():

    ax = plt.gca()

    ax.text(x=row[0]+0.001,y=row[1],s=row[2])