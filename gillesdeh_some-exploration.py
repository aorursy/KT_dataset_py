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
df = pd.read_csv('../input/movie_metadata.csv')



df.head()
df.columns
import seaborn as sns
sns.set_context('poster')

sns.set_style('whitegrid')
sns.factorplot('content_rating', 'imdb_score', data=df, aspect=4)
df.groupby(['actor_1_name']).size().sort_values(ascending=False)


df.columns
sns.factorplot('title_year', 'imdb_score', data=df[df.actor_1_name == 'Nicolas Cage'], aspect=5)
pd.crosstab(df, ['actor_1_name', 'actor_2_name'])
import pymc3 as pm
import pystan
df.groupby(['actor_2_name']).size().sort_values(ascending=False)