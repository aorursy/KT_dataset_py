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
from scipy import stats

import seaborn as sns

import matplotlib.pyplot as plt



dfb = pd.read_csv('../input/Health_AnimalBites.csv')
print (dfb.info())
print (dfb['SpeciesIDDesc'].unique())

print (dfb['BreedIDDesc'].unique())
dogs = dfb[dfb['SpeciesIDDesc'] == 'DOG']
dogs['bite_date'] = pd.to_datetime(dogs['bite_date'], format = '%Y-%m-%d')
print (dogs['bite_date'].describe())

print (dogs['bite_date'].isnull().sum())
dogs = dogs.sort_values(by='bite_date')
dogs_cleaned = dogs.loc[:, ['bite_date', 'SpeciesIDDesc', 'BreedIDDesc']]

dogs_cleaned = dogs_cleaned.dropna()
print (dogs_cleaned)
dogs_cleaned = dogs_cleaned.reset_index()
print (dogs_cleaned)
dogs_cleaned = dogs_cleaned.loc[:3696,]
dogs_cleaned['bite_date'] = pd.to_datetime(dogs_cleaned['bite_date'], format='%Y-%m-%d')
print (dogs_cleaned.info())
dogs_cleaned = dogs_cleaned[dogs_cleaned['bite_date'] >= '2010-01-01']

dogs_cleaned = dogs_cleaned[dogs_cleaned['bite_date'] < '2017-01-01']
dogs_cleaned['year'] = dogs_cleaned['bite_date'].apply(lambda x: x.year)
c_t_dogs = pd.crosstab(dogs_cleaned['year'], dogs_cleaned['SpeciesIDDesc'])
print (c_t_dogs)
dogs_cleaned = dogs_cleaned[dogs_cleaned['bite_date'] >= '2011-01-01']

c_t_dogs = pd.crosstab(dogs_cleaned['year'], dogs_cleaned['SpeciesIDDesc'])

print (c_t_dogs)
print (stats.chisquare(c_t_dogs['DOG']))
c_t_breed = pd.crosstab(dogs_cleaned['BreedIDDesc'], dogs_cleaned['SpeciesIDDesc'])

print (c_t_breed)

print (stats.chisquare(c_t_breed['DOG']))
print (c_t_breed.describe())
c_t_breed = c_t_breed[c_t_breed['DOG'] > 24.5]

print (c_t_breed)

print (stats.chisquare(c_t_breed['DOG']))
c_t_breed = c_t_breed[c_t_breed['DOG'] > 100]

print (c_t_breed)

print (stats.chisquare(c_t_breed['DOG']))