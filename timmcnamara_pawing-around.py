import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
! ls ../input
! head ../input/train.csv
train_df = pd.read_csv('../input/train.csv')
train_df.head()
# AgeuponOutcome will probably be more useful as an actual numeric variable. Also, it's in multiple units.
# Let's take a look at the types of units..

import math

def is_a_value(pandas_value):
    # pandas uses NaN for missing values, 
    # which is kind of annoying
    if not isinstance(pandas_value, float):
        return True
    return not math.isnan(pandas_value)

ages = train_df.AgeuponOutcome.tolist()
ages = filter(lambda a: is_a_value(a), ages)
units = set(age.split()[1] for age in ages)
sorted(list(units))
def normalise_age_at_outcome(age):
    """
    >>> normalise_age_at_outcome("3 weeks")
    3.0
    >>> normalise_age_at_outcome("1 month")
    3
    
    """
    if not is_a_value(age):
        return age
    n, unit = age.split()
    n = int(n)
    if unit.startswith('month'):
        length_of_month_in_weeks = 52.0/12.0
        return n * length_of_month_in_weeks
    elif unit.startswith('year'):
        return n * 52.0
    elif unit.startswith('week'):
        return float(n)
    elif unit.startswith('day'):
        return float(n) / 7.0  
# a few quick tests
normalise_age_at_outcome('3 days'), normalise_age_at_outcome('7 weeks'), normalise_age_at_outcome('4 months')
train_df['age_at_outcome_in_weeks'] = train_df['AgeuponOutcome'].apply(normalise_age_at_outcome)
sns.factorplot(x='OutcomeType', y='age_at_outcome_in_weeks', col='AnimalType', data=train_df, kind='bar')
# looking at the data - see above - 

def is_mixed_breed(animal):
    return animal.endswith('Mix')

train_df['is_mixed_breed'] = train_df['Breed'].apply(is_mixed_breed)
sns.factorplot(x='OutcomeType', y='age_at_outcome_in_weeks', col='AnimalType', hue='is_mixed_breed', data=train_df, kind='bar')
sns.factorplot(x='OutcomeType', y='age_at_outcome_in_weeks', col='Breed', data=train_df, kind='bar', orient="h")
# TODO .. build a proper classifier

from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)
X = v.fit_transform(train_df.to_dict(orient='records'))
X
