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

animals = pd.read_csv('../input/train.csv')

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
sns.countplot(animals.AnimalType, palette='Set3')
sns.countplot(animals.OutcomeType, palette='Set3')
sns.countplot(animals.AgeuponOutcome, palette='Set3')
def get_sex(x):
    x = str(x)
    if x.find('Male') >= 0: return 'male'
    if x.find('Female') >= 0: return 'female'
    return 'unknown'
def get_neutered(x):
    x = str(x)
    if x.find('Spayed') >= 0: return 'neutered'
    if x.find('Neutered') >= 0: return 'neutered'
    if x.find('Intact') >= 0: return 'intact'
    return 'unknown'
animals['Sex'] = animals.SexuponOutcome.apply(get_sex)
animals['Neutered'] = animals.SexuponOutcome.apply(get_neutered)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,4))
sns.countplot(animals.Sex, palette='Set2', ax=ax1)
sns.countplot(animals.Neutered, palette='Set1', ax=ax2)
def get_mix(x):
    x = str(x)
    if x.find('Mix') >= 0: return 'Mix'
    return 'not'
animals['Mix'] = animals.Breed.apply(get_mix)
sns.countplot(animals.Mix, palette='Set1')
#See how different parameters influence the outcome.
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))
sns.countplot(data=animals, x="OutcomeType", hue='Sex', ax=ax1)
sns.countplot(data=animals, x='Sex', hue='OutcomeType', ax=ax2)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
sns.countplot(data=animals, x='OutcomeType', hue="AnimalType", ax=ax1)
sns.countplot(data=animals, x='AnimalType', hue="OutcomeType", ax=ax2)
#Compare ages, but first we need to calculate every age in years
def calc_age_in_years(x):
    x = str(x)
    if x == 'nan': return 0
    age = int(x.split()[0])
    if x.find('year') > -1: return age
    if x.find('month') > -1: return age / 12.
    if x.find('week') > -1: return age / 52.
    if  x.find('day') > -1: return age / 365.
    else: return 0
animals['AgeInYears'] = animals.AgeuponOutcome.apply(calc_age_in_years)
sns.distplot(animals.AgeInYears, bins=20, kde=False)
#Most animals in the shelter are 0-1 years old.
#Let's see if age has any effect on outcome.
def calc_age_category(x):
    if x < 3: return 'young'
    if x < 5: return 'young adult'
    if x < 10: return 'adult'
    return 'old'
animals['AgeCategory'] = animals.AgeInYears.apply(calc_age_category)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))
sns.countplot(data=animals, x='OutcomeType', hue='AgeCategory', ax=ax1)
sns.countplot(data=animals, x='AgeCategory', hue='OutcomeType', ax=ax2)
