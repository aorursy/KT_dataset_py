# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from ggplot import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')
train
ggplot(train, aes(x='AnimalType', fill='OutcomeType')) + geom_bar(colour='black') + labs(y='Animals')

Breed_info = train.Breed
Breed_info.loc[Breed_info.str.contains('/')] = Breed_info.str.partition('/')[0]
Breed_info.loc[Breed_info.str.contains('/')] = Breed_info.str.partition(' /')[0]
Breed_info.loc[Breed_info.str.contains('Mix')] = Breed_info.str.partition(' Mix')[0]

Breeds = pd.DataFrame({'Breed': Breed_info.value_counts().keys(), 'Count': Breed_info.value_counts().values})
Breeds

cats = train.Color[train.AnimalType == 'Cat']
cats.loc[cats.str.contains('Tabby')] = 'Tabby'
cats.loc[cats.str.contains('/')] = cats.str.partition('/')[0]
train.Color[train.AnimalType == 'Cat'] = cats

dogs = train.Color[train.AnimalType == 'Dog']
dogs.loc[dogs.str.contains('/')] = dogs.str.partition('/')[0]
train.Color[train.AnimalType == 'Dog'] = dogs
cats = train.Breed[train.AnimalType == 'Cat']
cats[cats == 'American Shorthair'] = 'Domestic Shorthair'
train.Breed[train.AnimalType == 'Cats'] = cats
train.AgeuponOutcome[train.AgeuponOutcome.str.contains('s')]

