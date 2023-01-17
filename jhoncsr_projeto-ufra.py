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
train = pd.read_csv('../input/treino.csv')

test = pd.read_csv('../input/Municipios_brasileiros_1.1.csv')

sample = pd.read_csv('../input/Municipios_brasileiros.csv')
from sklearn.preprocessing import LabelEncoder 

# transforma os dados da coluna diagnosis em dados numericos

labelencoder = LabelEncoder()

train['diagnosis'] = labelencoder.fit_transform(train['diagnosis'])

train.head()