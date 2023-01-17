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
genderclassmodel = pd.read_csv('../input/genderclassmodel.csv')

gendermodel = pd.read_csv('../input/gendermodel.csv')

test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')
Pclass3Survived = train[(train['Pclass'] == 3) & (train['Survived'] == 1)]

Pclass2Survived = train[(train['Pclass'] == 2) & (train['Survived'] == 1)]

Pclass1Survived = train[(train['Pclass'] == 1) & (train['Survived'] == 1)]

Pclass3Survived