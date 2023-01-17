# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output





training_data = pd.read_csv('../input/train.csv')



test_data = pd.read_csv('../input/test.csv')



print(training_data.describe(include='all'))

print(training_data.info())

print()

print(test_data.describe(include='all'))

print(test_data.info())







# Any results you write to the current directory are saved as output.
training_data
combined = pd.concat((test_data, training_data))



categorical = ['Embarked', 'Sex']





for name in categorical:

    classes = set(combined[name])

    print(name, classes)
last_names = list(name.split(',')[0] for name in training_data.Name)

training_data['LastName'] = last_names

print(len(set(last_names)))



group = training_data.groupby('LastName')

print(group.count()['Survived'])