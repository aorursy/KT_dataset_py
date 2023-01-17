# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_data = pd.read_csv("../input/train.csv", dtype={'Age': np.float64 })

test_data = pd.read_csv("../input/test.csv", dtype={'Age': np.float64 })
len(train_data)
embarked_data = train_data[['Survived', 'Embarked']]

grouped_embark = embarked_data.groupby('Embarked').mean()



p1 = grouped_embark.plot(kind='bar')

p1.set_ylabel("mean()")



grouped_embark = embarked_data.groupby('Embarked').sum()

p2 = grouped_embark.plot(kind='bar')

p2.set_ylabel("count()")



embarked_data['c'] = [1 for _ in range(0, len(embarked_data))]

di_grouped_embark = embarked_data.groupby(['Embarked', 'Survived']).count()

di_grouped_embark.plot(kind='bar', y='c')
def show_correlation_for_variable(variable):

    dataset = train_data

    chosed_data = dataset[['Survived', variable]]

    grouped_data = chosed_data.groupby(variable).mean()

    

    grouped_data.plot(kind='bar')
show_correlation_for_variable('Pclass')

show_correlation_for_variable('Sex')

show_correlation_for_variable('SibSp')

show_correlation_for_variable('Parch')

show_correlation_for_variable('Embarked')

train_data.info()