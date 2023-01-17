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
train_data = pd.read_csv('../input/train.csv')

test = pd.read_csv("../input/test.csv")



train_data.head()
train_data["Survived"].value_counts(dropna=False, normalize=True)
submission1 = pd.read_csv('../input/gender_submission.csv')



submission1['Survived'] = submissin1['Survived'].map(lambda s: 0)

#submission1.to_csv('submission1.csv', index= False)
train_data.groupby("Sex")["Survived"].value_counts(dropna=False, normalize=True).sort_index()
pid2 = []

survived2 = []



for index, row in test.iterrows():

    pid2.append(str(row['PassengerId']))

    if row['Sex'] == 'female':

        survived2 += '1'

    else:

        survived2 += '0'



submission2 = pd.DataFrame.from_dict({'PassengerId': pid2, 'Survived': survived2})



#submission2.head()



submission2.to_csv('submission.csv', index= False)