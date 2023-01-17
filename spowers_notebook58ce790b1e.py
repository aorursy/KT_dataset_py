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
#Load training and test datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#Utilities



def gender_to_numeric(df):

    #convert gender category to numeric value.  modified in place

    df.loc[df['Sex'] == 'male', 'Sex'] = 0

    df.loc[df['Sex'] == 'female', 'Sex'] = 1

#deal with missing age data, use median age as substitute.  

train['Age'].fillna(train['Age'].median(), inplace=True)

test['Age'].fillna(test['Age'].median(), inplace=True)

#convert gender into numeric value

gender_to_numeric(train)

gender_to_numeric(test)