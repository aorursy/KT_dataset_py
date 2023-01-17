import numpy as np
import pandas as pd
df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')
df_train.shape
df_test.shape
df_train.head()
df_train.isnull().sum()
df_test.isnull().sum()
def getInitial(name):
    return name[name.find(',')+2:name.find('.')]
df_train.insert(2, 'Initial', df_train.Name.apply(getInitial), True)
df_test.insert(2, 'Initial', df_test.Name.apply(getInitial), True)
df_train.head()
df_train.Initial.unique()
df_train.insert(7, 'FamilySize', df_train.SibSp+df_train.Parch, True)
df_test.insert(7, 'FamilySize', df_test.SibSp+df_test.Parch, True)
