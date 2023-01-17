# https://www.kaggle.com/startupsci/titanic-data-science-solutions

import numpy as np 
import pandas as pd 


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
train_df.head()
def print_info_table(df):
    tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})
    tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))
    tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.
                             rename(index={0:'null values (%)'}))
    display(tab_info)
    
print_info_table(train_df)
print_info_table(test_df)
train_df.describe()
print('Parch_catagories: ', train_df.Parch.unique().tolist())
print('SibSp_catagories: ', train_df.SibSp.unique().tolist())
print('Pclass_catagories: ', train_df.Pclass.unique().tolist())
print('Sex_catagories: ', train_df.Sex.unique().tolist())
# Store our passenger ID for easy access
PassengerId = test_df['PassengerId']

full_data = [train_df, test_df]
test_df.describe(include=['O'])
# What is the distribution of categorical features?
train_df.describe(include=['O'])
# let's see the correlation btw the survivals and different non-null catagorial values

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# First class has a higher rate of surviving 
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Survived', 'Sex']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Females have higher survival rate