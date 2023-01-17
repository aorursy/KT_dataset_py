import pandas as pd
titanic_raw_train= pd.read_csv("../input/train.csv")
titanic_raw_train.head()
titanic_raw_train.shape
import numpy as np
titanic_raw_train.describe()
##bayes theorem
n_rows= titanic_raw_train.shape[0]
#(titanic_raw_train['Survived']==1).sum()
p_survived= (titanic_raw_train.Survived==1).sum() / n_rows
p_not_survived = 1 - p_survived
#p_survived
#titanic_raw_train.Sex.isnull().sum()
p_male= (titanic_raw_train.Sex=="male").sum() / n_rows
p_female= 1 - p_male
## do the survival rate have a certain gender affect??? --> P(Survived|Female)=P(Female and Survived)/P(Female)
n_women= (titanic_raw_train.Sex=="female").sum()
survived_women=titanic_raw_train[(titanic_raw_train.Sex=="female") & (titanic_raw_train.Survived==1)].shape[0]
p_survived_women= survived_women / n_women
#p_survived_given_women= p_survived_women / p_female
p_survived_women

#