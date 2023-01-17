import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
from sklearn import preprocessing

# Explanatory blog is present at the following url:
# https://medium.com/deepanker-koul/titanic-problem-on-kaggle-running-blog-6bb26f88b91e

titanic_data = pd.read_csv("../input/train.csv")
data = titanic_data.copy() # in case i need the original data again at some point
test_data = pd.read_csv("../input/test.csv")
titanic_data.head()
gender_data = titanic_data[['Survived', 'Sex']];
age_data = titanic_data[['Survived', 'Age']];
age_data = age_data.dropna()
age_data.reset_index(inplace=True)
total_data = titanic_data[['Survived']];
class_data = titanic_data[['Pclass', 'Survived']];
SibSp_data = titanic_data[['SibSp', 'Survived']];
parental_data = titanic_data[['Survived', 'Parch', 'Age']];
Parch_data = titanic_data[['Parch', 'Survived']]

def sex_vs_survivor(row):
    if ((row['Sex']=='male') and (row['Survived']== 1)):
        sex_vs_survivor_result = "male-Alive"
    elif ((row['Sex']=='male') and (row['Survived']== 0)):
        sex_vs_survivor_result = "male-Dead"
    elif ((row['Sex']=='female') and (row['Survived']== 1)):
        sex_vs_survivor_result = "female-Alive"
    elif ((row['Sex']=='female') and (row['Survived']== 0)):
        sex_vs_survivor_result = "female-Dead"
    return sex_vs_survivor_result;

def age_vs_survival(row):
    if ((row['Age']>10) and (row['Survived']== 1)):
        age_vs_survival_output = "Adult-Alive"
    elif ((row['Age']>10) and (row['Survived']== 0)):
        age_vs_survival_output = "Adult-Dead"
    elif ((row['Age']<=10) and (row['Survived']== 1)):
        age_vs_survival_output = "Child-Alive"
    elif ((row['Age']<=10) and (row['Survived']== 0)):
        age_vs_survival_output = "Child-Dead"
    return age_vs_survival_output;

def all_vs_survival(row):
    if ( (row['Survived']== 1)):
        all_vs_survival_output = "Adult-Alive"
    elif ( (row['Survived']== 0)):
        all_vs_survival_output = "Adult-Dead"
    return all_vs_survival_output

def class_vs_survival(row):
    if ((row['Pclass']==1) and (row['Survived']== 1)):
        class_vs_survival_outputs = "Class-1Alive"
    elif ((row['Pclass']==2) and (row['Survived']== 1)):
        class_vs_survival_outputs = "Class-2Alive"
    elif ((row['Pclass']==3) and (row['Survived']== 1)):
        class_vs_survival_outputs = "Class-3Alive"
    elif ((row['Pclass']==1) and (row['Survived']== 0)):
        class_vs_survival_outputs = "Class-1Dead"
    elif ((row['Pclass']==2) and (row['Survived']== 0)):
        class_vs_survival_outputs = "Class-2Dead"
    elif ((row['Pclass']==3) and (row['Survived']== 0)):
        class_vs_survival_outputs = "Class-3Dead"
    return class_vs_survival_outputs

def SibSp_vs_survival(row):
    if ((row['SibSp']==0) and (row['Survived']== 1)):
        SibSp_vs_survival_output = "Alone-Alive"
    elif ((row['SibSp']==0) and (row['Survived']== 0)):
        SibSp_vs_survival_output = "Alone-Dead"
    elif ((row['SibSp']==1) and (row['Survived']== 1)):
        SibSp_vs_survival_output = "Pair-Alive"
    elif ((row['SibSp']==1) and (row['Survived']== 0)):
        SibSp_vs_survival_output = "Pair-Dead"
    elif ((row['SibSp']>=2) and (row['Survived']== 1)):
        SibSp_vs_survival_output = "Group-Alive"
    elif ((row['SibSp']>=2) and (row['Survived']== 0)):
        SibSp_vs_survival_output = "Group-Dead"
    return SibSp_vs_survival_output

def Parch_vs_survival(row):
    if ((row['Parch']==0) and (row['Survived']== 1)):
        Parch_vs_survival_output = "Alone-Alive"
    elif ((row['Parch']==0) and (row['Survived']== 0)):
        Parch_vs_survival_output = "Alone-Dead"
    elif ((row['Parch']==1) and (row['Survived']== 1)):
        Parch_vs_survival_output = "Pair-Alive"
    elif ((row['Parch']==1) and (row['Survived']== 0)):
        Parch_vs_survival_output = "Pair-Dead"
    elif ((row['Parch']>=2) and (row['Survived']== 1)):
        Parch_vs_survival_output = "Group-Alive"
    elif ((row['Parch']>=2) and (row['Survived']== 0)):
        Parch_vs_survival_output = "Group-Dead"
    return Parch_vs_survival_output

def parental_Survival(row):
    if ((row['Parch']==1) and (row['Age']>= 25) and (row['Survived']== 1)):
        parental_Survival_output = "Parent-Alive"
    elif ((row['Parch']==1) and (row['Age']>= 25) and (row['Survived']== 0)):
        parental_Survival_output = "Parent-Dead"
    elif ((row['Parch']==0) and (row['Age']>= 25) and (row['Survived']== 1)):
        parental_Survival_output = "Non_parent-Alive"
    elif ((row['Parch']==0) and (row['Age']>= 25) and (row['Survived']== 0)):
        parental_Survival_output = "Non_parent-Dead"
    else:
        if(row['Survived']==1):
            parental_Survival_output = "Others_Alive"
        elif(row['Survived']==0):
            parental_Survival_output = "Others_Dead"
    return parental_Survival_output
gender_data['final_gender_data'] = gender_data.apply(sex_vs_survivor, axis = 1)
age_data['final_age_data'] = age_data.apply(age_vs_survival, axis = 1)
total_data['final_total_data'] = total_data.apply(all_vs_survival, axis = 1)
class_data['final_class_data'] = class_data.apply(class_vs_survival, axis = 1)
parental_data['final_parental_data'] = parental_data.apply(parental_Survival, axis=1)
SibSp_data['final_SibSp_data'] = SibSp_data.apply(SibSp_vs_survival, axis=1)
Parch_data['final_Parch_data'] = Parch_data.apply(Parch_vs_survival, axis=1)
gender_data = pd.get_dummies(gender_data['final_gender_data'])
age_data = pd.get_dummies(age_data['final_age_data'])
total_data = pd.get_dummies(total_data['final_total_data'])
class_data = pd.get_dummies(class_data['final_class_data'])
SibSp_data = pd.get_dummies(SibSp_data['final_SibSp_data'])
parental_data = pd.get_dummies(parental_data['final_parental_data'])
Parch_data = pd.get_dummies(Parch_data['final_Parch_data'])
gender_data.sum()
age_data.sum()
total_data.sum()
class_data.sum()
SibSp_data.sum()
Parch_data.sum()
parental_dat.sum()
age_mean = titanic_data['Age'].mean()
age_mean
titanic_data['Age'].head()
titanic_data["Age"] = titanic_data["Age"].fillna(age_mean);
titanic_data["Age"].isnull().values.any()
