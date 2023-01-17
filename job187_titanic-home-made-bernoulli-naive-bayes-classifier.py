# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
##########################################################################################################################################################
#################################################### Kaggle Titanic Challenge - Naive Bayes Classifier ###################################################
##########################################################################################################################################################

# Programmer:   Johan
# Date:         2020-06-27
# Description:  Naive Bayes classifier for predicting survival on the Titanic. Alot of googling is behind this model. Data clearining and preparations are taken from earlier work on the logistic regressoin model. First, an attempt will be made at constructing a Naive Bayes Bernoulli model from how I understand it, without the help of sklearn. Sklearn will be employed afterwards. 

# Naive Bayes:  Naive Bayes classifier assumes that the effect of a particular feature in a class is independent of other features. For example, a loan applicant is desirable or not depending on his/her     income, previous loan and transaction history, age, and location. Even if these features are interdependent, these features are still considered independently. This assumption simplifies computation, and that's why it is considered as naive. (https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn)
# P(A|B) = P(B|A)P(A) / P(B)


# %%

#######################################################
###### Plan for Bernoulli Naive Bayes model by "hand" 
#######################################################

# 0. Data cleaning by imputating missing values

# 1. I need the probabilities of survival and death.

# 2. I need the conditional probabilities of each feature. 

# 3. Score for the probabilities for survival and no survival case given the observed features

# 4. Base classification on the highest probability score.


# %%
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


#Import both test and train sets and concatinate them for data cleaning and feature engineering
df_train = pd.read_csv("D:\\Datasets\\Titanic Machine Learning from Disaster\\kaggle datasets\\train.csv")
df_test = pd.read_csv("D:\\Datasets\\Titanic Machine Learning from Disaster\\kaggle datasets\\test.csv")
df_full = pd.concat([df_train, df_test], axis=0, ignore_index=True)


# %%
###############################
###### Cleaning and imputation
###############################

#Put in the average age for each class
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 39.159930
        elif Pclass == 2:
            return 29.506705
        else:
            return 24.816367
    
    else:
        return Age

#Apply the function
df_full["Age"] = df_full[["Age","Pclass"]].apply(impute_age, axis=1)


# %%
# For Fare, get the mean fare for a specific class and port
df_full.groupby(["Embarked", "Pclass"]).mean()["Fare"]


# %%
# Fill in missing Fare
df_full.loc[df_full["Fare"].isna() == True ] 


# %%
df_full.loc[df_full["PassengerId"] == 1044, "Fare"] = 14.435422


# %%
df_full.loc[df_full["PassengerId"] == 1044]


# %%
# FIll in missing Embarked values
df_full.loc[df_full["Embarked"].isna()]


# %%
df_full.groupby("Embarked").mean()["Fare"]


# %%
df_full.loc[df_full["PassengerId"].isin([62, 830]), "Embarked"] = 'C'


# %%
df_full.loc[df_full["PassengerId"].isin([62, 830])]


# %%
# Feature engineering

# The explanatory variables needs to follow a bernoulli distribution - Fare needs to be split into over/under a certain level. Same with Age. Parch and SibSp as well

df_full["male"] = pd.get_dummies(df_full["Sex"], drop_first=True)


# %%
print(df_full["Fare"].median() ,  df_full["Age"].quantile(0.1) )


# %%
#Turn class into a binary variable, indicating first class or not

def Pclass(pclass):
    if pclass == 1:
        return 1
    else:
        return 0

#After looking at survival rate per age group, 14 years of age seem to be a good indicator of survival
def Age(age):
    if  age <= 14:
        return 1
    else:
        return 0
def Fare(fare):
    if fare >= 14.45:
        return 1
    else:
        return 0

def Parch(parch):
    if parch >= 1:
        return 1
    else:
        return 0
def SibSp(sibsp):
    if sibsp >= 1:
        return 1
    else:
        return 0


# %%
df_full["Kid"] = df_full["Age"].apply(Age)
df_full["High_fare"] = df_full["Fare"].apply(Fare)
df_full["1st_class"] = df_full["Pclass"].apply(Pclass)
df_full["has_parent_child"] = df_full["Parch"].apply(Parch)
df_full["has_sibling_spouse"] = df_full["SibSp"].apply(SibSp)


# %%
#############################################
##### Bayes naive classifier - probabilities
###############

train = df_full[:891]
test = df_full[891:]

# 1st get overall probability of survival and death
count_survived = train.loc[train["Survived"] == 1].count()["PassengerId"]
count_dead = train.loc[train["Survived"] == 0].count()["PassengerId"]
df_full["p_survive"] = count_survived / (count_survived + count_dead)
df_full["p_die"] = 1 - df_full["p_survive"] 


# %%
# 2nd get conditional probabilities for bernoulli distributed explanatory variables

df_full["p_survive_kid"] = train.loc[train["Survived"] == 1].mean()["Kid"]
df_full["p_die_kid"] = train.loc[train["Survived"] == 0].mean()["Kid"]

df_full["p_survive_rich"] = train.loc[train["Survived"] == 1].mean()["High_fare"]
df_full["p_die_rich"] = train.loc[train["Survived"] == 0].mean()["High_fare"]

df_full["p_survive_1stclass"] = train.loc[train["Survived"] == 1].mean()["1st_class"]
df_full["p_die_1stclass"] = train.loc[train["Survived"] == 0].mean()["1st_class"]

df_full["p_survive_parch"] = train.loc[train["Survived"] == 1].mean()["has_parent_child"]
df_full["p_die_parch"] = train.loc[train["Survived"] == 0].mean()["has_parent_child"]

df_full["p_survive_sibsp"] = train.loc[train["Survived"] == 1].mean()["has_sibling_spouse"]
df_full["p_die_sibsp"] = train.loc[train["Survived"] == 0].mean()["has_sibling_spouse"]

df_full["p_survive_male"] = train.loc[train["Survived"] == 1].mean()["male"]
df_full["p_die_male"] = train.loc[train["Survived"] == 0].mean()["male"]


# %%
df_full.columns


# %%
df_full.to_csv("D:\\Datasets\\Titanic Machine Learning from Disaster\\kaggle datasets\\bayes_probabilities.csv")


# %%
df_full["survival_score"] = df_full["p_survive"] * np.where(df_full["male"] == 1, df_full["p_survive_male"], 1) * np.where(df_full["Kid"] == 1, df_full["p_survive_kid"], 1) * np.where(df_full["High_fare"] == 1, df_full["p_survive_rich"], 1) * np.where(df_full["has_parent_child"] == 1, df_full["p_survive_parch"], 1) * np.where(df_full["has_sibling_spouse"] == 1, df_full["p_survive_sibsp"], 1)

df_full["death_score"] = df_full["p_die"] * np.where(df_full["male"] == 1, df_full["p_die_male"], 1) * np.where(df_full["Kid"] == 1, df_full["p_die_kid"], 1) * np.where(df_full["High_fare"] == 1, df_full["p_die_rich"], 1) * np.where(df_full["has_parent_child"] == 1, df_full["p_die_parch"], 1) * np.where(df_full["has_sibling_spouse"] == 1, df_full["p_die_sibsp"], 1)

df_full["pred_survive"] = np.where(df_full["death_score"] < df_full["survival_score"], 1, 0)


# %%
survived = df_full.loc[:890, "Survived"]
pred_sur = df_full.loc[:890, "pred_survive"] 


# %%
confusion_matrix(survived, pred_sur)


# %%
df_full.loc[891:, ["PassengerId","pred_survive"]].to_csv("D:\\Datasets\\Titanic Machine Learning from Disaster\\kaggle datasets\\nb_predictions.csv", index=False)


# %%




