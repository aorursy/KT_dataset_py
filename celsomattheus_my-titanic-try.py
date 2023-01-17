#repassar truques de feature selection e engineering do pessoal de Data

#https://queroedu.slack.com/archives/GFT1FHXT9/p1598456771017500



# Feature Engineering

# 01. Criar a feature 'is_alone'

# 02. Criar função para aplicar count e target enconder, e testar CountFrequencyCategoricalEnconder

# 12. Escalar variáveis numéricas, pode melhorar resultado de alguns modelos

#   normalização das features contínuas (MinMaxScale ou StandardScale)

# 14. Testar bagged tree imputation para o Age

# 15. Testar regression imputation para o Age



# Repassar código de R para python

# 03. Criar coluna Cabin_Section ("X" se for NA, primeira letra se for outro)

# 04. Criar family size (Parch + SibSp) em Python

# 05. Criar fare over_fifty em python

# 06. Criar função para centralizar todas as transformações das tabelas numa função

# 07. Arredondar para dezenas de Fare em python

# 08. Arredondar Fare para eliminar centavos em python

# 09. Limpar Age; WHEN Age < 1 THEN 1 ELSE int(Age) em python

# 10. For Embarked we will use the ‘mode’ to replace the null values

# 11. Criar a função honorific() de R para python 
# Wrangling in R



# setwd("~/Área de Trabalho/titanic")

# library(tidyverse)

# train <- read_csv("./train.csv")

# test <- read_csv("./test.csv")



# honorific <- function(full_name){

#   broken <- str_split(full_name, " ")[[1]]

#   honor <- broken[which(str_detect(broken, ","))+1]

#   return(honor)

# }



#pseudocode:

# function(string){

#     split string into elements of an array by the spaces

#     capture index of element with a comma (",") in it, as INX

#     catpure element with index equals to INX+1

#     return captured element

# }



# my_recipe <- function(dataframe){

#     result <- 

#         dataframe %>% 

#         rowwise() %>% 

#         mutate(

#             honorific = honorific(Name) 

#           , family_size = Parch + SibSp

#           , Age = ifelse(Age < 1, 1, as.integer(Age))

#           , Fare = round(Fare)

#           , Fare_d = round(Fare/10)*10

#           , Fare_over_fifty = Fare >= 50

#           , Cabin_Section = ifelse(is.na(Cabin), "X", str_sub(Cabin, 1, 1))

#         ) %>% 

#         ungroup()

    

#     return(result)

# }



# train.mod <- my_recipe(train)

# test.mod <- my_recipe(test)

# write_csv(train.mod, "train_mod.csv")

# write_csv(test.mod, "test_mod.csv")
!pip install pycaret
import pandas as pd

import numpy as np 



train = pd.read_csv('../input/my-first-titanic/train_mod.csv')

test = pd.read_csv('../input/my-first-titanic/test_mod.csv')

sub = pd.read_csv('../input/titanic/gender_submission.csv')
import category_encoders as ce



# Count Enconder: Honorific

count_enc = ce.CountEncoder()

count_enc.fit(train['honorific'])



#Target Enconder: Family Size

target_enc = ce.TargetEncoder(cols=['family_size'])

target_enc.fit(train['family_size'], train['Survived'])



# Transformar as features

train_mod = train.join(count_enc.transform(train['honorific']).add_suffix("_count"))

test_mod = test.join(count_enc.transform(test['honorific']).add_suffix("_count"))

train_mod = train_mod.join(target_enc.transform(train_mod['family_size']).add_suffix('_target'))

test_mod = test_mod.join(target_enc.transform(test_mod['family_size']).add_suffix('_target'))
from pycaret.classification import *



clf_titanic = setup(data = train_mod, 

             target = 'Survived',

             numeric_imputation = 'median',

             categorical_features = ['Sex','Embarked','Cabin_Section','family_size_target','Fare_d','Fare_over_fifty','honorific','honorific_count'], 

             ignore_features = ['Name','Ticket','Cabin','Parch','PassengerId','SibSp'],

             bin_numeric_features = ['Age','Fare'], #Quantizar variáveis numéricas,

             feature_selection = True,

             ignore_low_variance = True,

             silent = True)
top5 = compare_models(n_select = 5)

tuned_top5 = [tune_model(i, n_iter = 50) for i in top5]

bagged_top5 = [ensemble_model(i) for i in tuned_top5]

best = automl()
plot_model(best, 'confusion_matrix')
optimize_threshold(best, true_negative = 1400, false_negative = -2300)
predictions = predict_model(best, data=test_mod, probability_threshold=0.42)

sub['Survived'] = round(predictions['Score']).astype(int)

sub.to_csv('submission.csv',index=False)

sub.head()