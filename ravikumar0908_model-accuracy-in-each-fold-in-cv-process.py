# -*- coding: utf-8 -*-

"""

Created on Sat Mar 21 17:26:40 2020



@author: ravilocalaccount

"""



#Cross-Validation :- This Technique is used to know the skill of the model, how well it could learn



import pandas as pd

#loading the dataset, this dataset is stored by me in kaggle input directory,

#so thats why i am referring it from there only

boston=pd.read_csv("../input/labelledBoston.csv")



#extract the target variable

target=boston["labelled"]



#drop the target variable from dataset ; because to mode we pass dataset and targetset separately

boston=boston.drop(columns=["labelled"],axis=1)







#now our Objective :- To divide the dataset into 5 cross validation folds 

#    and find accuracy of model for  each fold 



#this cross validation can tell us how much the model can learn from the dataset



#importing cross_val_score class, to see the model score on each trained fold.

from sklearn.model_selection import cross_val_score

#importing the LinearRegression class from sklearn.linear_model 

from sklearn.linear_model import LogisticRegression

#creating the model

model=LogisticRegression(solver='lbfgs',max_iter=2000)



#importing the KFold class

from sklearn.model_selection import KFold

#creating the KFold object configuration to be used

#n_splits =5 means we want 5 folds

kfold1=KFold(n_splits=5,shuffle=True,random_state=1233)



#passing cv= kfold1 means we want to apply kFold cross-validation

#here we passed the dataset,targetVariable, and Kfold configuration 

#note: no folds passes

#scoring :- means the scoring metric used to evaluate the accuracy of each model

#link to find more metrics 

# https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules

#for LogisticRegression we are using accuracy as scoring metric

scores=cross_val_score(model,boston,target,cv=kfold1,scoring="accuracy")



print("scores of each 5 folds %s " %  scores)



print("mean of cross-validation %s" % scores.mean())

print("standard deviation of cross-validation %s" % scores.std())



#now Cross-Verifying model accuracy for each fold by passing each fold contained

#inside the  generator object, that is returned by kfold.split(boston,target) method

#with another model

from sklearn.model_selection import KFold



#creating the KFold object configuration to be used

kfold=KFold(n_splits=5,shuffle=True,random_state=1233)

#each fold is stored in generator object noe

generator=kfold.split(boston,target)

#creating the model

model2=LogisticRegression(solver='lbfgs',max_iter=2000)

# now passing  the dataset,targetVariable, and generator which internally contains 5 folds

scores1=cross_val_score(model2,boston,target,cv=generator,scoring="r2")



print("scores of each 5 folds %s " %  scores1)



print("mean of cross-validation %s" % scores1.mean())

print("standard deviation of cross-validation %s"% scores1.std())



print("accuracy of model across each process is same, so we sucess varified the functionlaity of cross validation :) :)")



#as accuracy across fold is same, so we sucess varified the functionlaity of cross validation










