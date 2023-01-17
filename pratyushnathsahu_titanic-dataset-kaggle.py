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
train_df=pd.read_csv("../input/train.csv")

test_df=pd.read_csv("../input/test.csv")
print(train_df.info())

print("--"*20)

print(test_df.info())
X=train_df.drop("Survived",axis=1)

Y=train_df["Survived"]
combined_df=pd.concat([X,test_df])
print(combined_df.info())
from sklearn.impute import SimpleImputer



Age_imputer=SimpleImputer(strategy="median")

Fare_imputer=SimpleImputer(strategy="median")

Embarked_imputer=SimpleImputer(strategy="most_frequent")





combined_df["Age"]=Age_imputer.fit_transform(combined_df[["Age"]])

combined_df["Fare"]=Age_imputer.fit_transform(combined_df[["Fare"]])

combined_df["Embarked"]=Embarked_imputer.fit_transform(combined_df[["Embarked"]])
combined_df.drop(["Cabin"],axis=1,inplace=True)
combined_df["Family_size"]=combined_df["SibSp"]+combined_df["Parch"]+1

combined_df["Is_alone"]=int(0)

combined_df["Is_alone"][combined_df["Family_size"]==1]=int(1)
combined_df['Title']=combined_df['Name'].str.split(",", expand=True)[1].str.split(".", expand=True)[0]
combined_df["Fare_bin"]=pd.qcut(combined_df["Fare"],4)

combined_df["Age_bin"]=pd.cut(combined_df["Age"].astype(int),4)
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()





combined_df["Sex_encode"]=encoder.fit_transform(combined_df["Sex"])

combined_df["Age_encode"]=encoder.fit_transform(combined_df["Age_bin"])

combined_df["Fare_encode"]=encoder.fit_transform(combined_df["Fare_bin"])

combined_df["Title_encode"]=encoder.fit_transform(combined_df["Title"])

combined_df["Embarked_encode"]=encoder.fit_transform(combined_df["Embarked"])
combined_df.head()
# Features to be used

df_final=combined_df[["Pclass","Family_size","Is_alone","Sex_encode","Age_encode","Fare_encode","Embarked_encode"]]

X_train=df_final[0:891]

X_test=df_final[891:]
X_train.head()
from sklearn import ensemble, neural_network , neighbors , svm , linear_model ,naive_bayes ,tree , gaussian_process,discriminant_analysis

from xgboost import XGBClassifier, XGBRFClassifier
#Machine Learning Algorithms

MLA=[

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),



    #Gaussian Processes

    gaussian_process.GaussianProcessClassifier(),

    

    #GLM

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    #Navies Bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    neighbors.KNeighborsClassifier(),

    

    #SVM

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    

    #Trees    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),



    #Neural Network

    neural_network.MLPClassifier(),

    

    #xgboost

    XGBClassifier(),

    XGBRFClassifier()

]
MLA_compare=["Algorithm_Name","Parameters", "Mean_Accuracy","Standard_Deviation"]

MLA_compare=pd.DataFrame(columns=MLA_compare)
from sklearn.model_selection import cross_val_score



index=0

for algo in MLA:

    clf=algo

    clf.fit(X_train,Y)

    cv=cross_val_score(estimator=algo,X=X_train,y=Y,cv=10,n_jobs=-1)

    cv_mean=cv.mean()

    cv_std=cv.std()

    MLA_compare.loc[index,"Algorithm_Name"]=algo.__class__.__name__

    MLA_compare.loc[index,"Parameters"]=str(algo.get_params())

    MLA_compare.loc[index,"Mean_Accuracy"]=cv_mean

    MLA_compare.loc[index,"Standard_Deviation"]=cv_std

    index+=1
MLA_compare
selected_model=svm.NuSVC(probability=True)

selected_model.fit(X_train,Y)
# Parameters=[{"base_score":[0.1,0.2,0.3,0.5,0.6,0.9,1,2],

#              "gamma":[0,0.1,0.2,0.3,0.4,0.5,1],

#              "learning_rate":[0,0.1,0.2,0.3,0.4,1],

#             }]
# from sklearn.model_selection import GridSearchCV

# grid=GridSearchCV(estimator=selected_model,param_grid=Parameters,n_jobs=-1,cv=10,scoring="accuracy")

# result=grid.fit(X=X_train,y=Y)
# best_accuracy=result.best_score_

# best_parameters=result.best_params_
y_pred=selected_model.predict(X_test)

y_pred=pd.DataFrame({"PassengerId":test_df["PassengerId"],"Survived":y_pred})

y_pred.to_csv("Submit.csv",index=False)
test_df