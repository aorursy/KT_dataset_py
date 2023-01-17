### Changelog:

### RESOURCES:
# (I) - https://github.com/savarin/pyconuk-introtutorial/tree/master/notebooks
# (II) - http://scikit-learn.org/stable/modules/preprocessing.html#label-encoding

#0.a - IMPORT libraries and read train and test set:
import numpy as np
import pandas as pd
from sklearn import preprocessing
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, ) #is a panda df
test  = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )  #is a panda df


#0.b - HELPER FUNCTION TO HANDLE MISSING DATA 
def harmonize_data(titanic):
    #assumptions
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())
    titanic.drop("Cabin",axis=1,inplace=True)
    titanic.drop("Ticket",axis=1,inplace=True)
    titanic.drop("Name",axis=1,inplace=True)    
    titanic["Embarked"] = titanic["Embarked"].fillna("S")#fill the two missing values with the most occurred value, which is "S".

    #clean features
    titanic.dropna(axis=0, how='any', subset=['Age'] , inplace=True)
    titanic['Gender'] = titanic['Sex'].map({'female': 0, 'male':1}).astype(int)
    titanic['Port'] = titanic['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int)
    titanic = titanic.drop(['Sex', 'Embarked'], axis=1)
    return titanic

def prepare_regression(titanic): #create dummy variables for logistic regression+performs feature scaling
    titanic = pd.concat([titanic, pd.get_dummies(titanic['Embarked'], prefix='Embarked')], axis=1)
    titanic.drop("Embarked",axis=1,inplace=True)
    #scale Fare adn Age 
    titanic['Fare_s'] = preprocessing.scale(titanic['Fare'].values.astype(float)) 
    titanic.drop("Fare",axis=1,inplace=True)
    titanic['Age_s'] = preprocessing.scale(titanic['Age'].values.astype(float)) 
    #titanic.drop("Age",axis=1,inplace=True)
    
    #titanic = pd.concat([titanic pd.DataFrame(preprocessing.scale(titanic['Fare'].values.astype(float))) #pd.DataFrame
    return titanic
def create_submission(alg, train, test, predictors, filename):

    alg.fit(train[predictors], train["Survived"])
    predictions = alg.predict(test[predictors])

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions})
    submission.to_csv(filename, index=False)


#1.a - CLEANING DATA:
print("---------------- ORIGINAL TRAIN DF")
print(train.head(2)) #display some of the DF
print('Size of the original train DF: ', train.shape)


train_data = harmonize_data(train) #notice this is a copy/view: ORIGINAL DATA is no longer available!
train_data = prepare_regression(train) 
test_data  = harmonize_data(test)
test_data  = prepare_regression(test)
print('Size of the harmonized train DF: ', train_data.shape)
print("---------------- HARMONIZED + DUMMY TRAIN DF")
print(train_data.head(4)) #Notice that "Cabin" has been removed



from sklearn import linear_model
from sklearn import cross_validation

c=0.8 #1 means no regularization

algLogistic = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, 
                                             C=c, fit_intercept=True, 
                                             intercept_scaling=1, class_weight=None, 
                                             random_state=None, solver='liblinear', 
                                             max_iter=100, multi_class='ovr', verbose=0, 
                                             warm_start=False, n_jobs=1)
predictors = ["Pclass", "Gender", "Age", "SibSp","Embarked_C","Embarked_S","Embarked_Q","Fare_s"] 
print(predictors)
create_submission(algLogistic, train_data, test_data, predictors, "run-01-logisticRegression.csv")

scores = cross_validation.cross_val_score(algLogistic,
                                          train_data[predictors],
                                          train_data["Survived"],
                                          cv=10)
print('Logistic Regression score with basic features: ', scores.mean())

import matplotlib.pyplot as plt
# Plot the feature coefficients of the Logistic Regression
importances = algLogistic.coef_
importances = importances.transpose()

#print(importances)
indices = np.argsort(importances,0)

plt.figure()
plt.title("Coefficients")
plt.bar(range(len(predictors)), importances[indices],
       color="r", align="center")
plt.xticks(range(len(predictors)), predictors)
plt.xlim([-1, len(predictors)])
plt.show()


#TRY TO SPLIT AGES INTO BINS:

def create_bin(titanic): #create dummy variables for logistic regression+performs feature scaling
    titanic['AgeBin'] = pd.cut(titanic['Age'], bins=[0, 14, 30, 50, 100], labels=False)
    titanic = pd.concat([titanic, pd.get_dummies(titanic['AgeBin'], prefix='agebins')], axis=1)
    titanic.drop("AgeBin",axis=1,inplace=True)
    return titanic
train_data = create_bin(train_data) 
test_data = create_bin(test_data) 
predictors = ["Pclass", "Gender", "agebins_0","agebins_1","agebins_2","agebins_3", "SibSp","Embarked_C","Embarked_S","Embarked_Q","Fare_s"] 
train_data.head(5)

create_submission(algLogistic, train_data, test_data, predictors, "run-01-logisticRegressionAgeBins.csv")
scores = cross_validation.cross_val_score(algLogistic,
                                          train_data[predictors],
                                          train_data["Survived"],
                                          cv=10)
print('Logistic Regression score with basic features: ', scores.mean())

#TRY ADDING SOME POLY FEATURES
from sklearn.preprocessing import PolynomialFeatures
#combineFeat = ["Fare_s", "Age_s"]
#poly = PolynomialFeatures(2)
#polyFeat = poly.fit_transform(train_data[combineFeat]) 
#addPoly = pd.DataFrame({'p0':polyFeat[:,0],'p1':polyFeat[:,1],'p2':polyFeat[:,2],'p3':polyFeat[:,4],'p4':polyFeat[:,4],'p5':polyFeat[:,5]})
#print(train_data.shape)
#print(addPoly.shape)
#train_data = pd.concat([train_data, addPoly], axis=1)
def add_poly_feat(titanic):
    combineFeat = ["Fare_s", "Age_s"]
    poly = PolynomialFeatures(2)
    polyFeat = poly.fit_transform(titanic[combineFeat]) 
    addPoly = pd.DataFrame({'p0':polyFeat[:,0],'p1':polyFeat[:,1],'p2':polyFeat[:,2],'p3':polyFeat[:,4],'p4':polyFeat[:,4],'p5':polyFeat[:,5]})
    titanic = pd.concat([titanic, addPoly], axis=1)
    return titanic

#print(prova.shape[1])
#for k in range(prova.shape[1]):
#    print(prova[:,k].shape)
#train_data = pd.concat([train_data, pd.Series(prova[:,2])], axis=1)
#train_data.head()







#let's see what happens now:
train_data = add_poly_feat(train_data) 
test_data  = add_poly_feat(test_data)






# Feature Importance
from sklearn.feature_selection import RFE
predictors = ["Pclass", "Gender", "agebins_0","agebins_1","agebins_2","agebins_3", "SibSp","Embarked_C","Embarked_S","Embarked_Q","Fare_s","p0","p1","p2","p3","p4","p5"] 
#
est = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, 
                                             C=c, fit_intercept=True, 
                                             intercept_scaling=1, class_weight=None, 
                                             random_state=None, solver='liblinear', 
                                             max_iter=100, multi_class='ovr', verbose=0, 
                                             warm_start=False, n_jobs=1)
# create the RFE model and select 3 attributes
rfe = RFE(est, 4)
rfe = rfe.fit(train_data[predictors], train_data["Survived"])
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)

predictors = ["Pclass", "Gender","p0","p2"] 
create_submission(algLogistic, train_data, test_data, predictors, "run-01-logisticPoly.csv")
scores = cross_validation.cross_val_score(algLogistic,
                                          train_data[predictors],
                                          train_data["Survived"],
                                          cv=10)

print('Logistic Regression score with SELECTED features: ', scores.mean())


from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score

#clf = AdaBoostClassifier(n_estimators=300)
predictors = ["Pclass", "Gender","p0","p2","agebins_0"] 
#predictors = ["Pclass", "Gender", "agebins_0","agebins_1","agebins_2","agebins_3", "SibSp","Embarked_C","Embarked_S","Embarked_Q","Fare_s","p0","p1","p2","p3","p4","p5"] 
#scores = cross_val_score(clf, train_data[predictors], train_data["Survived"])
#scores.mean()

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
        max_depth=1, random_state=0).fit(train_data[predictors], train_data["Survived"])
clf.score(train_data[predictors], train_data["Survived"])     
#THIS PERFORMS VERY BADLY. PROBABLY OVERFITTING??

create_submission(clf, train_data, test_data, predictors, "run-01-gradientBoost.csv")

