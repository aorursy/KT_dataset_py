#loading the necessary libaries for analysis 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import sklearn
import functools
from sklearn import ensemble 
from sklearn import model_selection
from functools import partial
from sklearn.model_selection import GridSearchCV
import optuna


# loading the train and test data
# here ndata is the tranin data and the tdata is the  test data 
ndata = pd.read_csv("../input/exceledited/train_orig_tita.csv")
tdata = pd.read_csv("../input/titania/test.csv")
ndata.info()


ndata['Sex'] = ndata.Sex.map({'female':0,'male':1})
tdata['Sex'] = tdata.Sex.map({'female':0,'male':1})
ndata.info()
#ndata.drop('Cabin',axis='columns', inplace=True)
tdata.head()
ndata['Age'].isna().sum()

ndata.head(2)
grouped_multiple_column = ndata.groupby(['Pclass','Sex','Survived'])['Age'].mean()
print(grouped_multiple_column)
#Pclass distribution 
print ("Upper class count-> " ,ndata['Pclass'].isin([1]).sum())
print ("Middle class count-> " ,ndata['Pclass'].isin([2]).sum())
print ("Lower class count-> " ,ndata['Pclass'].isin([3]).sum())
grouped_multiple_column = ndata.groupby(['Pclass','Sex'])['Sex'].count()
print(grouped_multiple_column)
#looking @ the result the max people that boarded the ship are 
#lwr_cls>upr_cls>mdl_cls
grouped_multiple_column = ndata.groupby(['Pclass','Sex','Survived'])['Sex'].count()
print(grouped_multiple_column)
#it gived us very intresting details of our data
#precentage of survival of upper class Female, Male
# So looking at this we can say, its a fortune to be the upper class female...
#But this also tells us that our modeling also will learn these probalities 
#and figure out a way to converge the data. The probable outcome will be chosen and the 
#survival classifier will conclude if the person would have survived /or not the titanic disaster.

#so as to say,if the a ponit taken from  test data is who is male from lower class, the max probability of it being in not-survived list will be 85%
#lets try to see their Ages and then we will conclude our Missing age data 
#for that  we will go to excel


ndata.head(3)


ndata.Name[1]


ndata.Name.head(8)
firstName = ndata.Name.str.split(".").str.get(0).str.split(",").str.get(-1)

print(firstName.value_counts())
# Replacing names to Bin them up .
firstName.replace({"Mlle":"Miss", "Ms":"Miss", "Mme":"Mirs"}, inplace = True,regex=True)
firstName.replace({"Mrs":"Mirs"}, inplace = True,regex=True)
firstName.replace(to_replace = ["Dona", "the Countess", "Lady"], value = "Nobel_f", inplace = True,regex=True)

firstName.replace(to_replace = ["Jonkheer", "Sir", "Don"], value = "Nobel_m", inplace = True,regex=True)

firstName.replace(to_replace = [ "Col", "Major", "Capt"], value = "Officer", inplace = True,regex=True)

firstName.replace(to_replace = ["Dr", "Rev"], value = "Care", inplace = True,regex=True)

ndata["NameProcessed"] = firstName


print(ndata.NameProcessed.value_counts())
ndata.info()
#ndata['Sex'] = ndata.Sex.map({'female':0,'male':1})
#ndata["NameProcessed"]= firstName_1
ndata["NameProcessed"].replace({"Mr":"0","Miss" :"1","Mirs":"2","Master":"3",'Officer':"4","Care":"5","Nobel_m":"6","Nobel_f":"7"}, inplace = True,regex=True)

#ndata["NameProcessed"] = ndata.NameProcessed.map({"Mr":"0","Miss" :"1","Mrs":"2","Master":"3",'Officer':"4","Care-Officer":"5","Nobel_m":"6","Nobel_f":"7"})
ndata.head(15)
ndata.NameProcessed.unique() 


# Create target object and call it y
y = ndata.Survived
# Create X
features = ['Age', 'Pclass', 'Sex', 'Fare','SibSp']
X = ndata[features]
#ndata['Age'] = pd.to_numeric(ndata['Age'],errors='coerce')
#ndata.Age = ndata.Age.astype(int)
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#print(scaler.fit(features))
ndata.head(3)
seed = 43


#Model selection before hyperparameter tuning 
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('RANDFRST', RandomForestClassifier(random_state = seed, n_estimators = 100)))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', DecisionTreeClassifier(random_state = seed)))
models.append(('DT', SVC()))
models.append(('GBC', GradientBoostingClassifier(random_state = seed)))
models.append(('ABC', AdaBoostClassifier(random_state = seed)))
models.append(('ETC', ExtraTreesClassifier(random_state = seed)))
models.append(('XGBC', XGBClassifier(random_state = seed)))


# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7,shuffle=True)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


"""
Optuna example that optimizes a classifier configuration using sklearn.
 We optimize the choice of classifier ( RandomForest) and their hyper parameters.
"""
# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    y = ndata.Survived
    # Create X
    features = ['Age', 'Pclass', 'Sex', 'Fare','SibSp']
    x = ndata[features]


    classifier_name = trial.suggest_categorical("classifier", [ "RandomForest"])
    if classifier_name == "RandomForest":
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        rf_entropy = trial.suggest_categorical("criterion", ["gini", "entropy"])
        rf_n_estimators=trial.suggest_int("n_estimators", 100, 1500)
        rf_max_features= trial.suggest_uniform("max_features",0.01,1.0)
        classifier_obj = sklearn.ensemble.RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=10
        )

    score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    print(study.best_trial)

clf_RFC = RandomForestClassifier(max_depth=6, random_state=0, n_estimators=236, max_features=0.626, criterion='entropy')
clf_RFC.fit(X, y)
clf_XGBC = XGBClassifier()
clf_XGBC.fit(X,y)
clf_LR=LogisticRegression()
clf_LR.fit(X,y)
clf_ETC= ExtraTreesClassifier()
clf_ETC.fit(X,y)
clf_GBC=GradientBoostingClassifier()
clf_GBC.fit(X,y)
clf_ABC=AdaBoostClassifier()
clf_ABC.fit(X,y)
tdata.info()
tdata.head(3)
tdata.Name.head(8)
firstName_1 = tdata.Name.str.split(".").str.get(0).str.split(",").str.get(-1)
print(firstName_1.value_counts())

# Replacing names to Bin them up .
firstName_1.replace({"Mlle":"Miss", "Ms":"Miss", "Mme":"Mirs"}, inplace = True,regex=True)
firstName_1.replace({"Mrs":"Mirs"}, inplace = True,regex=True)
firstName_1.replace(to_replace = ["Dona", "the Countess", "Lady"], value = "Nobel_f", inplace = True,regex=True)
firstName_1.replace(to_replace = ["Jonkheer", "Sir", "Don"], value = "Nobel_m", inplace = True,regex=True)
firstName_1.replace(to_replace = [ "Col", "Major", "Capt"], value = "Officer", inplace = True,regex=True)
firstName_1.replace(to_replace = ["Dr", "Rev"], value = "Care", inplace = True,regex=True)


tdata["NameProcessed"] = firstName_1
print(tdata.NameProcessed.value_counts())
tdata["NameProcessed"].replace({"Mr":"0","Miss" :"1","Mirs":"2","Master":"3",'Officer':"4","Care":"5","Nobel_m":"6","Nobel_f":"7"}, inplace = True,regex=True)
# Create target object and call it y
test_y = tdata.Survived
# Create X
features = ['Age', 'Pclass', 'Sex', 'Fare','SibSp','NameProcessed']
test_X = tdata[features]
pred_RF = clf_RFC.predict(test_X)
pred_XGBC = clf_XGBC.predict(test_X)
pred_ETC = clf_ETC.predict(test_X)
pred_GBC = clf_GBC.predict(test_X)
pred_LR = clf_LR.predict(test_X)
pred_ABC = clf_ABC.predict(test_X)




#print(pred_RF)
#print(pred_XGBC)
#print(pred_ETC)
#print(pred_GBC)
#print(pred_LR)
#print(pred_ABC)






arr_pred_RF= np.array(pred_RF)
arr_pred_XGBC= np.array(pred_XGBC)
arr_pred_ETC= np.array(pred_ETC)
arr_pred_GBC= np.array(pred_GBC)
arr_pred_LR= np.array(pred_LR)
arr_pred_ABC= np.array(pred_ABC)



#test_pred output to csv file 
# 1d array to list
list_RF = arr_pred_RF.tolist()
list_XGBC = arr_pred_XGBC.tolist()
list_ETC = arr_pred_ETC.tolist()
list_GBC = arr_pred_GBC.tolist()
list_LR =  arr_pred_LR.tolist()
list_ABC =  arr_pred_ABC.tolist()

print(f'List: {list_RF}')
print(f'List: {list_XGBC}')
print(f'List: {list_ETC}')
print(f'List: {list_GBC}')
print(f'List: {list_LR}')
print(f'List: {list_ABC}')
df_results = pd.DataFrame([])


df_list1 = pd.DataFrame(list_RF) 
df_list1.to_csv("./PRED_SUBMISSION.csv")
df_results.head(10)
df_results['RF'] = list_RF 
df_results['XGBC']= list_XGBC
df_results['ETC']= list_ETC 
df_results['GBC']= list_GBC
df_results['LR']= list_LR
df_results['ABC']= list_ABC

df_results.to_csv("./PRED_SUBMISSION.csv")












"""Ensembling"""
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import BaggingClassifier
from mlens.ensemble import BlendEnsemble
from vecstack import stacking
seed = (43)


"""Now initialize all the classifiers object."""
"""#1.Logistic Regression"""
lr = LogisticRegression()

"""#2.Support Vector Machines"""
svc = SVC(gamma = "auto")

"""#3.Random Forest Classifier"""
rf = RandomForestClassifier(random_state = seed, n_estimators = 100)

"""#4.KNN"""
knn = KNeighborsClassifier()

"""#5.Gaussian Naive Bayes"""
gnb = GaussianNB()

"""#6.Decision Tree Classifier"""
dt = DecisionTreeClassifier(random_state = seed)

"""#7.Gradient Boosting Classifier"""
gbc = GradientBoostingClassifier(random_state = seed)

"""#8.Adaboost Classifier"""
abc = AdaBoostClassifier(random_state = seed)

"""#9.ExtraTrees Classifier"""
etc = ExtraTreesClassifier(random_state = seed)

"""#10.Extreme Gradient Boosting"""
xgbc = XGBClassifier(random_state = seed)


"""List of all the models with their indices."""
modelNames = ["LR", "SVC", "RF", "KNN", "GNB", "DT", "GBC", "ABC", "ETC", "XGBC"]
models = [lr, svc, rf, knn, gnb, dt, gbc, abc, etc, xgbc]
"""Create a function that returns train accuracy of different models."""
def calculateTrainAccuracy(model):
    """Returns training accuracy of a model."""
    
    model.fit(X, y)
    trainAccuracy = model.score(X, y)
    trainAccuracy = round(trainAccuracy*100, 2)
    return trainAccuracy

# Calculate train accuracy of all the models and store them in a dataframe
modelScores = list(map(calculateTrainAccuracy, models))
trainAccuracy = pd.DataFrame(modelScores, columns = ["trainAccuracy"], index=modelNames)
trainAccuracySorted = trainAccuracy.sort_values(by="trainAccuracy", ascending=False)
print("Training Accuracy of the Classifiers:")
display(trainAccuracySorted)
"""Create a function that returns mean cross validation score for different models."""
def calculateXValScore(model):
    """Returns models' cross validation scores."""
    
    xValScore = cross_val_score(model, X, y, cv = 10, scoring="accuracy").mean()
    xValScore = round(xValScore*100, 2)
    return xValScore

# Calculate cross validation scores of all the models and store them in a dataframe
modelScores = list(map(calculateXValScore, models))
xValScores = pd.DataFrame(modelScores, columns = ["xValScore"], index=modelNames)
xValScoresSorted = xValScores.sort_values(by="xValScore", ascending=False)
print("Models 10-fold Cross Validation Score:")
display(xValScoresSorted)
"""Define all the models" hyperparameters one by one first::"""

"""Define hyperparameters the logistic regression will be tuned with. For LR, the following hyperparameters are usually tunned."""
lrParams = {"penalty":["l1", "l2"],
            "C": np.logspace(0, 4, 10),
            "max_iter":[5000]}

"""For GBC, the following hyperparameters are usually tunned."""
gbcParams = {"learning_rate": [0.01, 0.02, 0.05, 0.01],
              "max_depth": [4, 6, 8],
              "max_features": [1.0, 0.3, 0.1], 
              "min_samples_split": [ 2, 3, 4],
              "random_state":[seed]}

"""For SVC, the following hyperparameters are usually tunned."""
svcParams = {"C": np.arange(6,13), 
              "kernel": ["linear","rbf"],
              "gamma": [0.5, 0.2, 0.1, 0.001, 0.0001]}

"""For DT, the following hyperparameters are usually tunned."""
dtParams = {"max_features": ["auto", "sqrt", "log2"],
             "min_samples_split": np.arange(2,16), 
             "min_samples_leaf":np.arange(1,12),
             "random_state":[seed]}

"""For RF, the following hyperparameters are usually tunned."""
rfParams = {"criterion":["gini","entropy"],
             "n_estimators":[10, 15, 20, 25, 30],
             "min_samples_leaf":[1, 2, 3],
             "min_samples_split":np.arange(3,8), 
             "max_features":["sqrt", "auto", "log2"],
             "random_state":[44]}

"""For KNN, the following hyperparameters are usually tunned."""
knnParams = {"n_neighbors":np.arange(3,9),
              "leaf_size":[1, 2, 3, 5],
              "weights":["uniform", "distance"],
              "algorithm":["auto", "ball_tree","kd_tree","brute"]}

"""For ABC, the following hyperparameters are usually tunned."""
abcParams = {"n_estimators":[1, 5, 10, 15, 20, 25, 40, 50, 60, 80, 100, 130, 160, 200, 250, 300],
              "learning_rate":[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5],
              "random_state":[seed]}

"""For ETC, the following hyperparameters are usually tunned."""
etcParams = {"max_depth":[None],
              "max_features":[1, 3, 10],
              "min_samples_split":[2, 3, 10],
              "min_samples_leaf":[1, 3, 10],
              "bootstrap":[False],
              "n_estimators":[100, 300],
              "criterion":["gini"], 
              "random_state":[seed]}

"""For XGBC, the following hyperparameters are usually tunned."""
xgbcParams = {"n_estimators": (150, 250, 350, 450, 550, 650, 700, 800, 850, 1000),
              "learning_rate": (0.01, 0.6),
              "subsample": (0.3, 0.9),
              "max_depth": np.arange(3,10),
              "colsample_bytree": (0.5, 0.9),
              "min_child_weight": [1, 2, 3, 4],
              "random_state":[seed]}
"""Create a function to tune hyperparameters of the selected models."""
def tuneHyperparameters(model, params):
    """Returns best score of a model and its corresponding hyperparameters.
    model = model to be optimized.
    params = hyperparameters the models will be optimized with."""
    
    # Construct grid search object with 10 fold cross validation.
    gridSearch = GridSearchCV(model, params, verbose=0, cv=10, scoring="accuracy", n_jobs = -1)
    # Fit using grid search.
    gridSearch.fit(X, y)
    bestParams, bestScore = gridSearch.best_params_, round(gridSearch.best_score_*100, 2)
    return bestScore, bestParams
modelNamesToTune = [x for x in modelNames if x not in ["GNB","XGBC"]]
modelsToTune = [lr, svc, rf, knn, dt, gbc, abc, etc]
parametersLists = [lrParams, svcParams, rfParams, knnParams, dtParams, gbcParams, abcParams, etcParams]
bestScoreAndHyperparameters = list(map(tuneHyperparameters, modelsToTune, parametersLists))
"""Let's create a dataframe to store best score and best params."""
bestScoreAndHyperparameters = pd.DataFrame(bestScoreAndHyperparameters,
                                             index=modelNamesToTune,
                                             columns=["tunedAccuracy", "bestHyperparameters"])
bestScoreAndHyperparametersSorted = bestScoreAndHyperparameters.sort_values(by="tunedAccuracy",
                                                                                ascending=False)
print("Model's Accuracy after Tuning Hyperparameters:")
display(bestScoreAndHyperparametersSorted.iloc[:,0].to_frame())


















































