#Load some stuff.....
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
#----------------------
#1) read data and elaborate the data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
data_full = [df_train, df_test]

#for visualization of the data see below....
#----------------------------------------
#1) Feature engineering
#----------------------------------------
for df in data_full:
    #map sex to 0,1
    df["Sex"].replace(['male','female'],[0,1], inplace=True)
    
    #----------------
    #define new feature family_member
    df["FamMem"] = df["SibSp"] +df["Parch"]

    #----------------
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    #try_replacing with log....  #df["Fare"].replace(0.0, np.median(df["Fare"]), inplace=True) #df["Fare"].replace(np.log(df["Fare"]), inplace=True)
    #idea: do categoy for
    #DOES NOT IMPROVE THE PERFORMANCE
    #df['Fare_sec'] = pd.cut(df.Fare, bins=[-1,10,30,60,1000],labels=[1,2,3,4])

    #----------------
    #use port of embarkation
    df["Embarked"].fillna("S",inplace=True)
    #replace with S, the most common value
    df["Embarked"].replace(["S","C","Q"],[0,1,2],inplace=True)

    #-----------------
    # get the title of the passengers
    aa = df["Name"].str.split(",", expand = True) #with expand i creates a dataframe not a series
    bb = aa.loc[:,1].str.split(".", expand = True)
    #to take into account weird titles...
    normalized_titles = {
            ' Mr': "Mr",
            ' Miss': "Miss",
            ' Mlle': "Miss",
            ' Mrs': "Mrs",
            ' Mme': "Mrs",
            ' Ms': "Mrs",
            ' Master': "Master",
            ' Capt': "Officer",
            ' Dr': "Officer",
            ' Rev': "Officer",
            ' Major':"Officer" ,
            ' Col':"Officer",
            ' Lady': "Royalty",
            ' Jonkheer' : "Royalty",
            ' Sir' : "Royalty",
            ' the Countess'  : "Royalty",
            ' Don': "Royalty",
            ' Dona': "Royalty"
            }
    df["Title"] = bb.loc[:,0].map(normalized_titles)
    df["Title"].replace(["Mr","Miss","Mrs","Master", "Officer", "Royalty"],[0,1,2,3,4,5], inplace=True)

    #----------------
    #fill NaN of "Age" in a clever way (use the "Title mean")
    mr_age = np.nanmean(df.loc[df["Title"]==0,"Age"])
    miss_age = np.nanmean(df.loc[df["Title"]==1,"Age"])
    mrs_age = np.nanmean(df.loc[df["Title"]==2,"Age"])
    master_age = np.nanmean(df.loc[df["Title"]==3,"Age"])
    officer_age = np.nanmean(df.loc[df["Title"]==4,"Age"])
    royalty_age = np.nanmean(df.loc[df["Title"]==5,"Age"])
    title_mean = [ mr_age, miss_age, mrs_age, master_age, officer_age, royalty_age ]

    for i in range(len(df["Title"])):
        if(np.isnan(df.loc[i,"Age"])):
            df.loc[i,"Age"]=(title_mean[df.loc[i,"Title"]])
            
    #----------------
    # define a new feature "is alone"
    alone = (df["FamMem"] == 0).astype(int)
    df["isAlone"] = alone

    #----------------
    #see if the passenger has a defined cabin (0=NaN, 1=Cabin_number)
    df['def_Cabin'] = df.Cabin.notnull().astype(int)

    #-----------------
    #use first letter of cabin where NaN put 0
    df["Cabin"].fillna("NULL",inplace=True)
    df["Cabin"] = df.loc[df["Cabin"].notnull(),"Cabin"].str[0]
    df["Cabin"].replace(["N","A","B","C","D", "E", "F","G","T"],[0,1,2,3,4,5,6,7,7], inplace=True)

    #-----------------
    df['age*Pclass']= df['Age']*(1/df['Pclass'])

    #count repetition of the same ticket

    tmp1 = df.Ticket.value_counts()
    df['rep_ticket']= df['Ticket'].map(tmp1)

    df["fare/ticket"] = df["Fare"]/df["rep_ticket"]

    #"young_rich" feature
    df['Age_sec'] = pd.cut(df.Age, bins=[-1,18,25,40,60,100],labels=[1,2,3,4,5])
    for k in range(len(df["Age_sec"])):
            df.loc[k,'age*fare']= (1/df.loc[k,'Age_sec'])*df.loc[k,'fare/ticket']


###############################################################################
#----------------------------------------
#2) find best ML algorithm
#----------------------------------------
#standardize the data
x_train = df_train.loc[:,["Age", "Embarked", "Fare", "Pclass", "Sex", "Cabin", "FamMem","Title", "isAlone","def_Cabin","rep_ticket", "fare/ticket","age*Pclass", "age*fare"]]
y_train = df_train.loc[:,"Survived"]
#scaling = StandardScaler()
#x_train = scaling.fit_transform(x_train)

#-------------------
classifiers = [
        #found best parameters with gridSearch (see below
        RandomForestClassifier(bootstrap=True, max_depth=110, max_features=6, min_samples_leaf=4, min_samples_split=15, n_estimators=200)]
        #KNeighborsClassifier(3),
        #SVC(gamma = 'auto'),
        #DecisionTreeClassifier(),
        #AdaBoostClassifier(),
        #GradientBoostingClassifier()]
        #GaussianNB(),
        #LinearDiscriminantAnalysis(),
        #QuadraticDiscriminantAnalysis(),
        #LogisticRegression()]

classifier_name = [
        "RandomForestClassifier()"]
        #"KNeighborsClassifier",
        #"SVC",
        #"DecisionTreeClassifier()",
        #"AdaBoostClassifier()",
        #"GradientBoostingClassifier()"]
        #"GaussianNB()",
        #"LinearDiscriminantAnalysis()",
        #"QuadraticDiscriminantAnalysis()",
        #"LogisticRegression()"]


#evaluate performace of classifier (k-fold cross validation)
count = 0
for clf in classifiers:
    clf.fit(x_train, y_train)
    #-------------------
    kfold = StratifiedKFold(n_splits=50)
    scores_train = cross_val_score(clf,x_train, y_train,scoring = 'accuracy',  cv=kfold, n_jobs = 4, verbose = 0)
    print(classifier_name[count]," ",scores_train.mean(), " std:", scores_train.std())
    count +=1

'''
#----------
# TUNING RANDOM FOREST
#----------

rf = RandomForestClassifier()
param_grid = {
             'bootstrap': [True],
             'max_depth': [80, 90, 100, 110, 130, 150],
             'max_features': [2, 3, 4, 5 , 6],
             'min_samples_leaf': [3, 4, 5],
             'min_samples_split': [8, 10, 12, 15],
             'n_estimators': [100, 200, 300,600, 700, 800, 1000]
             }
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(x_train,y_train)
##now print the results....
print(grid_search.best_score_)
print(grid_search.best_params_)
#grid_search.best_estimator_

#best parameters!
#{'bootstrap': True,
# 'max_depth': 110,
# 'max_features': 6,
# 'min_samples_leaf': 4,
# 'min_samples_split': 15,
# 'n_estimators': 200}
'''
######################################
# choose RF as  classifier (and create submision file)
######################################
final_clf = RandomForestClassifier(bootstrap=True, max_depth=110, max_features=6, min_samples_leaf=4, min_samples_split=15, n_estimators=200)
final_clf.fit(x_train, y_train)
x_test= df_test.loc[:,["Age", "Embarked", "Fare", "Pclass", "Sex", "Cabin", "FamMem","Title", "isAlone","def_Cabin","rep_ticket", "fare/ticket","age*Pclass", "age*fare"]]
#x_test = preprocessing.scale(x_test)
#x_test = scaling.transform(x_test)
##create submission file
submission = pd.DataFrame({'PassengerId':   df_test['PassengerId'],'Survived':final_clf.predict(x_test)})
filename = 'titanic_tuned_rf_submitted.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)

'''
#now visualize the data
import matplotlib.pyplot as plt
import seaborn as sns

#see the pearson correlation matrix
df_train.corr(method='pearson')
#------------------
plt.subplot(231)
# explore Fare-Survival rate
plt.hist(df_train["Fare"],log = True, alpha = 0.3, label = "all_passenger")
plt.hist(df_train.loc[df_train["Survived"]==1,"Fare"], label = "Survived", alpha = 0.3)
plt.xlabel("Fare")
plt.legend()

#------------------
# explore Age-Survival rate
plt.subplot(232)
plt.hist(df_train["Age"],log = True, alpha = 0.3, label = "all_passenger")
plt.hist(df_train.loc[df_train["Survived"]==1,"Age"], label = "Survived", alpha = 0.3)
plt.xlabel("Age")

#------------------
# explore Parch-Survival rate
plt.subplot(233)
plt.hist(df_train["Pclass"],bins = [0.5,1.5,2.5,3.5], log = False, alpha = 0.3, label = "all_passenger")
plt.hist(df_train.loc[df_train["Survived"]==1,"Pclass"], bins = [0.5,1.5,2.5,3.5],label = "Survived", alpha = 0.3)
plt.xlabel("Pclass")

#------------------
# explore Sex-Survival rate (0=male, 1=female)
plt.subplot(234)
plt.hist(df_train["Sex"],bins = [-0.5,0.5,1.5], log = False, alpha = 0.3, label = "all_passenger")
plt.hist(df_train.loc[df_train["Survived"]==1,"Sex"], bins = [-0.5,0.5,1.5],label = "Survived", alpha = 0.3)
plt.xlabel("Sex")

#------------------
# explore FamMem-Survival rate (0=male, 1=female)
plt.subplot(235)
plt.hist(df_train["FamMem"], log = False, alpha = 0.3, label = "all_passenger")
plt.hist(df_train.loc[df_train["Survived"]==1,"FamMem"], label = "Survived", alpha = 0.3)
plt.xlabel("FamMem")
'''
