import time
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import StandardScaler
dict_title = {
    'Capt': 'Dr/Clerc/Mil',
    'Col': 'Dr/Clerc/Mil',
    'Major': 'Dr/Clerc/Mil',
    'Jonkheer': 'Honor',
    'Don': 'Honor',
    'Dona': 'Honor',
    'Sir': 'Honor',
    'Dr': 'Dr/Clerc/Mil',
    'Rev': 'Dr/Clerc/Mil',
    'the Countess': 'Honor',
    'Mme': 'Mrs',
    'Mlle': 'Miss',
    'Ms': 'Mrs',
    'Mr': 'Mr',
    'Mrs': 'Mrs',
    'Miss': 'Miss',
    'Master': 'Master',
    'Lady': 'Honor'
}

def extractTitle(df, nameCol, dictTitle):
    '''
    extractTitle(df, nameCol, dictTitle)
    Input : df : dataframe, will be copied.
            nameCol : name of the columns where to extract titles.
            dictTitle : dictionary of title and their conversion.
    This fonction extract title from a specific column with a custom dict and remove nameCol.
    '''
    
    df_new = df.copy()
    df_new["Title"] = ""
    for row in range(df_new.shape[0]):
        name = df_new.loc[row][nameCol]
        for title in dictTitle:
            if title in name:
                df_new["Title"][row] = dictTitle[title]
    return df_new.drop([nameCol], axis=1)

def getDummiesTitanic(df, dummies):
    '''
    getDummiesTitanic(df, dummies)
    Input : df : dataframe, will be copied.
            dummies : list of dummies to transform.
            dictTitle : dictionary of title and their conversion
    This fonction get dummies for a given list and drop the original column.
    '''
    df_new = df.copy()
    for dummy in dummies:
        try :
            df_new = df_new.join(pd.get_dummies(df_new[dummy], prefix = dummy))
            df_new = df_new.drop([dummy], axis=1)
        except KeyError:
            print("Warning : column {} is missing".format(dummy))
        
    return df_new

def drawConfusionMatrix(y_test, y_pred):
    '''
    drawConfusionMatrix(y_test, y_pred)
    Input : y_test : list of real target.
            y_pred : list of predicted target.

    This fonction draw a confusion matrix from y_test and y_pred.
    '''
    cf_matrix = confusion_matrix(y_test, y_pred)
    cm_sum = np.sum(cf_matrix, axis=1, keepdims=True)
    cm_perc = cf_matrix / cm_sum.astype(float) * 100
    annot = np.empty_like(cf_matrix).astype(str)
    nrows, ncols = cf_matrix.shape
    labels = ["Died", "Survived"]
    sns.heatmap(cf_matrix/np.sum(cf_matrix), 
                xticklabels=labels, 
                yticklabels=labels, 
                annot=True)
    plt.yticks(rotation=0)
    plt.ylabel('Predicted values', rotation=0)
    plt.xlabel('Actual values')
    plt.show()
df_train_org = pd.read_csv("../input/titanic/train.csv")
df_test_org = pd.read_csv("../input/titanic/test.csv")
df_train_org.dtypes
print("In the train data we have {} rows and {} columns".format(df_train_org.shape[0], df_train_org.shape[1]))
df_test_org.dtypes
print("In the test data we have {} rows and {} columns".format(df_test_org.shape[0], df_test_org.shape[1]))
df_train_org["Sex"].value_counts()
df_train_org["Embarked"].value_counts()
df_train = df_train_org.copy()
df_train = df_train.drop(["PassengerId", "Ticket"],axis=1) # Remove unique ID
df_train["SexNum"] = df_train["Sex"]
df_train["SexNum"].loc[df_train["SexNum"] == "male"] = 1
df_train["SexNum"].loc[df_train["SexNum"] == "female"] = 0

df_train["EmbarkedNum"] = df_train["Embarked"]
df_train["EmbarkedNum"] = df_train["EmbarkedNum"].fillna(0)
df_train["EmbarkedNum"].loc[df_train["EmbarkedNum"] == "S"] = 2
df_train["EmbarkedNum"].loc[df_train["EmbarkedNum"] == "C"] = 1
df_train["EmbarkedNum"].loc[df_train["EmbarkedNum"] == "Q"] = 0
df_train["EmbarkedNum"] = df_train["EmbarkedNum"].astype(int)

df_test= df_test_org.copy()
df_test= df_test.drop(["PassengerId", "Ticket"],axis=1) # Remove unique ID
df_test["SexNum"] = df_test["Sex"]
df_test["SexNum"].loc[df_test["SexNum"] == "male"] = 1
df_test["SexNum"].loc[df_test["SexNum"] == "female"] = 0

df_test["EmbarkedNum"] = df_test["Embarked"]
df_test["EmbarkedNum"] = df_test["EmbarkedNum"].fillna(0)
df_test["EmbarkedNum"].loc[df_test["EmbarkedNum"] == "S"] = 2
df_test["EmbarkedNum"].loc[df_test["EmbarkedNum"] == "C"] = 1
df_test["EmbarkedNum"].loc[df_test["EmbarkedNum"] == "Q"] = 0
df_test["EmbarkedNum"] = df_test["EmbarkedNum"].astype(int)
start_time = time.time()
plt.figure(figsize=(8,8))
sns.heatmap(df_train.corr(), annot=True, linewidths=.5, annot_kws={"size":10})
plt.show()
elapsed_time = time.time() - start_time
print("This graphic took me : {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
df_train.isna().mean()
df_test.isna().mean()
df_train_remove = df_train.drop(["Cabin", "Age", "Embarked", "Fare"], axis=1)
df_test_remove = df_test.drop(["Cabin", "Age", "Embarked", "Fare"], axis=1)
df_train["Embarked"].value_counts()
start_time = time.time()
df_train_mean = df_train.drop(["Cabin"], axis=1).copy()
df_train_mean["Age"] = df_train_mean["Age"].fillna(df_train_mean["Age"].mean())
df_train_mean["Embarked"] = df_train_mean["Embarked"].fillna("S")
df_test_mean = df_test.drop(["Cabin"], axis=1).copy()
df_test_mean["Age"] = df_test_mean["Age"].fillna(df_test_mean["Age"].mean())
df_test_mean["Fare"] = df_test_mean["Fare"].fillna(df_test_mean["Fare"].mean())
elapsed_time = time.time() - start_time
print("This calculations took me : {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
df_train_mean.isna().sum()
df_train_median = df_train.drop(["Cabin"], axis=1).copy()
df_train_median["Age"] = df_train_median["Age"].fillna(df_train_median["Age"].median())
df_train_median["Embarked"] = df_train_median["Embarked"].fillna("S")
df_test_median = df_test.drop(["Cabin"], axis=1).copy()
df_test_median["Age"] = df_test_median["Age"].fillna(df_test_median["Age"].median())
df_test_median["Fare"] = df_test_median["Fare"].fillna(df_test_median["Fare"].median())
start_time = time.time()
df_train_remove = extractTitle(df_train_remove, "Name", dict_title)
df_test_remove = extractTitle(df_test_remove, "Name", dict_title)
df_train_mean = extractTitle(df_train_mean, "Name", dict_title)
df_test_mean = extractTitle(df_test_mean, "Name", dict_title)
df_train_median = extractTitle(df_train_median, "Name", dict_title)
df_test_median = extractTitle(df_test_median, "Name", dict_title)
elapsed_time = time.time() - start_time
print("This calculations took me : {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
df_train_remove.head()
start_time = time.time()
list_dummies = ["Sex", "Embarked", "Title"]

df_train_remove = getDummiesTitanic(df_train_remove, list_dummies)
df_test_remove = getDummiesTitanic(df_test_remove, list_dummies)
df_train_mean = getDummiesTitanic(df_train_mean, list_dummies)
df_test_mean = getDummiesTitanic(df_test_mean, list_dummies)
df_train_median = getDummiesTitanic(df_train_median, list_dummies)
df_test_median = getDummiesTitanic(df_test_median, list_dummies)
elapsed_time = time.time() - start_time
print("This calculations took me : {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
df_train_remove.head()
X_train_remove, X_test_remove, y_train_remove, y_test_remove = train_test_split(df_train_remove.drop(["Survived"], axis=1), 
                                                               df_train_remove["Survived"], 
                                                               test_size=0.2,
                                                               random_state=0)

X_train_mean, X_test_mean, y_train_mean, y_test_mean = train_test_split(df_train_mean.drop(["Survived"], axis=1), 
                                                       df_train_mean["Survived"], 
                                                       test_size=0.2,
                                                       random_state=0)
X_train_median, X_test_median, y_train_median, y_test_median = train_test_split(df_train_median.drop(["Survived"], axis=1), 
                                                               df_train_median["Survived"], 
                                                               test_size=0.2,
                                                               random_state=0)


SCORES = {"Remove":{},"Mean":{},"Median":{}}
start_time = time.time()
clf = svm.SVC(kernel='linear', C = 1.0) #Check other models
clf.fit(X_train_remove, y_train_remove)
y_pred_remove = clf.predict(X_test_remove)

drawConfusionMatrix(y_test_remove, y_pred_remove)

score = (((y_pred_remove == y_test_remove).sum())/y_test_remove.shape[0])
score = round(score*100,2)
SCORES["Remove"]["SVM"] = score
print("Perfomace is : {}% for SVM_Remove".format(score))
elapsed_time = time.time() - start_time
print("This calculations took me : {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
start_time = time.time()
clf.fit(X_train_mean, y_train_mean)
y_pred_mean = clf.predict(X_test_mean)

drawConfusionMatrix(y_test_mean, y_pred_mean)

score = (((y_pred_mean == y_test_mean).sum())/y_test_mean.shape[0])
score = round(score*100,2)
SCORES["Mean"]["SVM"] = score
print("Perfomace is : {}% for SVM_Mean".format(score))
elapsed_time = time.time() - start_time
print("This calculations took me : {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
start_time = time.time()
clf.fit(X_train_median, y_train_median)
y_pred_median = clf.predict(X_test_median)

drawConfusionMatrix(y_test_median, y_pred_median)

score = (((y_pred_median == y_test_median).sum())/y_test_median.shape[0])
score = round(score*100,2)
SCORES["Median"]["SVM"] = score
print("Perfomace is : {}% for SVM_Median".format(score))
elapsed_time = time.time() - start_time
print("This calculations took me : {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
SCORES
# Dict to save best parameters
BEST_PARAMS = {"Remove":{},"Mean":{},"Median":{}}

# defining parameter range 
param_grid = [{"C": [0.1, 1, 10, 100, 1000],  
              "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
              "kernel": ["rbf", "sigmoid"]},
              {"C": [0.1, 1, 10, 100, 1000],
              "kernel": ["linear"]},
             {"C": [0.1, 1, 10, 100, 1000],  
              "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
              "kernel": ["poly"],
              "degree" : [1,2,3,4,5,6,7,8,9,10]}]
# tol and max_iter because It's taking too long to train
grid = GridSearchCV(svm.SVC(max_iter=1000000), param_grid, refit = True, verbose=3, n_jobs=-1, cv=5)
grid.fit(X_train_remove, y_train_remove) 
print("The best parameters are : {} with removed features and the score is {}".format(grid.best_params_, grid.best_score_))
score = round((grid.best_score_*100),2)
SCORES["Remove"]["SVM_BestParam"] = score
BEST_PARAMS["Remove"]["SVM"] = grid.best_params_
# defining parameter range 
param_grid = [{"C": [0.1, 1, 10, 100, 1000],  
              "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
              "kernel": ["rbf", "sigmoid"]},
              {"C": [0.1, 1, 10, 100, 1000],
              "kernel": ["linear"]},
             {"C": [0.1, 1, 10, 100, 1000],  
              "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
              "kernel": ["poly"],
              "degree" : [1,2,3,4,5,6,7,8,9,10]}]
# tol and max_iter because It's taking too long to train
grid = GridSearchCV(svm.SVC(max_iter=1000000), param_grid, refit = True, verbose=3, n_jobs=-1, cv=5)
grid.fit(X_train_mean, y_train_mean)
print("The best parameters are : {} with mean features and the score is {}".format(grid.best_params_, grid.best_score_))
score = round((grid.best_score_*100),2)
SCORES["Mean"]["SVM_BestParam"] = score
BEST_PARAMS["Mean"]["SVM"] = grid.best_params_
# defining parameter range 
param_grid = [{"C": [0.1, 1, 10, 100, 1000],  
              "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
              "kernel": ["rbf", "sigmoid"]},
              {"C": [0.1, 1, 10, 100, 1000],
              "kernel": ["linear"]},
             {"C": [0.1, 1, 10, 100, 1000],  
              "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
              "kernel": ["poly"],
              "degree" : [1,2,3,4,5,6,7,8,9,10]}]
# tol and max_iter because It's taking too long to train
grid = GridSearchCV(svm.SVC(max_iter=1000000), param_grid, refit = True, verbose=3, n_jobs=-1, cv=5)
grid.fit(X_train_median, y_train_median)
print("The best parameters are : {} with median featuresand the score is {}".format(grid.best_params_, grid.best_score_))
score = round((grid.best_score_*100),2)
SCORES["Median"]["SVM_BestParam"] = score
BEST_PARAMS["Median"]["SVM"] = grid.best_params_
SCORES
BEST_PARAMS
pd.DataFrame(SCORES).plot(kind='bar')
plt.ylim(0, 120)
plt.ylabel('Precision')
plt.show()
knn = KNeighborsClassifier(n_neighbors=3)
print("Train/Test/Record for df_train_remove")
knn.fit(X_train_remove, y_train_remove)
y_pred_remove = knn.predict(X_test_remove)
print(confusion_matrix(y_test_remove, y_pred_remove))
score = (((y_pred_remove == y_test_remove).sum())/y_test_remove.shape[0])
score = round(score*100,2)
SCORES["Remove"]["KNN_3"] = score
print("Performace is : {}% for KNN_3_Remove".format(score))
print("Train/Test/Record for df_train_mean")
knn.fit(X_train_mean, y_train_mean)
y_pred_mean = knn.predict(X_test_mean)
print(confusion_matrix(y_test_mean, y_pred_mean))
score = (((y_pred_mean == y_test_mean).sum())/y_test_mean.shape[0])
score = round(score*100,2)
SCORES["Mean"]["KNN_3"] = score
print("Performace is : {}% for KNN_3_Mean".format(score))
print("Train/Test/Record for df_train_median")
knn.fit(X_train_median, y_train_median)
y_pred_median = knn.predict(X_test_median)
print(confusion_matrix(y_test_median, y_pred_median))
score = (((y_pred_median == y_test_median).sum())/y_test_median.shape[0])
score = round(score*100,2)
SCORES["Median"]["KNN_3"] = score
print("Performace is : {}% for KNN_3_Median".format(score))
SCORES
# defining parameter range 
param_grid = [{"n_neighbors": range(1,101),  
              "weights": ["uniform", "distance"],
              "algorithm": ["auto", "brute"],
              "p" : [1,2]},
             {"n_neighbors": range(1,101),  
              "weights": ["uniform", "distance"],
              "algorithm": ["ball_tree", "kd_tree"],
              "p" : [1,2],
             "leaf_size": [1,2,3,4,5,10,15,20,25,30]}]

# With Remove features
grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit = True, verbose=3, n_jobs=-1, cv=5)
grid.fit(X_train_remove, y_train_remove) 
print("The best parameters are : {} with remove features and the score is {}".format(grid.best_params_, grid.best_score_))
score = round((grid.best_score_*100),2)
SCORES["Remove"]["KNN_BestParam"] = score
BEST_PARAMS["Remove"]["KNN"] = grid.best_params_

# With Mean features
grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit = True, verbose=3, n_jobs=-1, cv=5)
grid.fit(X_train_mean, y_train_mean)
print("The best parameters are : {} with mean features and the score is {}".format(grid.best_params_, grid.best_score_))
score = round((grid.best_score_*100),2)
SCORES["Mean"]["KNN_BestParam"] = score
BEST_PARAMS["Mean"]["KNN"] = grid.best_params_

# With Median features
grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit = True, verbose=3, n_jobs=-1, cv=5)
grid.fit(X_train_median, y_train_median)
print("The best parameters are : {} with mean features and the score is {}".format(grid.best_params_, grid.best_score_))
score = round((grid.best_score_*100),2)
SCORES["Median"]["KNN_BestParam"] = score
BEST_PARAMS["Median"]["KNN"] = grid.best_params_
pd.DataFrame(SCORES).plot(kind='bar')
plt.ylim(0, 120)
plt.ylabel('Precision')
plt.show()
SCORES
X_train = df_train_remove.drop(["Survived"], axis=1)
Y_train = df_train_remove["Survived"]
X_test = df_test_remove
BEST_PARAMS["Remove"]["SVM"]
start_time = time.time()
clf = svm.SVC(C = 1, gamma = 0.1, kernel = 'rbf') #Use best params
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
# Creation of the submission file :
DF_Fin = pd.DataFrame(columns=["PassengerId","Survived"])
DF_Fin["PassengerId"] = df_test_org["PassengerId"]
DF_Fin["Survived"] = y_pred

DF_Fin.head()
