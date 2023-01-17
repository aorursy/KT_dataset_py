# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

plt.style.use("seaborn-whitegrid")       

import pandas_profiling as pp 



import seaborn as sns



from collections import Counter



import warnings

warnings.filterwarnings("ignore")



%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId = test_df["PassengerId"]
train_df.head()
train_df.shape
train_df.describe().T
train_df.info()


profile_report = pp.ProfileReport(train_df)
profile_report
train_df.isnull().sum()
#Cabin 



train_df.drop("Cabin", axis = 1, inplace = True)

#Embarked



train_df[train_df["Embarked"].isnull()]
train_df.boxplot(column="Fare",by = "Embarked")

plt.show()

train_df["Embarked"] = train_df["Embarked"].fillna("C")

#Age



name = train_df["Name"]

train_df["Name_Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]
train_df["Name_Title"].value_counts()
train_df['Name_Title'].replace( ['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Jonkheer',

'Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other',

'Other','Mr','Mr','Mr'],inplace=True)
sns.countplot(x="Name_Title", data = train_df);

plt.xticks(rotation = 90);

train_df.groupby('Name_Title')['Age'].mean()
train_df.loc[(train_df["Age"].isnull())&(train_df["Name_Title"]=='Mr'),'Age']=33

train_df.loc[(train_df["Age"].isnull())&(train_df["Name_Title"]=='Mrs'),'Age']=36

train_df.loc[(train_df["Age"].isnull())&(train_df["Name_Title"]=='Master'),'Age']=5

train_df.loc[(train_df["Age"].isnull())&(train_df["Name_Title"]=='Miss'),'Age']=22

train_df.loc[(train_df["Age"].isnull())&(train_df["Name_Title"]=='Other'),'Age']=46
train_df.isnull().sum()
test_df.isnull().sum()
#Cabin



test_df.drop("Cabin", axis = 1, inplace = True)
#Fare



test_df[test_df["Fare"].isnull()]
test_df[["Embarked","Fare"]].groupby(["Embarked"],as_index = False).mean() 
test_df["Fare"].fillna(66, inplace = True)
#Age



name = test_df["Name"]

test_df["Name_Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]
test_df['Name_Title'].replace( ['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer',

'Col','Rev','Capt','Sir','Don','Dona'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other',

'Other','Mr','Mr','Mr','Other'],inplace=True)
test_df.groupby('Name_Title')['Age'].mean()
test_df.loc[(test_df["Age"].isnull())&(test_df["Name_Title"]=='Mr'),'Age']=32

test_df.loc[(test_df["Age"].isnull())&(test_df["Name_Title"]=='Mrs'),'Age']=38

test_df.loc[(test_df["Age"].isnull())&(test_df["Name_Title"]=='Master'),'Age']=7

test_df.loc[(test_df["Age"].isnull())&(test_df["Name_Title"]=='Miss'),'Age']=21

test_df.loc[(test_df["Age"].isnull())&(test_df["Name_Title"]=='Other'),'Age']=42
test_df.isnull().sum()
#Pclass - Survived

sns.barplot(train_df["Pclass"], train_df["Survived"]);
train_df[["Pclass","Survived"]].groupby(["Pclass"],

as_index = False).mean().sort_values(by="Survived",ascending = False)
#Sex - Survived

sns.barplot(train_df["Sex"], train_df["Survived"]);
train_df[["Sex","Survived"]].groupby(["Sex"],

as_index = False).mean().sort_values(by="Survived",ascending = False)
#SibSp - Survived

sns.barplot(train_df["SibSp"], train_df["Survived"]);

train_df[["SibSp","Survived"]].groupby(["SibSp"],

as_index = False).mean().sort_values(by="Survived",ascending = False)
#Parch - Survived

sns.barplot(train_df["Parch"], train_df["Survived"]);

train_df[["Parch","Survived"]].groupby(["Parch"],

as_index = False).mean().sort_values(by="Survived",ascending = False)
#Age - Survived



g = sns.FacetGrid(train_df, col = "Survived")

g.map(sns.distplot, "Age", bins = 25)

plt.show()
#Pclass - Survived - Age



g = sns.FacetGrid(train_df, col = "Survived", row = "Pclass", size = 2)

g.map(plt.hist, "Age", bins = 25)

g.add_legend()

plt.show()
#Embarked - Sex - Pclass - Survived



g = sns.FacetGrid(train_df, row = "Embarked", size = 2)

g.map(sns.pointplot, "Pclass","Survived","Sex")

g.add_legend()

plt.show()
#Embarked - Sex - Fare - Survived



g = sns.FacetGrid(train_df, row = "Embarked", col = "Survived", size = 2.3)

g.map(sns.barplot, "Sex", "Fare")

g.add_legend()

plt.show()
#HeatMap

sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Fare","Pclass","Survived"]].corr(), annot = True)

plt.show()
train_df.head(2)
test_df.head(2)
train_test = [train_df, test_df]
train_df['Age_Band'] = pd.cut(train_df['Age'], 5)

train_df[['Age_Band', 'Survived']].groupby(['Age_Band'], 

        as_index=False).mean().sort_values(by='Age_Band', ascending=True)
for dataset in train_test:

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

train_df.head()
for dataset in train_test:

    dataset['Sex'].replace(['male','female'],[0,1],inplace=True)

    dataset['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

    dataset['Name_Title'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)



train_df.head()


for dataset in train_test:

    dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1



print (train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
for dataset in train_test:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    

print (train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
train_df['Fare_Band'] = pd.qcut(train_df['Fare'], 4)

print (train_df[['Fare_Band', 'Survived']].groupby(['Fare_Band'], as_index=False).mean())

train_df.head()
for dataset in train_test:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

train_df.head(2)
test_df.head(2)
train_df.drop(["Name", "SibSp", "Parch", "Ticket", "PassengerId", "Age_Band", "Fare_Band"], axis = 1, inplace = True)

test_df.drop(["Name","SibSp", "Parch","Ticket","PassengerId"], axis = 1, inplace = True)
train_df.head(2)
test_df.head(2)
sns.heatmap(train_df.corr(),annot=True,linewidths=0.2)



plt.show()
train = train_df

train.to_csv("titanic_train.csv", index = False)



test = test_df

test.to_csv("titanic_test.csv", index = False)
train_df.head()
train_df1 = train_df.copy()

test_df1 = test_df.copy()



train_test1 = [train_df1,test_df1]
for dataset in train_test1:

    dataset["Pclass"] = dataset["Pclass"].astype("category")

    dataset["Sex"] = dataset["Sex"].astype("category")

    dataset["Fare"] = dataset["Fare"].astype("category")

    dataset["Embarked"] = dataset["Embarked"].astype("category")

    dataset["Name_Title"] = dataset["Name_Title"].astype("category")

    dataset["Age"] = dataset["Age"].astype("category")

    dataset["IsAlone"] = dataset["IsAlone"].astype("category")

    dataset["FamilySize"] = dataset["FamilySize"].astype("category")
#train_df1=pd.get_dummies(train_df1,drop_first=True)



train_df1 = pd.get_dummies(train_df1, columns=["Sex"])

train_df1 = pd.get_dummies(train_df1, columns=["Pclass"])

train_df1 = pd.get_dummies(train_df1, columns=["Fare"])

train_df1 = pd.get_dummies(train_df1, columns=["Name_Title"])

train_df1 = pd.get_dummies(train_df1, columns=["Age"])

train_df1 = pd.get_dummies(train_df1, columns=["FamilySize"])

train_df1 = pd.get_dummies(train_df1, columns=["IsAlone"])

train_df1 = pd.get_dummies(train_df1, columns=["Embarked"])
#test_df1=pd.get_dummies(test_df1,drop_first=True)



test_df1 = pd.get_dummies(test_df1, columns=["Sex"])

test_df1 = pd.get_dummies(test_df1, columns=["Pclass"])

test_df1 = pd.get_dummies(test_df1, columns=["Fare"])

test_df1 = pd.get_dummies(test_df1, columns=["Name_Title"])

test_df1 = pd.get_dummies(test_df1, columns=["Age"])

test_df1 = pd.get_dummies(test_df1, columns=["FamilySize"])

test_df1 = pd.get_dummies(test_df1, columns=["IsAlone"])

test_df1 = pd.get_dummies(test_df1, columns=["Embarked"])
train_df1.head()
test_df1.head()
X_train = train_df.drop(["Survived"], axis = 1)

y_train = train_df["Survived"]

X_test = test_df



X_train1 = train_df1.drop(["Survived"], axis = 1)

y_train1 = train_df1["Survived"]

X_test1 = test_df1
X_train.head()
X_train1.head()
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.linear_model import LogisticRegression

log = LogisticRegression(solver = "liblinear")

log_model = log.fit(X_train,y_train)

log_model
confusion_matrix(y_train, log_model.predict(X_train))

print(classification_report(y_train, log_model.predict(X_train)))
accuracy_score(y_train, log_model.predict(X_train))

cross_val_score(log_model, X_train, y_train, cv = 10).mean()
logit_roc_auc = roc_auc_score(y_train, log_model.predict(X_train))



fpr, tpr, thresholds = roc_curve(y_train, log_model.predict_proba(X_train)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive ')

plt.ylabel('True Positive ')

plt.title('ROC')

plt.show()
from sklearn.naive_bayes import GaussianNB





nb = GaussianNB()

nb_model = nb.fit(X_train, y_train)

nb_model
accuracy_score(y_train, nb_model.predict(X_train))
cross_val_score(nb_model, X_train, nb_model.predict(X_train), cv = 10).mean()
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn_model = knn.fit(X_train, y_train)

knn_model


accuracy_score(y_train, knn_model.predict(X_train))
knn_params = {"n_neighbors": np.arange(1,20)}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, knn_params, cv=10)

knn_cv.fit(X_train, y_train)
print("Best KNN score:" + str(knn_cv.best_score_))

print("Best KNN parameter: " + str(knn_cv.best_params_))
knn = KNeighborsClassifier(10)

knn_tuned = knn.fit(X_train, y_train)


accuracy_score(y_train, knn_tuned.predict(X_train))
d = {'Accuracy in KNN before GridSearchCV ': [0.84], 'Accuracy in KNN After GridSearchCV': [0.84]}

knn_data = pd.DataFrame(data=d)

knn_data
from sklearn.svm import SVC





svm_model = SVC(kernel = "rbf").fit(X_train, y_train)



accuracy_score(y_train, svm_model.predict(X_train))
svc_params = {"C": [0.0001, 0.001, 0.1, 1, 5, 10 ,50 ,100],

             "gamma": [0.0001, 0.001, 0.1, 1, 5, 10 ,50 ,100]}



svc = SVC()

svc_cv_model = GridSearchCV(svc, svc_params, 

                         cv = 10, 

                         n_jobs = -1,

                         verbose = 2)



svc_cv_model.fit(X_train, y_train)
print("Best Params: " + str(svc_cv_model.best_params_))
svc_tuned = SVC(C = 10, gamma = 0.1).fit(X_train, y_train)



accuracy_score(y_train, svc_tuned.predict(X_train))
d = {'Accuracy in SVM before GridSearchCV ': [0.83], 'Accuracy in SVM After GridSearchCV': [0.85]}

svm_data = pd.DataFrame(data=d)

svm_data
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier().fit(X_train, y_train)



accuracy_score(y_train, rf_model.predict(X_train))
rf_params = {"max_depth": [2,5,8],

            "max_features": [2,5,8],

            "n_estimators": [10,500,1000],

            "min_samples_split": [2,5,10]}



rf_model = RandomForestClassifier()



rf_cv_model = GridSearchCV(rf_model, 

                           rf_params, 

                           cv = 10, 

                           n_jobs = -1, 

                           verbose = 2) 



rf_cv_model.fit(X_train, y_train)
print("Best Params: " + str(rf_cv_model.best_params_))
rf_tuned = RandomForestClassifier(max_depth = 5, 

                                  max_features = 2, 

                                  min_samples_split = 2,

                                  n_estimators = 1000)
rf_tuned.fit(X_train, y_train)



accuracy_score(y_train, rf_tuned.predict(X_train))
confusion_matrix(y_train, rf_tuned.predict(X_train))

print(classification_report(y_train, rf_tuned.predict(X_train)))
Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_*100},

                         index = X_train.columns)



Importance.sort_values(by = "Importance", 

                       axis = 0, 

                       ascending = True).plot(kind ="barh", color = "r");
d = {'Accuracy in RF before GridSearchCV ': [0.88], 'Accuracy in RF After GridSearchCV': [0.83]}

rf_data = pd.DataFrame(data=d)

rf_data
from sklearn.ensemble import GradientBoostingClassifier



gbm_model = GradientBoostingClassifier().fit(X_train, y_train)



accuracy_score(y_train, gbm_model.predict(X_train))
gbm_params = {"learning_rate" : [0.001, 0.01, 0.1, 0.05],

             "n_estimators": [100,500,100],

             "max_depth": [3,5,10],

             "min_samples_split": [2,5,10]}



gbm = GradientBoostingClassifier()



gbm_cv = GridSearchCV(gbm, gbm_params, cv = 10, n_jobs = -1, verbose = 2)

gbm_cv.fit(X_train, y_train)
print("Best Params: " + str(gbm_cv.best_params_))
gbm = GradientBoostingClassifier(learning_rate = 0.01, 

                                 max_depth = 3,

                                min_samples_split = 10,

                                n_estimators = 500)



gbm_tuned =  gbm.fit(X_train,y_train)


accuracy_score(y_train, gbm_tuned.predict(X_train))
confusion_matrix(y_train, gbm_tuned.predict(X_train))

print(classification_report(y_train, gbm_tuned.predict(X_train)))
Importance = pd.DataFrame({"Importance": gbm_tuned.feature_importances_*100},

                         index = X_train.columns)



Importance.sort_values(by = "Importance", 

                       axis = 0, 

                       ascending = True).plot(kind ="barh", color = "r");
d = {'Accuracy in GBM before GridSearchCV ': [0.8473], 'Accuracy in GBM After GridSearchCV': [0.8451]}

gbm_data = pd.DataFrame(data=d)

gbm_data
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier().fit(X_train, y_train)





accuracy_score(y_train, cat_model.predict(X_train))
catb_params = {

    'iterations': [200,500],

    'learning_rate': [0.01,0.05, 0.1],

    'depth': [3,5,8] }



catb = CatBoostClassifier()

catb_cv_model = GridSearchCV(catb, catb_params, cv=5, n_jobs = -1, verbose = 2)

catb_cv_model.fit(X_train, y_train)
catb_cv_model.best_params_

catb = CatBoostClassifier(iterations = 200, 

                          learning_rate = 0.01, 

                          depth = 5)



catb_tuned = catb.fit(X_train, y_train)



accuracy_score(y_train, catb_tuned.predict(X_train))
models = [

    knn_tuned,

    log_model,

    svc_tuned,

    nb_model,

    rf_tuned,

    gbm_tuned,

    catb_tuned,

    

]





for model in models:

    name = model.__class__.__name__

    y_pred = model.predict(X_train)

    accuracy = accuracy_score(y_train, y_pred)

    print("-"*28)

    print(name + ":" )

    print("Accuracy: {:.4%}".format(accuracy))
result = []



results = pd.DataFrame(columns= ["Models","Accuracy"])



for model in models:

    name = model.__class__.__name__

    y_pred = model.predict(X_train)

    accuracy = accuracy_score(y_train, y_pred)    

    result = pd.DataFrame([[name, accuracy*100]], columns= ["Models","Accuracy"])

    results = results.append(result)

    

    

sns.barplot(x= 'Accuracy', y = 'Models', data=results, color="r")

plt.xlabel('Accuracy %')

plt.title('accuracy rate of models'); 
log = LogisticRegression(solver = "liblinear")

log_model_onehot = log.fit(X_train1,y_train1)

log_model_onehot


print(classification_report(y_train1, log_model_onehot.predict(X_train1)))
accuracy_score(y_train, log_model_onehot.predict(X_train1))

cross_val_score(log_model, X_train1, y_train1, cv = 10).mean()
nb = GaussianNB()

nb_model_onehot = nb.fit(X_train1, y_train1)

nb_model_onehot
accuracy_score(y_train1, nb_model_onehot.predict(X_train1))
cross_val_score(nb_model_onehot, X_train1, nb_model_onehot.predict(X_train1), cv = 10).mean()
knn = KNeighborsClassifier()

knn_model_onehot = knn.fit(X_train1, y_train1)

knn_model_onehot
accuracy_score(y_train1, knn_model_onehot.predict(X_train1))
knn_params = {"n_neighbors": np.arange(1,20)}

knn = KNeighborsClassifier()

knn_cv_onehot = GridSearchCV(knn, knn_params, cv=10)

knn_cv_onehot.fit(X_train1, y_train1)
print("Best KNN parameter: " + str(knn_cv_onehot.best_params_))
knn = KNeighborsClassifier(10)

knn_tuned_onehot = knn.fit(X_train1, y_train1)

accuracy_score(y_train1, knn_tuned_onehot.predict(X_train1))
d = {'Accuracy in KNN before GridSearchCV ': [0.81], 'Accuracy in KNN After GridSearchCV': [0.82]}

knn_data = pd.DataFrame(data=d)

knn_data
from sklearn.svm import SVC





svm_model_onehot = SVC(kernel = "rbf").fit(X_train1, y_train1)



accuracy_score(y_train1, svm_model_onehot.predict(X_train1))
svc_params = {"C": [0.0001, 0.001, 0.1, 1, 5, 10 ,50 ,100],

             "gamma": [0.0001, 0.001, 0.1, 1, 5, 10 ,50 ,100]}



svc = SVC()

svc_cv_model_onehot = GridSearchCV(svc, svc_params, 

                         cv = 10, 

                         n_jobs = -1,

                         verbose = 2)



svc_cv_model_onehot.fit(X_train1, y_train1)
print("Best Params: " + str(svc_cv_model_onehot.best_params_))
svc_tuned_onehot = SVC(C = 1, gamma = 0.1).fit(X_train1, y_train1)



accuracy_score(y_train1, svc_tuned_onehot.predict(X_train1))
d = {'Accuracy in SVM before GridSearchCV ': [0.84], 'Accuracy in SVM After GridSearchCV': [0.83]}

svm_data = pd.DataFrame(data=d)

svm_data
rf_model_onehot = RandomForestClassifier().fit(X_train1, y_train1)



accuracy_score(y_train1, rf_model_onehot.predict(X_train1))
rf_params = {"max_depth": [2,5,8],

            "max_features": [2,5,8],

            "n_estimators": [10,500,1000],

            "min_samples_split": [2,5,10]}



rf_model = RandomForestClassifier()



rf_cv_model_onehot = GridSearchCV(rf_model, 

                           rf_params, 

                           cv = 10, 

                           n_jobs = -1, 

                           verbose = 2) 



rf_cv_model_onehot.fit(X_train1, y_train1)
print("Best Params: " + str(rf_cv_model_onehot.best_params_))
rf_tuned_onehot = RandomForestClassifier(max_depth = 5, 

                                  max_features = 5, 

                                  min_samples_split = 10,

                                  n_estimators = 500)
rf_tuned_onehot.fit(X_train1, y_train1)



accuracy_score(y_train1, rf_tuned_onehot.predict(X_train1))
confusion_matrix(y_train1, rf_tuned_onehot.predict(X_train1))

print(classification_report(y_train1, rf_tuned_onehot.predict(X_train1)))
d = {'Accuracy in RF before GridSearchCV ': [0.88], 'Accuracy in RF After GridSearchCV': [0.83]}

rf_data = pd.DataFrame(data=d)

rf_data
from sklearn.ensemble import GradientBoostingClassifier



gbm_model_onehot = GradientBoostingClassifier().fit(X_train1, y_train1)



accuracy_score(y_train, gbm_model_onehot.predict(X_train1))
gbm_params = {"learning_rate" : [0.001, 0.01, 0.1, 0.05],

             "n_estimators": [100,500,100],

             "max_depth": [3,5,10],

             "min_samples_split": [2,5,10]}



gbm = GradientBoostingClassifier()



gbm_cv_onehot = GridSearchCV(gbm, gbm_params, cv = 10, n_jobs = -1, verbose = 2)

gbm_cv_onehot.fit(X_train1, y_train1)
print("Best Params: " + str(gbm_cv_onehot.best_params_))
gbm = GradientBoostingClassifier(learning_rate = 0.01, 

                                 max_depth = 3,

                                min_samples_split = 2,

                                n_estimators = 100)



gbm_tuned_onehot =  gbm.fit(X_train1,y_train1)
accuracy_score(y_train1, gbm_tuned_onehot.predict(X_train1))
d = {'Accuracy in GBM before GridSearchCV ': [0.84], 'Accuracy in GBM After GridSearchCV': [0.82]}

gbm_data = pd.DataFrame(data=d)

gbm_data
models = [

    

    knn_tuned_onehot,

    log_model_onehot,

    svc_tuned_onehot,

    nb_model_onehot,

    rf_tuned_onehot,

    gbm_tuned_onehot,

    

]





for model in models:

    name = model.__class__.__name__

    y_pred = model.predict(X_train1)

    accuracy = accuracy_score(y_train1, y_pred)

    print("-"*28)

    print(name + ":" )

    print("Accuracy: {:.4%}".format(accuracy))
result = []



results = pd.DataFrame(columns= ["Models","Accuracy"])



for model in models:

    name = model.__class__.__name__

    y_pred = model.predict(X_train1)

    accuracy = accuracy_score(y_train1, y_pred)    

    result = pd.DataFrame([[name, accuracy*100]], columns= ["Models","Accuracy"])

    results = results.append(result)

    

    

sns.barplot(x= 'Accuracy', y = 'Models', data=results, color="r")

plt.xlabel('Accuracy %')

plt.title('accuracy rate of models'); 
df = [("KNN",84),

      ("KNN(onehot)",82),

      ("Logreg", 80),

      ("logreg(onehot)",82),

      ("SVC", 85),

      ("SVC(onehot)",83),

      ("NB", 80),

      ("NB(onehot)",44),

      ("RF",83),

      ("RF(onehot)",83),

      ("GBM", 85),

      ("GBM(onehot)",82)]

model = pd.DataFrame(df, columns=['Model' , 'Accuracy %'])

model
sns.barplot(x= 'Accuracy %', y = 'Model', data=model, color="r");
test_survived = pd.Series(gbm_tuned.predict(X_test), name = "Survived").astype(int)

results = pd.concat([test_PassengerId, test_survived],axis = 1)

results.to_csv("titanic.csv", index = False)