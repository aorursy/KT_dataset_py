import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import re as re

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from subprocess import check_output



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split,KFold

from sklearn.metrics.classification import classification_report,accuracy_score



import xgboost as xgb



sns.set(style='white', context='notebook', palette='deep')

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

PassengerIDs = test["PassengerId"].values

print(PassengerIDs.shape,train.shape,test.shape)
print(train.isnull().sum())

print("+++++======")

print(test.isnull().sum())
complete_data = [train,test]

for data in complete_data:

    data["FamilySize"] = data["Parch"]+data["SibSp"]+1

    data["IsAlone"] = 0

    data.loc[data["FamilySize"] == 1,'IsAlone'] = 1

    #filling NA values in Age Fare Embarked

    mean_age = data["Age"].mean()

    std_age = data["Age"].std()

    mean_fare = data["Fare"].mean()

    data.loc[np.isnan(data["Age"]),"Age"] = np.random.randint(mean_age-std_age,mean_age+std_age,size=data["Age"].isnull().sum())    

    data.loc[np.isnan(data["Fare"]),"Fare"] = mean_fare

    data["Embarked"] = data.Embarked.fillna("S")

    data["Sex"] = data["Sex"].map({'female':0,'male':1}).astype(int)

    data["Embarked"] = data["Embarked"].map({'S':0,'C':1,'Q':2}).astype(int)
for data in complete_data:

    data["CategoricalFare"] = pd.qcut(train["Fare"],3,labels=[0,1,2]) 

    data["CategoricalAge"] = pd.cut(train["Age"],5,labels=[0,1,2,3,4])
def get_tit(name):

    return name.split(",")[1].split(".")[0].strip()

def get_tit_reg(name):

    query = re.search('([A-Za-z]+)\.',name)

    if query:

        return query.group(1)

    return ""

for data in complete_data:

    data["Title"] = data["Name"].apply(get_tit_reg)

    data["Title"] = data["Title"].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')

    data['Title'] = data['Title'].replace('Mlle', 'Miss')

    data['Title'] = data['Title'].replace('Ms', 'Miss')

    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    data["Title"] = data["Title"].map({

        "Mr":1,"Miss":2,"Mrs":3,"Master":4,"Rare":5

    }).astype(int)

    data["Title"] = data["Title"].fillna(0)
plt.figure(figsize=(12,10))

corr_hmp = sns.heatmap(train.drop("PassengerId",axis=1).corr(),

                       vmax=0.6,square=True,annot=True)
cols = ['Survived','Pclass','Age','SibSp','Parch','Fare','Sex','Embarked','FamilySize','IsAlone','Title']

g = sns.pairplot(data=train[cols],vars=cols,size=1.25,hue='Survived',palette=["red","blue"])
for data in complete_data:

    data = data.drop(["PassengerId","Name","Ticket","Cabin"],axis=1,inplace=True)

    #print(data.columns)

Y_train = train["Survived"].values

X_train = train.drop("Survived",axis=1).values

X_test = test.values

print(Y_train.shape,X_train.shape,X_test.shape)
np.set_printoptions(threshold='nan')

DTree = DecisionTreeClassifier(max_depth=5,min_samples_split=6,random_state=1)

DTree = DTree.fit(X_train,Y_train)

print("Score for training set",DTree.score(X_train,Y_train))

DTree_pred = DTree.predict(X_test)

#print("Score for cross validation",DTree.score(X_CV,Y_CV))



KNC =  KNeighborsClassifier(n_neighbors=2,p=2)

KNC = KNC.fit(X_train,Y_train)

print("Score for training set",KNC.score(X_train,Y_train))

KNC_pred = KNC.predict(X_test)

#print("Score for cross validation",KNC.score(X_CV,Y_CV))





SVM = SVC()

SVM = SVM.fit(X_train,Y_train)

print("Score for training set",SVM.score(X_train,Y_train))

SVM_pred = SVM.predict(X_test)

#print("Score for cross validation",SVM.score(X_CV,Y_CV))



X_meta_train = np.column_stack((

                DTree.predict(X_train),

                KNC.predict(X_train),

                SVM.predict(X_train)

))

X_meta_test = np.column_stack((DTree_pred,KNC_pred,SVM_pred))

xmetdf = pd.DataFrame(X_meta_test)



rf_params = {

    'n_jobs': -1,

    'n_estimators': 575,

     'warm_start': True, 

     #'max_features': 0.2,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'verbose': 3 

}

et_params = {

    'n_jobs': -1,

    'n_estimators':575,

    #'max_features': 0.5,

    'max_depth': 5,

    'min_samples_leaf': 3,

    'verbose': 3

}

ada_params = {

    'n_estimators': 575,

    'learning_rate' : 0.95

}



gb_params = {

    'n_estimators': 575,

     #'max_features': 0.2,

    'max_depth': 5,

    'min_samples_leaf': 3,

    'verbose': 3

}

rfc = RandomForestClassifier(n_jobs=4,n_estimators=575,warm_start=True,max_depth=5,min_samples_leaf=2,max_features='sqrt',verbose=2)

abc = AdaBoostClassifier(n_estimators=575,learning_rate=0.95)

gbc = GradientBoostingClassifier(n_estimators=575,max_depth=5,min_samples_leaf=3,verbose=2)

etc = ExtraTreesClassifier(n_jobs=4,n_estimators=575,max_depth=5,min_samples_leaf=3,verbose=2)

rfc = rfc.fit(X_meta_train,Y_train)

abc = abc.fit(X_meta_train,Y_train)

gbc = gbc.fit(X_meta_train,Y_train)

etc = etc.fit(X_meta_train,Y_train)



X_meta_meta_train = np.column_stack((

            rfc.predict(X_meta_train),

            abc.predict(X_meta_train),

            gbc.predict(X_meta_train),

            etc.predict(X_meta_train)

))

X_meta_meta_test = np.column_stack((

            rfc.predict(X_meta_test),

            abc.predict(X_meta_test),

            gbc.predict(X_meta_test),

            etc.predict(X_meta_test)

))



gbm = xgb.XGBClassifier(learning_rate = 0.95,

 n_estimators= 16000,

 max_depth= 4,

 min_child_weight= 2,

 #gamma=1,

 gamma=1,                        

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread= 4,

 scale_pos_weight=1).fit(X_meta_meta_train,Y_train)



predictions = gbm.predict(X_meta_meta_test)

#print(predictions.shape,PassengerIDs.shape)

StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerIDs,

                            'Survived': predictions })

StackingSubmission.to_csv("StackingSubmission.csv", index=False)
kf = KFold(n_splits=10,shuffle=True)

def get_predictions(classifier,X_train,Y_train,X_test):

    for train_index, test_index in kf.split(X_train.values):

        X_kf_train = X_train[train_index]

        Y_kf_train = Y_train[train_index]

        X_kf_cv = X_test