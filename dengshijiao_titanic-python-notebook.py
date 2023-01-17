import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
from scipy import stats
import re

from scipy.stats import chi2_contingency

from collections import Counter
import seaborn as sns
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
test_y = pd.read_csv("../input/gender_submission.csv")
train.head(10)
train["FamilleMember"] = train["SibSp"]+train["Parch"]
train = train.drop(["SibSp","Parch"],axis=1)
for i in range(train.shape[0]):
    l = re.split("[,|.|()]",train.Name[i])
    train.loc[i,"First Name"] = l[0].strip()
    train.loc[i,"Title"] = l[1].strip()
    train.loc[i,"Last Name"] = l[2].strip()
    try:
        train.loc[i,"Commentaire"] = l[3].strip()
    except:
        train.loc[i,"Commentaire"] = float("nan")
train.head()
train.describe()
train.describe(include='O')
Counter(train.Title)
train = train.drop(["PassengerId","Name","Ticket","First Name","Last Name","Commentaire","Cabin"],axis=1)
train.Title = train.Title.map({'Sir':'Autre', 'Lady':'Miss', 'Dr':'Autre', 'Jonkheer':'Autre', 'the Countess':'Autre', 'Don':'Autre', 'Mme':'Mrs', 'Mlle':'Miss', 'Major':'Autre', 'Col':'Autre', 'Ms':'Mrs', 'Rev':'Autre', 'Capt':'Autre','Miss':'Miss','Mr':'Mr','Master':'Master','Mrs':'Mrs'})
Counter(train.Title)
t = pd.crosstab(train.Survived,train.Title)
x = [1,2,3,4,5]
y_non_survived = t.iloc[0,:]
y_survived = t.iloc[1,:]
p1 = plt.bar(x,y_non_survived,alpha=0.5)
plt.xticks(x,t.columns)
p2 = plt.bar(x,y_survived,bottom=y_non_survived,alpha=0.5)
plt.xticks(x,t.columns)
plt.legend((p1[0],p2[0]),("Not Survived","Survived"))
plt.show()
x = [1,2,3,4,5]
y_non_survived = t.iloc[0,:]/t.sum()
y_survived = t.iloc[1,:]/t.sum()
p1 = plt.bar(x,y_non_survived,alpha=0.5)
plt.xticks(x,t.columns)
p2 = plt.bar(x,y_survived,bottom=y_non_survived,alpha=0.5)
plt.xticks(x,t.columns)
plt.legend((p1[0],p2[0]),("Not Survived","Survived"))
plt.show()
from scipy import stats
stats.normaltest(train.loc[train.Age.notna(),"Age"])
np.random.seed(12345)
age_avg = train['Age'].mean()
age_std = train['Age'].std()
age_null_count = train['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
plt.hist(age_null_random_list,alpha=0.7)
plt.xlim(0,80)
plt.show()
train.Age.hist(alpha=0.7)
plt.xlim(0,80)
plt.show()
train['Age'][np.isnan(train['Age'])] = age_null_random_list
train['Age'] = train['Age'].astype(int)
Counter(train.Embarked)
train.loc[train.Embarked.isna(),"Embarked"] = "S"
import seaborn as sns
sns.pairplot(train, diag_kind="kde",hue="Survived")
plt.title("Distribution of the variable")
plt.show()
# Colonnes à supprimer : Cabin, Commentaire, Last Name, Ticket, First Name, Name, PassengerId
train.head()
train.describe(include='O')
train.describe()
train["Survived"] = train.Survived.astype(int)
train["Pclass_1"] = train["Pclass"] == 1
train["Pclass_2"] = train["Pclass"] == 2
train["Pclass_3"] = train["Pclass"] == 3
train["Sex"] = train["Sex"].map({"male":1,"female":0})
Counter(train.Sex)
Counter(train.Embarked)
train["Embarked_1"] = train["Embarked"] == 'C'
train["Embarked_2"] = train["Embarked"] == 'Q'
train["Embarked_3"] = train["Embarked"] == 'S'
Counter(train.FamilleMember)
train["FamilleMember"] = train["FamilleMember"].map({0:'0',1:'1',2:'2',3:'3',4:'>4',5:'>4',6:'>4',7:'>4',10:'>4'})
train["Famille_0"] = train["FamilleMember"] == '0'
train["Famille_1"] = train["FamilleMember"] == '1'
train["Famille_2"] = train["FamilleMember"] == '2'
train["Famille_3"] = train["FamilleMember"] == '3'
train["Famille_4"] = train["FamilleMember"] == '>4'
Counter(train.Title)
train["Title_master"] = train["Title"] == 'Master'
train["Title_miss"] = train["Title"] == 'Miss'
train["Title_mr"] = train["Title"] == 'Mr'
train["Title_mrs"] = train["Title"] == 'Mrs'
train["Title_autre"] = train["Title"] == 'Autre'
train["Age_1"] = train["Age"]<4
train["Age_2"] = (4 <= train["Age"])&(train["Age"]<15)
train["Age_3"] = (15 <= train["Age"])&(train["Age"]<30)
train["Age_4"] = (30 <= train["Age"])&(train["Age"]<45)
train["Age_5"] = (45 <= train["Age"])&(train["Age"]<60)
train["Age_6"] = (60 <= train["Age"])
train["Fare_1"] = train["Fare"]<20
train["Fare_2"] = (20 <= train["Fare"])&(train["Fare"]<40)
train["Fare_3"] = (40 <= train["Fare"])&(train["Fare"]<100)
train["Fare_4"] = (100 <= train["Fare"])
test.describe()
test.describe(include='O')
np.random.seed(12345)
age_avg = test['Age'].mean()
age_std = test['Age'].std()
age_null_count = test['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
test['Age'][np.isnan(test['Age'])] = age_null_random_list
test['Age'] = test['Age'].astype(int)
test["Fare"][test.Fare.isna()] = test["Fare"].mean()
test["Pclass_1"] = test["Pclass"] == 1
test["Pclass_2"] = test["Pclass"] == 2
test["Pclass_3"] = test["Pclass"] == 3
test["Sex"] = test["Sex"].map({"male":1,"female":0})
test["Embarked_1"] = test["Embarked"] == 'C'
test["Embarked_2"] = test["Embarked"] == 'Q'
test["Embarked_3"] = test["Embarked"] == 'S'
test["FamilleMember"] = test["SibSp"]+test["Parch"]
Counter(test.FamilleMember)
test["FamilleMember"] = test["FamilleMember"].map({0:'0',1:'1',2:'2',3:'3',4:'>4',5:'>4',6:'>4',7:'>4',10:'>4'})
test["Famille_0"] = test["FamilleMember"] == '0'
test["Famille_1"] = test["FamilleMember"] == '1'
test["Famille_2"] = test["FamilleMember"] == '2'
test["Famille_3"] = test["FamilleMember"] == '3'
test["Famille_4"] = test["FamilleMember"] == '>4'
for i in range(test.shape[0]):
    l = re.split("[,|.|()]",test.Name[i])
    test.loc[i,"First Name"] = l[0].strip()
    test.loc[i,"Title"] = l[1].strip()
    test.loc[i,"Last Name"] = l[2].strip()
    try:
        test.loc[i,"Commentaire"] = l[3].strip()
    except:
        test.loc[i,"Commentaire"] = float("nan")
test.Title = test.Title.map({'Sir':'Autre', 'Lady':'Miss', 'Dr':'Autre', 'Jonkheer':'Autre', 'the Countess':'Autre', 'Don':'Autre', 'Mme':'Mrs', 'Mlle':'Miss', 'Major':'Autre', 'Col':'Autre', 'Ms':'Mrs', 'Rev':'Autre', 'Capt':'Autre','Miss':'Miss','Mr':'Mr','Master':'Master','Mrs':'Mrs'})

test["Title_master"] = test["Title"] == 'Master'
test["Title_miss"] = test["Title"] == 'Miss'
test["Title_mr"] = test["Title"] == 'Mr'
test["Title_mrs"] = test["Title"] == 'Mrs'
test["Title_autre"] = test["Title"] == 'Autre'
test["Age_1"] = test["Age"]<4
test["Age_2"] = (4 <= test["Age"])&(test["Age"]<15)
test["Age_3"] = (15 <= test["Age"])&(test["Age"]<30)
test["Age_4"] = (30 <= test["Age"])&(test["Age"]<45)
test["Age_5"] = (45 <= test["Age"])&(test["Age"]<60)
test["Age_6"] = (60 <= test["Age"])
test["Fare_1"] = test["Fare"]<20
test["Fare_2"] = (20 <= test["Fare"])&(test["Fare"]<40)
test["Fare_3"] = (40 <= test["Fare"])&(test["Fare"]<100)
test["Fare_4"] = (100 <= test["Fare"])
train = train.loc[:,['Survived','Sex', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_1', 'Embarked_2','Embarked_3'
                     ,'Famille_0', 'Famille_1', 'Famille_2', 'Famille_3', 'Famille_4'
                     ,'Title_master', 'Title_miss', 'Title_mr', 'Title_mrs','Title_autre'
                     ,'Age_1', 'Age_2', 'Age_3', 'Age_4', 'Age_5', 'Age_6'
                     ,'Fare_1', 'Fare_2', 'Fare_3', 'Fare_4']]
colormap = plt.cm.RdBu
x = [i for i in range(train.shape[1])]
plt.figure(figsize=(15,10))
plt.bar(x,train.corr()["Survived"],alpha=0.5)
plt.xticks(x,train.columns,rotation=-45)
plt.show()
columns_complete = ['Sex', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_1', 'Embarked_2','Embarked_3'
                     ,'Famille_0', 'Famille_1', 'Famille_2', 'Famille_3', 'Famille_4'
                     ,'Title_master', 'Title_miss', 'Title_mr', 'Title_mrs','Title_autre'
                     ,'Age_1', 'Age_2', 'Age_3', 'Age_4', 'Age_5', 'Age_6'
                     ,'Fare_1', 'Fare_2', 'Fare_3', 'Fare_4']

columns_reduced = ['Sex', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_1', 'Embarked_3'
                     ,'Famille_0', 'Famille_1', 'Famille_2', 'Famille_3', 'Famille_4'
                     ,'Title_master', 'Title_miss', 'Title_mr', 'Title_mrs'
                     ,'Age_1', 'Age_2', 'Age_3'
                     ,'Fare_1', 'Fare_3', 'Fare_4']
from sklearn.metrics import accuracy_score
from sklearn import tree
train_y = train["Survived"].ravel()
classifier = tree.DecisionTreeClassifier()
dt_complete = classifier.fit(train.loc[:,columns_complete],train_y)
dt_complete_pred = dt_complete.predict(test.loc[:,columns_complete])
accuracy_score(test_y["Survived"],dt_complete_pred)
classifier = tree.DecisionTreeClassifier()
dt_reduce = classifier.fit(train.loc[:,columns_reduced],train_y)
dt_reduce_pred = dt_reduce.predict(test.loc[:,columns_reduced])
accuracy_score(test_y["Survived"],dt_reduce_pred)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
nb_complete = classifier.fit(train.loc[:,columns_complete],train_y)
nb_complete_pred = nb_complete.predict(test.loc[:,columns_complete])
accuracy_score(test_y["Survived"],nb_complete_pred)
classifier = GaussianNB()
nb_reduce = classifier.fit(train.loc[:,columns_reduced],train_y)
nb_reduce_pred = nb_reduce.predict(test.loc[:,columns_reduced])
accuracy_score(test_y["Survived"],nb_reduce_pred)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
rf_complete = classifier.fit(train.loc[:,columns_complete],train_y)
rf_complete_pred = rf_complete.predict(test.loc[:,columns_complete])
accuracy_score(test_y["Survived"],rf_complete_pred)
classifier = RandomForestClassifier()
rf_reduce = classifier.fit(train.loc[:,columns_reduced],train_y)
rf_reduce_pred = rf_reduce.predict(test.loc[:,columns_reduced])
accuracy_score(test_y["Survived"],rf_reduce_pred)
from sklearn import svm
classifier = svm.SVC() #rbf
svm_complete = classifier.fit(train.loc[:,columns_complete],train_y)
svm_complete_pred_rbf = svm_complete.predict(test.loc[:,columns_complete])
accuracy_score(test_y["Survived"],svm_complete_pred_rbf)
classifier = svm.SVC(kernel='poly') #rbf
svm_complete = classifier.fit(train.loc[:,columns_complete],train_y)
svm_complete_pred_poly = svm_complete.predict(test.loc[:,columns_complete])
accuracy_score(test_y["Survived"],svm_complete_pred_poly)
classifier = svm.SVC() #rbf
svm_reduce = classifier.fit(train.loc[:,columns_reduced],train_y)
svm_reduce_pred_rbf = svm_reduce.predict(test.loc[:,columns_reduced])
accuracy_score(test_y["Survived"],svm_reduce_pred_rbf)
classifier = svm.SVC(kernel='poly') #rbf
svm_reduce = classifier.fit(train.loc[:,columns_reduced],train_y)
svm_reduce_pred_poly = svm_reduce.predict(test.loc[:,columns_reduced])
accuracy_score(test_y["Survived"],svm_reduce_pred_poly)
data_to_submit = pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived':svm_reduce_pred_rbf
})
data_to_submit.to_csv("output_2.csv",index=False)
