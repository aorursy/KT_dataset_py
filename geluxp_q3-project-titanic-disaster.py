# import packages

import numpy as np

import pandas as pd

import seaborn as seaborn

from sklearn import preprocessing

from sklearn import tree

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
# Import data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.info()

test.info()
def dummies(col,train,test):

    train_dum = pd.get_dummies(train[col])

    test_dum = pd.get_dummies(test[col])

    train = pd.concat([train, train_dum], axis=1)

    test = pd.concat([test,test_dum],axis=1)

    train.drop(col,axis=1,inplace=True)

    test.drop(col,axis=1,inplace=True)

    return train, test



# get rid of the useless cols

dropping = ['PassengerId', 'Name', 'Ticket']

train.drop(dropping,axis=1, inplace=True)

test.drop(dropping,axis=1, inplace=True)

print(train.Pclass.value_counts())

seaborn.factorplot("Pclass",'Survived',data=train,order=[1,2,3])



train, test = dummies('Pclass',train,test)
print(train.Sex.value_counts(dropna=False))

seaborn.factorplot('Sex','Survived',data=train)

train,test = dummies('Sex',train,test)

train.drop('male',axis=1,inplace=True)

test.drop('male',axis=1,inplace=True)

nan_num = len(train[train['Age'].isnull()])

age_mean = train['Age'].mean()

age_std = train['Age'].std()

filling = np.random.randint(age_mean-age_std,age_mean+age_std,size=nan_num)

train['Age'][train['Age'].isnull()==True] = filling

nan_num = train['Age'].isnull().sum()

# dealing the missing val in test

nan_num = test['Age'].isnull().sum()

# 86 null

age_mean = test['Age'].mean()

age_std = test['Age'].std()

filling = np.random.randint(age_mean-age_std,age_mean+age_std,size=nan_num)

test['Age'][test['Age'].isnull()==True]=filling

nan_num = test['Age'].isnull().sum()



s = seaborn.FacetGrid(train,hue='Survived',aspect=2)

s.map(seaborn.kdeplot,'Age',shade=True)

s.set(xlim=(0,train['Age'].max()))

s.add_legend()



def under15(row):

    result = 0.0

    if row<15:

        result = 1.0

    return result

def young(row):

    result = 0.0

    if row>=15 and row<30:

        result = 1.0

    return result

train['under15'] = train['Age'].apply(under15)

train['young'] = train['Age'].apply(young)

test['under15'] = test['Age'].apply(under15)

test['young'] = test['Age'].apply(young)



train.drop('Age',axis=1,inplace=True)

test.drop('Age',axis=1,inplace=True)

print (train.SibSp.value_counts(dropna=False))

print (train.Parch.value_counts(dropna=False))

seaborn.factorplot('SibSp','Survived',data=train,size=5)

seaborn.factorplot('Parch','Survived',data=train,szie=5)



train['family'] = train['SibSp'] +  train['Parch']

test['family'] = test['SibSp'] + test['Parch']

seaborn.factorplot('family','Survived',data=train,size=5)



train.drop(['SibSp','Parch'],axis=1,inplace=True)

test.drop(['SibSp','Parch'],axis=1,inplace=True)

print (train.Fare.isnull().sum())

print (test.Fare.isnull().sum())

seaborn.factorplot('Survived','Fare',data=train,size=4)

s = seaborn.FacetGrid(train,hue='Survived',aspect=2)

s.map(seaborn.kdeplot,'Fare',shade=True)

s.set(xlim=(0,train['Fare'].max()))

s.add_legend()



test['Fare'].fillna(test['Fare'].median(),inplace=True)

#Cabin

print (train.Cabin.isnull().sum())

print (test.Cabin.isnull().sum())



train.drop('Cabin',axis=1,inplace=True)

test.drop('Cabin',axis=1,inplace=True)
#Embarked

print (train.Embarked.isnull().sum())

print (test.Embarked.isnull().sum())



print (train['Embarked'].value_counts(dropna=False))

print (train['Embarked'].fillna('S',inplace=True))



seaborn.factorplot('Embarked','Survived',data=train,size=5)



train,test = dummies('Embarked',train,test)

train.drop(['S','Q'],axis=1,inplace=True)

test.drop(['S','Q'],axis=1,inplace=True)
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score, KFold

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier





def modeling(clf,ft,target):

    acc = cross_val_score(clf,ft,target,cv=kf)

    acc_lst.append(acc.mean())

    return 



accuracy = []

def ml(ft,target,time):

    accuracy.append(acc_lst)

     #logisticregression

    logreg = LogisticRegression()

    modeling(logreg,ft,target)

    #RandomForest

    rf = RandomForestClassifier(n_estimators=50,min_samples_split=4,min_samples_leaf=2)

    modeling(rf,ft,target)

    #svc rbf

    svc1 = SVC(kernel="rbf")

    modeling(svc1,ft,target)

    #svc 

    svc2 = SVC(kernel="linear")

    modeling(svc2,ft,target)

    #knn

    knn = KNeighborsClassifier(n_neighbors = 3)

    modeling(knn,ft,target)

    #MLPClassifier

    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,3),random_state=10)

    modeling(mlp, ft, target)



    

    # see the coefficient

    logreg.fit(ft,target)

    feature = pd.DataFrame(ft.columns)

    feature.columns = ['Features']

    feature["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

    print(feature)

    return

#test1

train_ft = train.drop('Survived',axis=1)

train_y = train['Survived']



kf = KFold(n_splits=3,random_state=1)

acc_lst = []

ml(train_ft,train_y,'test_1')

# testing 2, lose young

train_ft_2=train.drop(['Survived','young'],axis=1)

test_2 = test.drop('young',axis=1)

train_ft.head()



# ml

kf = KFold(n_splits=3,random_state=1)

acc_lst=[]

ml(train_ft_2,train_y,'test_2')

#test3, lose young, c

train_ft_3=train.drop(['Survived','young','C'],axis=1)

test_3 = test.drop(['young','C'],axis=1)

train_ft.head()



# ml

kf = KFold(n_splits=3,random_state=1)

acc_lst = []

ml(train_ft_3,train_y,'test_3')

# test4, no FARE

train_ft_4=train.drop(['Survived','Fare'],axis=1)

test_4 = test.drop(['Fare'],axis=1)

train_ft.head()

# ml

kf = KFold(n_splits=3,random_state=1)

acc_lst = []

ml(train_ft_4,train_y,'test_4')

accuracy_df=pd.DataFrame(data=accuracy,

                         index=['test1','test2','test3','test4'],

                         columns=['logistic','rf','svc1','svc2','knn', 'mlp'])

accuracy_df

svc = SVC()

svc.fit(train_ft_4,train_y)

svc_pred = svc.predict(test_4)

print(svc.score(train_ft_4,train_y))



submission_test = pd.read_csv("../input/test.csv")

submission = pd.DataFrame({"PassengerId":submission_test['PassengerId'],

                          "Survived":svc_pred})

submission.to_csv("kaggle_SVC.csv",index=False)