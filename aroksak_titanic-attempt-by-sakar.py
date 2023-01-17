import warnings

warnings.filterwarnings('ignore')



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.
def dummies(col,train,test):

    train = pd.get_dummies(train,columns=col)

    test = pd.get_dummies(test,columns=col)

    return train, test
train  =pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head(5)
print(train.Pclass.value_counts(dropna=False))

sns.factorplot('Pclass', 'Survived',data=train, order=[1,2,3])

train,test = dummies(['Pclass'],train,test)

train.drop(['Pclass_3'],axis=1,inplace=True)

test.drop(['Pclass_3'],axis=1,inplace=True)
print(train.Sex.value_counts(dropna=False))

sns.factorplot('Sex','Survived',data=train)

train, test = dummies(['Sex'], train, test)

#delete male

train.drop(['Sex_male','PassengerId','Name','Ticket'],axis=1,inplace=True)

test.drop(['Sex_male','PassengerId','Name','Ticket'],axis=1,inplace=True)
age_mean = train['Age'].mean()

age_std = train['Age'].std()

nan_num = train['Age'].isnull().sum()

filling = np.random.randint(age_mean-age_std, age_mean+age_std, size=nan_num)

train['Age'][train['Age'].isnull()==True] = filling

nan_num = test['Age'].isnull().sum()

filling = np.random.randint(age_mean-age_std, age_mean+age_std, size=nan_num)

test['Age'][test['Age'].isnull()==True] = filling



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



train['under15']=train['Age'].apply(under15)

test['under15']=test['Age'].apply(under15)

train['young']=train['Age'].apply(young)

test['young']=train['Age'].apply(young)



train.drop('Age',axis=1,inplace=True)

test.drop('Age',axis=1,inplace=True)
train['family'] = train['SibSp'] + train['Parch']

test['family'] = test['SibSp'] + test['Parch']

sns.factorplot('family','Survived',data=train,size=5)



def SmallFamily(row):

    result = 0.0

    if row<4:

        result = 1.0

    return result



train['SmallFamily']=train['family'].apply(SmallFamily)

test['SmallFamily']=test['family'].apply(SmallFamily)



train.drop(['SibSp','Parch','family'],axis=1,inplace=True)

test.drop(['SibSp','Parch','family'],axis=1,inplace=True)
train.Fare.isnull().sum()

test.Fare.isnull().sum()



def BigFare(row):

    result = 0.0

    if row>35:

        result = 1.0

    return result



train['BigFare']=train['Fare'].apply(BigFare)

test['BigFare']=test['Fare'].apply(BigFare)



sns.factorplot('BigFare','Survived',data=train,size=5)



train.drop('Fare',axis=1,inplace=True)

test.drop('Fare',axis=1,inplace=True)
print(train.Cabin.isnull().sum())

train.drop('Cabin',axis=1,inplace=True)

test.drop('Cabin',axis=1,inplace=True)
train.Embarked.isnull().sum()

train.Embarked.value_counts()

train['Embarked'].fillna('S',inplace=True)



train,test = dummies(['Embarked'],train,test)

train.drop(['Embarked_S','Embarked_Q'],axis=1,inplace=True)

test.drop(['Embarked_S','Embarked_Q'],axis=1,inplace=True)
# import machine learning libraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score, KFold



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

    #svc

    svc = SVC()

    modeling(svc,ft,target)

    #knn

    knn = KNeighborsClassifier(n_neighbors = 3)

    modeling(knn,ft,target)

    

    # see the coefficient

    logreg.fit(ft,target)

    feature = pd.DataFrame(ft.columns)

    feature.columns = ['Features']

    feature["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

    print(feature)

    return 
# testing no.1, using all the feature

train_ft=train.drop('Survived',axis=1)

train_y=train['Survived']

#set kf

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

train_ft_3=train.drop(['Survived','young','Embarked_C'],axis=1)

test_3 = test.drop(['young','Embarked_C'],axis=1)

train_ft.head()



# ml

kf = KFold(n_splits=3,random_state=1)

acc_lst = []

ml(train_ft_3,train_y,'test_3')
# test4, no FARE

train_ft_4=train.drop(['Survived','BigFare'],axis=1)

test_4 = test.drop(['BigFare'],axis=1)

train_ft.head()

# ml

kf = KFold(n_splits=3,random_state=1)

acc_lst = []

ml(train_ft_4,train_y,'test_4')
# test5, get rid of c 

train_ft_5=train.drop(['Survived','Embarked_C'],axis=1)

test_5 = test.drop('Embarked_C',axis=1)



# ml

kf = KFold(n_splits=3,random_state=1)

acc_lst = []

ml(train_ft_5,train_y,'test_5')
# test6, lose Fare and young

train_ft_6=train.drop(['Survived','BigFare','young'],axis=1)

test_6 = test.drop(['BigFare','young'],axis=1)

train_ft.head()

# ml

kf = KFold(n_splits=3,random_state=1)

acc_lst = []

ml(train_ft_6,train_y,'test_6')
accuracy_df=pd.DataFrame(data=accuracy,

                         index=['test1','test2','test3','test4','test5','test6'],

                         columns=['logistic','rf','svc','knn'])

accuracy_df
rf = RandomForestClassifier(n_estimators=50,min_samples_split=4,min_samples_leaf=2)

rf.fit(train_ft_3,train_y)

rf_pred = rf.predict(test_3)

print(rf.score(train_ft_3,train_y))



test = pd.read_csv('../input/test.csv')

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": rf_pred

    })

submission.to_csv("kaggle3.csv", index=False)