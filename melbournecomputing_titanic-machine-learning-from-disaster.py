import warnings

warnings.filterwarnings('ignore')



import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.info()
train.describe()
# show the overall survival rate (38.38), as the standard when choosing the fts

print('Overall Survival Rate:',train['Survived'].mean())
# output train to check

print('train:',train)
# get_dummies function

#def dummies(col,train,test):

    #train_dum = pd.get_dummies(train[col])

    #test_dum = pd.get_dummies(test[col])

    #train = pd.concat([train, train_dum], axis=1)

    #test = pd.concat([test,test_dum],axis=1)

    #train.drop(col,axis=1,inplace=True)

    #test.drop(col,axis=1,inplace=True)

    #return train, test



# get rid of the useless cols

dropping = ['PassengerId', 'Name', 'Ticket']

train.drop(dropping,axis=1, inplace=True)

test.drop(dropping,axis=1, inplace=True)
# output train to check

print('train:',train)
#pclass

# ensure no na contained

print(train.Pclass.value_counts(dropna=False))

sns.factorplot('Pclass', 'Survived',data=train, order=[3,2,1])

# according to the graph, we found there are huge differences between

# each pclass group. keep the ft

train, test = dummies('Pclass', train, test)
# sex

print(train.Sex.value_counts(dropna=False))

sns.factorplot('Sex','Survived', data=train)

# female survival rate is way better than the male

train, test = dummies('Sex', train, test)

# cos the male survival rate is so low, delete the male col

train.drop('male',axis=1,inplace=True)

test.drop('male',axis=1,inplace=True)
#age 

#dealing the missing data

nan_num = train['Age'].isnull().sum()

# there are 177 missing value, fill with random int

age_mean = train['Age'].mean()

age_std = train['Age'].std()

filling = np.random.randint(age_mean-age_std, age_mean+age_std, size=nan_num)

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



#look into the age col

s = sns.FacetGrid(train,hue='Survived',aspect=3)

s.map(sns.kdeplot,'Age',shade=True)

s.set(xlim=(0,train['Age'].max()))

s.add_legend()



# from the graph, we see that the survival rate of children

# is higher than other and the 15-30 survival rate is lower

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

test['under15'] = test['Age'].apply(under15)

train['young'] = train['Age'].apply(young)

test['young'] = test['Age'].apply(young)



train.drop('Age',axis=1,inplace=True)

test.drop('Age',axis=1,inplace=True)
#family

# chek

print(train['SibSp'].value_counts(dropna=False))

print(train['Parch'].value_counts(dropna=False))



sns.factorplot('SibSp','Survived',data=train,size=5)

sns.factorplot('Parch','Survived',data=train,size=5)



'''through the plot, we suggest that with more family member, 

the survival rate will drop, we can create the new col

add up the parch and sibsp to check our theory''' 

train['family'] = train['SibSp'] + train['Parch']

test['family'] = test['SibSp'] + test['Parch']

sns.factorplot('family','Survived',data=train,size=5)



train.drop(['SibSp','Parch'],axis=1,inplace=True)

test.drop(['SibSp','Parch'],axis=1,inplace=True)
# fare

# checking null, found one in test group. leave it alone til we find out

# wether we should use this ft

train.Fare.isnull().sum()

test.Fare.isnull().sum()



sns.factorplot('Survived','Fare',data=train,size=5)

#according to the plot, smaller fare has higher survival rate, keep it

#dealing the null val in test

test['Fare'].fillna(test['Fare'].median(),inplace=True)
#Cabin

# checking missing val

# 687 out of 891 are missing, drop this col

train.Cabin.isnull().sum()

train.drop('Cabin',axis=1,inplace=True)

test.drop('Cabin',axis=1,inplace=True)
#Embark

train.Embarked.isnull().sum()

# 2 missing value

train.Embarked.value_counts()

# fill the majority val,'s', into missing val col

train['Embarked'].fillna('S',inplace=True)



sns.factorplot('Embarked','Survived',data=train,size=6)

# c has higher survival rate, drop the other two

train,test = dummies('Embarked',train,test)

train.drop(['S','Q'],axis=1,inplace=True)

test.drop(['S','Q'],axis=1,inplace=True)
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
# test5, get rid of c 

train_ft_5=train.drop(['Survived','C'],axis=1)

test_5 = test.drop('C',axis=1)



# ml

kf = KFold(n_splits=3,random_state=1)

acc_lst = []

ml(train_ft_5,train_y,'test_5')
# test6, lose Fare and young

train_ft_6=train.drop(['Survived','Fare','young'],axis=1)

test_6 = test.drop(['Fare','young'],axis=1)

train_ft.head()

# ml

kf = KFold(n_splits=3,random_state=1)

acc_lst = []

ml(train_ft_6,train_y,'test_6')
accuracy_df=pd.DataFrame(data=accuracy,

                         index=['test1','test2','test3','test4','test5','test6'],

                         columns=['logistic','rf','svc','knn'])

accuracy_df
'''

According to the accuracy chart, 'features test4 with svc'

got best performance

'''  

#test4 svc as submission

svc = SVC()

svc.fit(train_ft_4,train_y)

svc_pred = svc.predict(test_4)

print(svc.score(train_ft_4,train_y))





test = pd.read_csv('../input/test.csv')

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": svc_pred

    })

#submission.to_csv("kaggle.csv", index=False)