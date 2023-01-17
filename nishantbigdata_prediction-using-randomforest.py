import pandas as pd

import matplotlib.pyplot as plt

from sklearn.svm import SVC

from sklearn.utils import shuffle

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

%matplotlib inline
tita_train = pd.read_csv('../input/train.csv')

tita_test  = pd.read_csv('../input/test.csv')
tita_train.describe()
#checking for null values

tita_train.isnull().sum()
tita_train['Pclass'].value_counts()
tita_train['Embarked'].value_counts()
Aff_FL30 = tita_train[(tita_train['Fare'] < 30) & (tita_train['Survived'] == 1) ]

Aff_but_dead1 = tita_train[(tita_train['Survived'] == 0) & (tita_train['Fare'] < 30) ]

Aff_but_alive1 = tita_train[(tita_train['Survived'] == 1) & (tita_train['Fare'] > 30) ]
plt.scatter(tita_train['Survived'],tita_train['Fare'])
Aff_FL30['Sex'].value_counts()
Aff_FG30 = tita_train[(tita_train['Survived'] == 1) & (tita_train['Fare'] < 32) ]
Aff_FG30['Sex'].value_counts()
# Function for pre processing

#Segregating categorical and numerical variables

def cat_num(X):   

    cat = []

    num = []

    for i in X.columns:

        if X[i].dtype == object:

            cat.append(i)

        else:

            num.append(i)

    return (cat, num)



#creating dummy variable

def dum_cre(X):

    X = X.drop(['Name'],axis = 1)

    X = X.drop(['Cabin'],axis = 1)

    X = X.drop(['Ticket'],axis = 1)

    cat_ti = pd.get_dummies(X)

    cat_ti = cat_ti.drop('Sex_female', axis = 1)

    cat_ti = cat_ti.drop('Embarked_S', axis = 1)

    return cat_ti 



#joining categorical and numerical data

def join_cat_num(X,Y):

    j = pd.concat([X,Y],axis =1)

    return j 

#Drop numerical attribute

def rem_num(X):

    j = X.drop(['PassengerId'],axis = 1)

    return j



#imputing mean with median

def imput_med(X):

    median = X['Age'].median()

    X['Age'].fillna(median, inplace=True)

    return X



def imput_med1(X):

    median = X['Fare'].median()

    X['Fare'].fillna(median, inplace=True)

    return X

#Preprocessing the training dataset

cat_tr, num_tr = cat_num(tita_train)

cat_tr1 = dum_cre(tita_train[cat_tr])

tita_num = tita_train[num_tr]

tit_final = join_cat_num(tita_num,cat_tr1)

tit_final = rem_num(tit_final)

tit_final = imput_med(tit_final)
#lets draw histogram 

tit_final.hist(bins = 10, figsize = (10,15))
#Preprocessing the test dataset

cat_te, num_te = cat_num(tita_test)

cat_te1 = dum_cre(tita_test[cat_te])

tita_numt = tita_test[num_te]

test_final = join_cat_num(tita_numt,cat_te1)

#test_final = rem_num(test_final)

test_final = imput_med(test_final)

test_final = imput_med1(test_final)
#shuffling the index

tit_final = shuffle(tit_final)
X_train = tit_final.ix[:,1:]

y_train = tit_final.ix[:,0]
#using sv

svc = SVC()

svc.fit(X_train, y_train)
y_pred = svc.predict(test_final.ix[:,1:])
y_pred = pd.DataFrame(y_pred, columns=['Survived'])
sub = pd.concat([test_final.ix[:,0],y_pred],axis=1)
svc_score = cross_val_score(svc, X_train, y_train, cv=10)

svc_score.mean()
#using random forest

rfc = RandomForestClassifier()

rf_score = cross_val_score(rfc, X_train, y_train, cv=10)

rf_score.mean()