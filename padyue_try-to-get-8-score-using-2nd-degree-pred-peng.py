# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame
import random
# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# normalizeation
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree,svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

# preview the data
titanic_df.head()
titanic_df.info()
print("----------------------------")
test_df.info()
# drop unnecessary columns, these columns won't be useful in analysis and prediction
titanic_df = titanic_df.drop(['PassengerId','Ticket','Cabin','Fare'], axis=1)
test_df    = test_df.drop(['Ticket','Cabin','Fare'], axis=1)
titanic_df=titanic_df.dropna(axis=0,how='any')


def get_masterormiss(passenger):
    name = passenger
    if (   ('Master' in str(name)) \
        or ('Miss'   in str(name)) \
        or ('Mlle'   in str(name)) ):
        return 1
    else:
        return 0

titanic_df['MasterMiss'] = \
    titanic_df[['Name']].apply( get_masterormiss, axis=1 )


test_df['MasterMiss'] = \
    test_df[['Name']].apply( get_masterormiss, axis=1 )

titanic_df = titanic_df.drop(['Name'], axis=1)
test_df    = test_df.drop(['Name'], axis=1)


# get average, std, and number of NaN values in test_df
average_age_test   = test_df["Age"].mean()
std_age_test       = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)


test_df["Age"][np.isnan(test_df["Age"])] = rand_2


test_df['Age']    = test_df['Age'].astype(int)
        

# Sex

# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.
# So, we can classify passengers as males, females, and child
def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16  else sex
    
titanic_df['Person'] = titanic_df[['Age','Sex']].apply(get_person,axis=1)
test_df['Person']    = test_df[['Age','Sex']].apply(get_person,axis=1)

# No need to use Sex column since we created Person column
titanic_df.drop(['Sex'],axis=1,inplace=True)
test_df.drop(['Sex'],axis=1,inplace=True)

# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
person_dummies_titanic  = pd.get_dummies(titanic_df['Person'])
person_dummies_titanic.columns = ['Child','Female','Male']

person_dummies_test  = pd.get_dummies(test_df['Person'])
person_dummies_test.columns = ['Child','Female','Male']



titanic_df = titanic_df.join(person_dummies_titanic)
test_df    = test_df.join(person_dummies_test)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

# sns.factorplot('Person',data=titanic_df,kind='count',ax=axis1)
sns.countplot(x='Person', data=titanic_df, ax=axis1)

# average of survived for each Person(male, female, or child)
person_perc = titanic_df[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()
sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])

titanic_df.drop(['Person'],axis=1,inplace=True)
test_df.drop(['Person'],axis=1,inplace=True)

# Pclass

sns.factorplot('Pclass','Survived',order=[1,2,3], data=titanic_df,size=5)

# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']


titanic_df = titanic_df.join(pclass_dummies_titanic)
test_df    = test_df.join(pclass_dummies_test)


# Embarked

# only in titanic_df, fill the two missing values with the most occurred value, which is "S".

# plot
sns.factorplot('Embarked','Survived', data=titanic_df,size=4,aspect=3)

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

# sns.factorplot('Embarked',data=titanic_df,kind='count',order=['S','C','Q'],ax=axis1)
# sns.factorplot('Survived',hue="Embarked",data=titanic_df,kind='count',order=[1,0],ax=axis2)
sns.countplot(x='Embarked', data=titanic_df, ax=axis1)
sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[1,0], ax=axis2)

# group by embarked, and get the mean for survived passengers for each value in Embarked
embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)

# Either to consider Embarked column in predictions,
# and remove "S" dummy variable, 
# and leave "C" & "Q", since they seem to have a good rate for Survival.

# OR, don't create dummy variables for Embarked column, just drop it, 
# because logically, Embarked doesn't seem to be useful in prediction.

embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])


embark_dummies_test  = pd.get_dummies(test_df['Embarked'])


titanic_df = titanic_df.join(embark_dummies_titanic)
test_df    = test_df.join(embark_dummies_test)



#pclass 12 male
def get_malem(passenger):
    pc,male,mm=passenger
    if ( (pc==1 or pc==2 ) and male==1 and mm==0):
        return 1
    else:
        return 0

titanic_df['Pmm'] = \
    titanic_df[['Pclass','Male','MasterMiss']].apply( get_malem, axis=1 )
test_df['Pmm'] = \
    test_df[['Pclass','Male','MasterMiss']].apply( get_malem, axis=1 )

    
titanic_df['ParchBinary'] = \
  titanic_df[['Parch']].apply( (lambda x: int(int(x) > 0) ), axis=1)
test_df['ParchBinary'] = \
  test_df[['Parch']].apply( (lambda x: int(int(x) > 0) ), axis=1)

#pclass1 male without parch 
def get_malepm(passenger):
    pch,male,pc=passenger
    if ( pch==0 and male==1 and pc==1 ):
        return 1
    else:
        return 0
titanic_df['PchM1'] = \
    titanic_df[['ParchBinary','Male','Pclass']].apply( get_malepm, axis=1 )
test_df['PchM1'] = \
    test_df[['ParchBinary','Male','Pclass']].apply( get_malepm, axis=1 )
#pclass3 female embarked s
def get_3fs(passenger):
    pc,female,s=passenger
    if ( pc==3 and female==1 and s==1):
        return 1
    else:
        return 0

titanic_df['3fs'] = \
    titanic_df[['Pclass','Female','S']].apply( get_3fs, axis=1 )
test_df['3fs'] = \
    test_df[['Pclass','Female','S']].apply( get_3fs, axis=1 )
titanic_df['FamilySize'] = titanic_df ['SibSp'] + titanic_df['Parch'] + 1
test_df['FamilySize'] = test_df ['SibSp'] + test_df['Parch'] + 1

titanic_df['BigFamily'] = \
  titanic_df[['FamilySize']].apply( (lambda x: int(int(x) >= 5) ), axis=1)
test_df['BigFamily'] = \
  test_df[['FamilySize']].apply( (lambda x: int(int(x) >= 5) ), axis=1)
    
titanic_df['SmallFamily'] = \
  titanic_df[['FamilySize']].apply( (lambda x: int(int(x) >= 2 and int(x)<=4) ), axis=1)
test_df['SmallFamily'] = \
  test_df[['FamilySize']].apply( (lambda x: int(int(x) >= 2 and int(x)<=4) ), axis=1)
    
titanic_df['Alone'] = \
  titanic_df[['FamilySize']].apply( (lambda x: int(int(x) ==1) ), axis=1)
test_df['Alone'] = \
  test_df[['FamilySize']].apply( (lambda x: int(int(x) ==1) ), axis=1)
    
temp=titanic_df.loc[titanic_df['Male']==1]


fig, axis1 = plt.subplots(1,1,figsize=(18,4))
average_age = temp[["FamilySize", "Survived"]].groupby(['FamilySize'],as_index=False).mean()
sns.barplot(x='FamilySize', y='Survived', data=average_age)
#child with big family
def get_Bc(passenger):
    child,big=passenger
    if ( child==1 and big==1):
        return 1
    else:
        return 0

titanic_df['Bc'] = \
    titanic_df[['Child','BigFamily']].apply( get_Bc, axis=1 )
test_df['Bc'] = \
    test_df[['Child','BigFamily']].apply( get_Bc, axis=1 )

#female embarked c
def get_fmalec(passenger):
    emc,male=passenger
    if ( emc==1 and male==1):
        return 1
    else:
        return 0

titanic_df['Fc'] = \
    titanic_df[['C','Male']].apply( get_fmalec, axis=1 )
test_df['Fc'] = \
    test_df[['C','Male']].apply( get_fmalec, axis=1 )

    
titanic_df.drop(['Embarked'],axis=1,inplace=True)
test_df.drop(['Embarked'],axis=1,inplace=True) 
titanic_df.drop(['Pclass'],axis=1,inplace=True)
test_df.drop(['Pclass'],axis=1,inplace=True)
titanic_df.drop(['SibSp','ParchBinary','Parch','MasterMiss','FamilySize','SmallFamily','S','C','Fc'],axis=1,inplace=True)
test_df.drop(['SibSp','ParchBinary','Parch','MasterMiss','FamilySize','SmallFamily','S','C','Fc'],axis=1,inplace=True)

#titanic_df.drop(['PchM','Mq','SmallFamily','Alone','Fc'],axis=1,inplace=True)
#test_df.drop(['PchM','Mq','SmallFamily','Alone','Fc'],axis=1,inplace=True)
#titanic_df.drop(['C','MasterMiss','Parch','ParchBinary','S','Q','SibSp','FamilySize','BigFamily','Fc','Mq','PchM','Female','Age','Child'], axis=1,inplace=True)
#test_df.drop(['C','MasterMiss','Parch','ParchBinary','S','Q','SibSp','FamilySize','BigFamily','Fc','Mq','PchM','Female','Age','Child'], axis=1,inplace=True)
X_train = titanic_df.drop("Survived",axis=1)
Y_train = titanic_df["Survived"]
X_test  = test_df.drop("PassengerId",axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
X_train= min_max_scaler.fit_transform(X_train)
X_train=pd.DataFrame(X_train)
min_max_scaler = preprocessing.MinMaxScaler()
X_test= min_max_scaler.fit_transform(X_test)
X_test=pd.DataFrame(X_test)

model = LogisticRegression()
n_f=14 #number of features
rfe = RFE(model, n_f)#(fitting model, number of features)
fit = rfe.fit(X_train, Y_train)
print(fit.n_features_)#number
print(fit.support_)#selected
print(fit.ranking_)#rank
print(fit.score(X_train, Y_train))#fits score

X_train=fit.fit_transform(X_train,Y_train)
X_test=fit.transform(X_test)
titanic_df.drop('Survived',axis=1,inplace=True)
titanic_df=titanic_df.iloc[:,fit.support_]
#cross-validation parameter search


clf = RandomForestClassifier(n_estimators=20)

param_dist = {"max_depth": [3,4,5,6, None],
              "max_features": sp_randint(1, n_f),
              "min_samples_split": sp_randint(2, 4),
              "min_samples_leaf": sp_randint(1, n_f),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
#random_search.fit(X_train, Y_train)
#report(random_search.cv_results_)
clfe = ExtraTreesClassifier(n_estimators=20)

param_dist = {"max_depth": [3,4,5,6, None],
              "max_features": sp_randint(1, n_f),
              "min_samples_split": sp_randint(2, 4),
              "min_samples_leaf": sp_randint(1, n_f),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}


# run randomized search
n_iter_search = 20
random_search_e = RandomizedSearchCV(clfe, param_distributions=param_dist,
                                   n_iter=n_iter_search)
#voting

logreg = LogisticRegression()
rf=random_search #from cross-validation
extree = random_search_e
svc=svm.SVC()

ada=AdaBoostClassifier()

#vcr=VotingClassifier(estimators=[('lg',logreg),('rf',rf),('extree',extree),('svc',svc),('xg',xg),('knn',knn),('gb',gb)],
#                     voting='hard',weights=[2,1,1,1,1,1,1])

#vcr.fit(X_train, Y_train)

#Y_pred = vcr.predict(X_test)

#print('voting',vcr.score(X_train, Y_train))

logreg.fit(X_train, Y_train)
rf.fit(X_train, Y_train)
svc.fit(X_train, Y_train)
extree.fit(X_train, Y_train)
ada.fit(X_train, Y_train)

print('logreg',logreg.score(X_train, Y_train))
print('randforest',rf.score(X_train, Y_train))
print('extree',extree.score(X_train, Y_train))
print('svc',svc.score(X_train, Y_train))
print('ada',ada.score(X_train, Y_train))

#report(random_search.cv_results_)
#report(random_search_xgb.cv_results_)
#make prediction for second degree training
Y_pred_logreg = pd.Series(logreg.predict(X_train),name='l')
Y_pred_rf =pd.Series(rf.predict(X_train),name='r')
Y_pred_svc = pd.Series(svc.predict(X_train),name='s')
Y_pred_extree = pd.Series(extree.predict(X_train),name='e')
Y_pred_ada = pd.Series(ada.predict(X_train),name='g')

Y_test_logreg = pd.Series(logreg.predict(X_test),name='l')
Y_test_rf = pd.Series(rf.predict(X_test),name='r')
Y_test_svc = pd.Series(svc.predict(X_test),name='s')
Y_test_extree = pd.Series(extree.predict(X_test),name='e')
Y_test_ada = pd.Series(ada.predict(X_test),name='g')



sec_deg_train=pd.concat([Y_pred_logreg,Y_pred_rf,Y_pred_svc,Y_pred_extree],axis=1)
sec_deg_test=pd.concat([Y_test_logreg,Y_test_rf,Y_test_svc,Y_test_extree],axis=1)



#incase choosing randomforest
param_dist = {"max_depth": [3,4,5,6, None],
              "max_features": sp_randint(1, 4),
              "min_samples_split": sp_randint(2, 4),
              "min_samples_leaf": sp_randint(1, 4),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
random_search_e = RandomizedSearchCV(clfe, param_distributions=param_dist,
                                   n_iter=n_iter_search)
extree=random_search_e
rf=random_search
vcr=VotingClassifier(estimators=[('lg',logreg),('rf',rf),('extree',extree)],
                     voting='hard',weights=[2,1,1])

vcr.fit(sec_deg_train, Y_train)
vcr.fit(sec_deg_train, Y_train)
Y_pred = vcr.predict(sec_deg_test)

print(vcr.score(sec_deg_train,Y_train))
#coeff_df = DataFrame(sec_deg_train.columns)
#coeff_df.columns = ['Features']
#coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

# preview
#coeff_df
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic.csv', index=False)