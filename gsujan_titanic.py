# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas_profiling
train=pd.read_csv("../input/titanic/train.csv")

test=pd.read_csv("../input/titanic/test.csv")
#train.profile_report()
#train.info()
#test.profile_report()
def missing_percentage(df):

    """This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage"""

    ## the two following line may seem complicated but its actually very simple. 

    total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]

    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]

    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])
missing_percentage(train)
missing_percentage(test)
def missing_value_count(df,feature):

    total=pd.DataFrame(df.loc[:,feature].value_counts(dropna=False))

    percent=round(total/len(df)*100,2)

    total.columns = ["Total"]

    percent.columns = ['Percent']

    return  pd.concat([total,percent],axis=1)
missing_value_count(train,'Embarked')
train[train.Embarked.isnull()]
train.loc[(train['Survived'] == 1) & (train['Pclass'] == 1) & (train['Sex']=='female' )].loc[:,'Embarked'].value_counts()
train.Embarked.fillna('S',inplace=True)
#FOR imputation of cabin feature we concat the train and test set

survivers=train.Survived
train.drop('Survived',inplace=True,axis=1)
all_data=pd.concat([train,test],ignore_index=False)
all_data.Cabin.fillna('N',inplace=True)
all_data.Cabin=[i[0] for i in all_data.Cabin]
missing_value_count(all_data,'Cabin')
with_N=all_data[all_data.Cabin=='N']
without_N=all_data[all_data.Cabin!='N']
all_data.groupby('Cabin').Fare.mean().sort_values(ascending=False)
all_data.Cabin.value_counts(ascending=False)
def cabin_estimator(i):

    """Grouping cabin feature by the first letter"""

    a = 0

    if i<16:

        a = "G"

    elif i>=16 and i<27:

        a = "F"

    elif i>=27 and i<38:

        a = "T"

    elif i>=38 and i<47:

        a = "A"

    elif i>= 47 and i<53:

        a = "E"

    elif i>= 53 and i<54:

        a = "D"

    elif i>=54 and i<116:

        a = 'C'

    else:

        a = "B"

    return a
with_N['Cabin']=with_N.Fare.apply(lambda x: cabin_estimator(x))
with_N.Cabin.unique()
## getting back train. 

all_data = pd.concat([with_N, without_N], axis=0)

## PassengerId helps us separate train and test. 

all_data.sort_values(by = 'PassengerId', inplace=True)
all_data.Cabin.unique()
## Separating train and test from all_data. 

train = all_data[:891]#exclude 891 means up to 890 index value



test = all_data[891:]
train.info()
train['Survived']=survivers
test[test.Fare.isnull()]
missing_value=test[(test.Pclass==3)&(test.Sex=='male')&(test.Embarked=='S')].Fare.mean()
missing_value
test.Fare.fillna(missing_value,inplace=True)
import seaborn as sns

import matplotlib.pyplot as plt
sns.barplot(x = "Sex", 

                 y = "Survived", 

                 data=train, 

#                 linewidth=5,

 #                capsize = .05,



                )



plt.title("Survived/Non-Survived Passenger Gender Distribution", fontsize = 25,loc = 'center', pad = 40)
plt.subplots(figsize = (15,8))

ax = sns.countplot(x = "Sex", 

                   hue="Survived",

                   data = train, 

                   linewidth=4,

)

plt.title("Passenger Gender Distribution - Survived vs Not-survived", fontsize = 25, pad=40)

leg = ax.get_legend()

leg.set_title("Survived")

legs = leg.texts

legs[0].set_text("No")

legs[1].set_text("Yes")

plt.show()    
train.describe()
#Heatmap

#representing correlation 

plt.subplots(figsize = (15,12))

sns.heatmap(train.corr(),annot=True,cmap = 'RdBu',linewidths=.9,linecolor='grey', center = 0,square=True)

plt.title("Correlations Among Features", y = 1.03,fontsize = 20, pad = 40)
#Creating feature that conatins name length

train['name_length']=[len(i) for i in train.Name]
test['name_length']=[len(i) for i in test.Name]
def name_length_group(size):

    a=''

    if (size<=20):

        a='short'

    elif(size<=35):

        a='medium'

    elif (size<=45):

        a='good'

    else:

        a='long'

    return a

                
train['nlength_group']=train['name_length'].map(name_length_group)
test['nlength_group']=test['name_length'].map(name_length_group)
train['title']=[i.split('.')[0] for i in train.Name]
train['title']=[i.split(',')[1] for i in train.title]
test['title']=[i.split('.')[0] for i in test.Name]

test['title']=[i.split(',')[1] for i in test.title]
train['title']=[i.replace('Ms','Miss') for i in train.title]

train["title"] = [i.replace('Mlle', 'Miss') for i in train.title]

train["title"] = [i.replace('Mme', 'Mrs') for i in train.title]

train["title"] = [i.replace('Dr', 'rare') for i in train.title]

train["title"] = [i.replace('Col', 'rare') for i in train.title]

train["title"] = [i.replace('Major', 'rare') for i in train.title]

train["title"] = [i.replace('Don', 'rare') for i in train.title]

train["title"] = [i.replace('Jonkheer', 'rare') for i in train.title]

train["title"] = [i.replace('Sir', 'rare') for i in train.title]

train["title"] = [i.replace('Lady', 'rare') for i in train.title]

train["title"] = [i.replace('Capt', 'rare') for i in train.title]

train["title"] = [i.replace('the Countess', 'rare') for i in train.title]

train["title"] = [i.replace('Rev', 'rare') for i in train.title]

test['title'] = [i.replace('Ms', 'Miss') for i in test.title]

test['title'] = [i.replace('Dr', 'rare') for i in test.title]

test['title'] = [i.replace('Col', 'rare') for i in test.title]

test['title'] = [i.replace('Dona', 'rare') for i in test.title]

test['title'] = [i.replace('Rev', 'rare') for i in test.title]
test.title.unique()
train[['Sex','title']].groupby('title').count()
train['family_size']=train.SibSp+train.Parch+1

test['family_size']=test.SibSp+test.Parch+1
print(train.family_size.unique())

print(test.family_size.unique())
def family_group(size):

    a=''

    if (size<=1):

        a='loner'

    elif (size<=4):

        a='small'

    else:

        a='large'

    return a
train["family_group"]=train.family_size.map(family_group)

train.head()
test['family_group'] = test['family_size'].map(family_group)
train['is_alone'] = [1 if i<2 else 0 for i in train.family_size]

test['is_alone'] = [1 if i<2 else 0 for i in test.family_size]
train.drop(['Ticket'], axis=1, inplace=True)

test.drop(['Ticket'], axis=1, inplace=True)
train['calculated_fare']=train.Fare/train.family_size

train.head()
test['calculated_fare']=test.Fare/test.family_size
bins=[0,4,10,20,45,600]

group=['very_low','small','medium','high','very_high']

train['fare_group']=pd.cut(train['calculated_fare'],bins,labels=group)

test['fare_group']=pd.cut(test['calculated_fare'],bins,labels=group)
train.drop(['PassengerId'], axis=1, inplace=True)



test.drop(['PassengerId'], axis=1, inplace=True)
train = pd.get_dummies(train, columns=['title',"Pclass", 'Cabin','Embarked','nlength_group', 'family_group', 'fare_group'], drop_first=False)
test = pd.get_dummies(test, columns=['title',"Pclass",'Cabin','Embarked','nlength_group', 'family_group', 'fare_group'], drop_first=False)

train.drop(['family_size','Name', 'Fare','name_length'], axis=1, inplace=True)
test.drop(['Name','family_size',"Fare",'name_length'], axis=1, inplace=True)
train = pd.concat([train[["Survived", "Age", "Sex","SibSp","Parch"]], train.loc[:,"is_alone":]], axis=1)

test = pd.concat([test[["Age", "Sex"]], test.loc[:,"SibSp":]], axis=1)
from sklearn.ensemble import RandomForestRegressor
def completing_age(df):

    ## gettting all the features except survived

    age_df = df.loc[:,"Age":] 

    

    temp_train = age_df.loc[age_df.Age.notnull()] ## df with age values

    temp_test = age_df.loc[age_df.Age.isnull()] ## df without age values

    

    y = temp_train.Age.values ## setting target variables(age) in y 

    x = temp_train.loc[:, "Sex":].values

    

    rfr = RandomForestRegressor(n_estimators=1500, n_jobs=-1)

    rfr.fit(x, y)

    

    predicted_age = rfr.predict(temp_test.loc[:, "Sex":])

    

    df.loc[df.Age.isnull(), "Age"] = predicted_age

    



    return df
train.Sex=train.Sex.apply(lambda x : 0 if x=='female' else 1)

test.Sex=test.Sex.apply(lambda x : 0 if x=='female' else 1)
completing_age(train)
completing_age(test);
## create bins for age

def age_group_fun(age):

    a = ''

    if age <= 1:

        a = 'infant'

    elif age <= 4: 

        a = 'toddler'

    elif age <= 13:

        a = 'child'

    elif age <= 18:

        a = 'teenager'

    elif age <= 35:

        a = 'Young_Adult'

    elif age <= 45:

        a = 'adult'

    elif age <= 55:

        a = 'middle_aged'

    elif age <= 65:

        a = 'senior_citizen'

    else:

        a = 'old'

    return a

        

## Applying "age_group_fun" function to the "Age" column.

train['age_group'] = train['Age'].map(age_group_fun)

test['age_group'] = test['Age'].map(age_group_fun)



## Creating dummies for "age_group" feature. 

train = pd.get_dummies(train,columns=['age_group'], drop_first=True)

test = pd.get_dummies(test,columns=['age_group'], drop_first=True);



train.drop('Age', axis=1, inplace=True)

test.drop('Age', axis=1, inplace=True)
X = train.drop(['Survived'], axis = 1)

y = train["Survived"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .33, random_state=0)
headers = X_train.columns 



X_train.head()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



## transforming "train_x"

X_train = sc.fit_transform(X_train)

## transforming "test_x"

X_test = sc.transform(X_test)



## transforming "The testset"

test = sc.transform(test)
pd.DataFrame(X_train, columns=headers).head()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_absolute_error,accuracy_score

logreg=LogisticRegression(solver='liblinear',penalty='l1')

logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)

print('Our accuracy score is :{}'.format(round(accuracy_score(y_pred,y_test),4)))
#use of cross-validation

from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score

cv=StratifiedShuffleSplit(n_splits=10,test_size=.25,random_state=0)

## saving the feature names for decision tree display

column_names = X.columns
X = sc.fit_transform(X)

accuracies = cross_val_score(LogisticRegression(solver='liblinear'), X,y, cv  = cv)
print ("Cross-Validation accuracy scores:{}".format(accuracies))

print ("Mean Cross-Validation accuracy score: {}".format(round(accuracies.mean(),5)))
from sklearn.model_selection import GridSearchCV, StratifiedKFold

C_vals = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,16.5,17,17.5,18]

penalties = ['l1','l2']

cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25)

param = {'penalty': penalties, 'C': C_vals}

logreg = LogisticRegression(solver='liblinear')

grid = GridSearchCV(estimator=LogisticRegression(),param_grid = param,scoring = 'accuracy',n_jobs =-1,cv = cv)

grid.fit(X, y)
## Getting the best of everything. 

print (grid.best_score_)

print (grid.best_params_)

print(grid.best_estimator_)
### Using the best parameters from the grid-search.

logreg_grid = grid.best_estimator_

logreg_grid.score(X,y)
from sklearn.neighbors import KNeighborsClassifier

## calling on the model oject. 

knn = KNeighborsClassifier(metric='minkowski', p=2)

## knn classifier works by doing euclidian distance 





## doing 10 fold staratified-shuffle-split cross validation 

cv = StratifiedShuffleSplit(n_splits=10, test_size=.25, random_state=2)



accuracies = cross_val_score(knn, X,y, cv = cv, scoring='accuracy')

print ("Cross-Validation accuracy scores:{}".format(accuracies))

print ("Mean Cross-Validation accuracy score: {}".format(round(accuracies.mean(),3)))
k_range = range(1,31)

k_scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, X,y, cv = cv, scoring = 'accuracy')

    k_scores.append(scores.mean())

print("Accuracy scores are: {}\n".format(k_scores))

print ("Mean accuracy score: {}".format(np.mean(k_scores)))
from matplotlib import pyplot as plt

plt.plot(k_range, k_scores)
from sklearn.model_selection import GridSearchCV

## trying out multiple values for k

k_range = range(1,31)

## 

weights_options=['uniform','distance']

# 

param = {'n_neighbors':k_range, 'weights':weights_options}

## Using startifiedShufflesplit. 

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

# estimator = knn, param_grid = param, n_jobs = -1 to instruct scikit learn to use all available processors. 

knn_grid = GridSearchCV(KNeighborsClassifier(), param,cv=cv,verbose = False, n_jobs=-1)

## Fitting the model. 

knn_grid.fit(X,y)

knn_estimator= knn_grid.best_estimator_

knn_estimator.score(X,y)
from sklearn.model_selection import RandomizedSearchCV

## trying out multiple values for k

k_range = range(1,31)

## 

weights_options=['uniform','distance']

# 

param = {'n_neighbors':k_range, 'weights':weights_options}

## Using startifiedShufflesplit. 

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30)

# estimator = knn, param_grid = param, n_jobs = -1 to instruct scikit learn to use all available processors. 

## for RandomizedSearchCV, 

rgrid = RandomizedSearchCV(KNeighborsClassifier(), param,cv=cv,verbose = False, n_jobs=-1, n_iter=40)

## Fitting the model. 

rgrid.fit(X,y)

knn_restimator = rgrid.best_estimator_

knn_restimator.score(X,y)
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



gaussian = GaussianNB()

gaussian.fit(X, y)

y_pred = gaussian.predict(X_test)

gaussian_accy = round(accuracy_score(y_pred, y_test), 3)

print(gaussian_accy)
from sklearn.svm import SVC

Cs = [0.001, 0.01, 0.1, 1,1.5,2,2.5,3,4,5, 10] ## penalty parameter C for the error term. 

gammas = [0.0001,0.001, 0.01, 0.1, 1]

param_grid = {'C': Cs, 'gamma' : gammas}

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

sv_grid = GridSearchCV(SVC(kernel = 'rbf', probability=True), param_grid, cv=cv) ## 'rbf' stands for gaussian kernel

sv_grid.fit(X,y)

sv_estimator = sv_grid.best_estimator_

sv_estimator.score(X,y)
from sklearn.tree import DecisionTreeClassifier

max_depth = range(1,30)

max_feature = [21,22,23,24,25,26,28,29,30,'auto']

criterion=["entropy", "gini"]



param = {'max_depth':max_depth, 

         'max_features':max_feature, 

         'criterion': criterion}

dt_grid = GridSearchCV(DecisionTreeClassifier(), 

                                param_grid = param, 

                                 verbose=False, 

                                 cv=StratifiedKFold(n_splits=20, random_state=15, shuffle=True),

                                n_jobs = -1)

dt_grid.fit(X, y) 

dt_estimator = dt_grid.best_estimator_ 

dt_estimator.score(X,y)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit

from sklearn.ensemble import RandomForestClassifier

n_estimators = [140,145,150,155,160];

max_depth = range(1,10);

criterions = ['gini', 'entropy'];

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)





parameters = {'n_estimators':n_estimators,

              'max_depth':max_depth,

              'criterion': criterions

              

        }

rf_grid = GridSearchCV(estimator=RandomForestClassifier(max_features='auto'),

                                 param_grid=parameters,

                                 cv=cv,

                                 n_jobs = -1)

rf_grid.fit(X,y) 

rf_estimator=rf_grid.best_estimator_

rf_estimator.score(X,y)
from sklearn.ensemble import BaggingClassifier

n_estimators = [10,30,50,70,80,150,160, 170,175,180,185];

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)



parameters = {'n_estimators':n_estimators,

              

        }

grid = GridSearchCV(BaggingClassifier(base_estimator= None, ## If None, then the base estimator is a decision tree.

                                      bootstrap_features=False),

                                 param_grid=parameters,

                                 cv=cv,

                                 n_jobs = -1)

grid.fit(X,y) 

grid.best_score_

grid.best_params_

grid.best_estimator_

bagging_grid = grid.best_estimator_

bagging_grid.score(X,y)

bagging_grid = grid.best_estimator_

bagging_estimator=grid.best_estimator_

bagging_grid.score(X,y)


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit,learning_curve

def plot_learning_curve(estimator1, X, y, ylim=None, cv=None,n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    f, ax1 = plt.subplots()

    # First Estimator

    train_sizes, train_scores, test_scores = learning_curve(

        estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax1.set_title("bagging classifier Learning Curve", fontsize=14)

    ax1.set_xlabel('Training size (m)')

    ax1.set_ylabel('Score')

    ax1.grid(True)

    ax1.legend(loc="best")

    return plt

cv = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

plot_learning_curve(bagging_estimator, X, y, cv=cv, n_jobs=4)


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit,learning_curve

def plot_learning_curve(estimator1, X, y, ylim=None, cv=None,n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    f, ax1 = plt.subplots()

    # First Estimator

    train_sizes, train_scores, test_scores = learning_curve(

        estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax1.set_title("random forest classifier Learning Curve", fontsize=14)

    ax1.set_xlabel('Training size (m)')

    ax1.set_ylabel('Score')

    ax1.grid(True)

    ax1.legend(loc="best")

    return plt

cv = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

plot_learning_curve(rf_estimator, X, y, cv=cv, n_jobs=4)


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit,learning_curve

def plot_learning_curve(estimator1, X, y, ylim=None, cv=None,n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    f, ax1 = plt.subplots()

    # First Estimator

    train_sizes, train_scores, test_scores = learning_curve(

        estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax1.set_title("decision tree classifier Learning Curve", fontsize=14)

    ax1.set_xlabel('Training size (m)')

    ax1.set_ylabel('Score')

    ax1.grid(True)

    ax1.legend(loc="best")

    return plt

cv = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

plot_learning_curve(dt_estimator, X, y, cv=cv, n_jobs=4)


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit,learning_curve

def plot_learning_curve(estimator1, X, y, ylim=None, cv=None,n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    f, ax1 = plt.subplots()

    # First Estimator

    train_sizes, train_scores, test_scores = learning_curve(

        estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax1.set_title("SVM classifier Learning Curve", fontsize=14)

    ax1.set_xlabel('Training size (m)')

    ax1.set_ylabel('Score')

    ax1.grid(True)

    ax1.legend(loc="best")

    return plt

cv = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

plot_learning_curve(sv_estimator, X, y, cv=cv, n_jobs=4)


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit,learning_curve

def plot_learning_curve(estimator1, X, y, ylim=None, cv=None,n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    f, ax1 = plt.subplots()

    # First Estimator

    train_sizes, train_scores, test_scores = learning_curve(

        estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax1.set_title("knn classifier Learning Curve", fontsize=14)

    ax1.set_xlabel('Training size (m)')

    ax1.set_ylabel('Score')

    ax1.grid(True)

    ax1.legend(loc="best")

    return plt

cv = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

plot_learning_curve(knn_restimator, X, y, cv=cv, n_jobs=4)
test_prediction = sv_estimator.predict(test)
test_data=pd.read_csv("../input/titanic/test.csv")
submission = pd.DataFrame({

        "PassengerId":test_data.PassengerId,

        "Survived": test_prediction

    })
submission.PassengerId = submission.PassengerId.astype(int)

submission.Survived = submission.Survived.astype(int)
submission.to_csv("titanic_1st_submission.csv", index=False)