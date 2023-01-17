#Importing Libraries



import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
#Loading the dataset



df_train = pd.read_csv('../input/titanic/train.csv') 

df_test  = pd.read_csv('../input/titanic/test.csv')
#Viewing the dataset



display(df_train.head())

display()

display(df_test.head())
#Preprocessing needs to be done on both test and train data. So we will concatenate both test and train data.

#Creating a new column in test data

df_test['Survived'] = 999      #Assigning some random value to the test dataset
#Now both test and train can be concatenated



df = pd.concat([df_train , df_test] , axis = 0)

print(df_train.shape)

print(df_test.shape)

print('Combined dataframe shape :',df.shape)
df.info()
plt.style.use('ggplot')

print('Null Values :',df['Pclass'].isnull().sum())  #To find the number of null values

df['Pclass'].value_counts(sort = True).plot(kind = 'barh' ,title = 'Passenger Class')
#Null values

print('The number of null values in age columns',df['Age'].isnull().sum())

print('The % of null values in age columns',round(df['Age'].isnull().mean()* 100,2)) 
#Age column has lots of null values. 

#Around 20 % of the values are missing.
df['Age'].plot(kind='box')

print('Range of age is between',df['Age'].min() , 'and' ,df['Age'].max())
#Eventhough the box plot shows outlier we wont remove any outliers in age column , because maximum age of the person in titanic was 80
#Null values

print('The number of null values in cabin columns',df['Cabin'].isnull().sum())

print('The % of null values in cabin columns',round(df['Cabin'].isnull().mean()* 100,2)) 
#Dropping Cabin column

df.drop(columns = 'Cabin' , inplace = True)
df['Embarked'].value_counts().plot(kind=  'bar' , rot = 0 )
print('The number of null values in Embarked is ', df['Embarked'].isnull().sum())
#NaN values are replaced with Southampton

df['Embarked'].replace({np.nan:'S'} , inplace = True)
#Only one value of fare is missing.

df[df['Fare'].isnull()]
plt.figure(figsize = (10,4))

plt.subplot(1,2,1)

sns.distplot(df['Fare'])

plt.subplot(1,2,2)

sns.boxplot(df['Fare'])
df['Name'].isnull().sum()

#There are no missing values in name
df['Name'].head()

#Name is useless feature , but Title in the name can give some additional information for prediction
def GetTitle_temp(name):

    fname_title = name.split(',')[1]

    title = fname_title.split('.')[0]

    title = title.strip().lower()

    return title



df.Name.map(GetTitle_temp).value_counts()
def GetTitle(name):

    titles = {'mr' : 'Mr', 

               'mrs' : 'Mrs', 

               'miss' : 'Miss', 

               'master' : 'Master',

               'don' : 'Mr',

               'rev' : 'Mr',

               'dr' : 'Mr',

               'mme' : 'Mrs',

               'ms' : 'Mrs',

               'major' : 'Mr',

               'lady' : 'Miss',

               'sir' : 'Mr',

               'mlle' : 'Miss',

               'col' : 'Mr',

               'capt' : 'Mr',

               'the countess' : 'Miss',

               'jonkheer' : 'Mr',

               'dona' : 'Miss'

                 }

    fname_title = name.split(',')[1]

    title = fname_title.split('.')[0]

    title = title.strip().lower()

    return titles[title]



df['Name'] = df.Name.map(GetTitle)
sns.countplot(df['Name'])
df['Parch'].value_counts()
df['SibSp'].value_counts()
#Combining Sibsp and Parch columns to form number of people accompanying

df['Accomp'] = df['SibSp'] + df['Parch']

df.drop(columns = ['SibSp' , 'Parch'] , inplace = True)
df['Sex'].value_counts()
sns.countplot(df['Sex'])
#Encoding the Sex column

df['Sex'] = df['Sex'].map({'female':0 , 'male':1 })
#Dropping the ticket number

df.drop(columns = ['Ticket'] , inplace = True)
#Viewing the Data

df.head()
cat = pd.get_dummies(df[['Embarked' , 'Name']] , drop_first=True)
df = pd.concat([df,cat] , axis = 1)

df.drop(columns = ['Embarked' , 'Name'] , inplace = True)    #Dropping the original columns
#After Encoding

df.head()
#Checking the percentage of missing values:

df.isnull().mean()*100
import missingno as msno

msno.matrix(df)
msno.heatmap(df)   #Heatmap to find any correlation of missing values between other missing columns

#From this heatpmap we can see that there is no correlation between missing values
msno.dendrogram(df)
#Imputing using KNN : 



from fancyimpute import KNN

knn_imputer = KNN()

df_knn = df.copy()

df_knn.iloc[:,:] = knn_imputer.fit_transform(df_knn)
#Imputing using MICE



from fancyimpute import IterativeImputer

MICE_imputer = IterativeImputer()

df_mice = df.copy()

df_mice.iloc[:,:] = knn_imputer.fit_transform(df_mice)
sns.kdeplot(df['Age'] , c = 'r' , label = 'No imputation')

sns.kdeplot(df_knn['Age'] , c = 'g' , label = 'KNN imputation')

sns.kdeplot(df_mice['Age'] , c = 'b' , label = 'MICE imputation')

sns.kdeplot(df['Age'].fillna(df['Age'].mean()) , c = 'k' , label = 'Fillna_Mean')

#Distribution of the columns are maintained while using this fancy imputation techniques

#The black kde plot shows how distribution when when we fill the null values with the mean value
df_mice.head()
df_mice.Survived.unique()
#Spiltting Test and Train Datas:

dfm_train = df_mice[df_mice['Survived'] != 999]

dfm_test  = df_mice[df_mice['Survived'] == 999]



print('Train Shape :',dfm_train.shape)

print('Test Shape :',dfm_test.shape)



dfm_test.drop(columns = 'Survived' , inplace = True)
dfm_train.head()
import xgboost as xgb

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

X, y = dfm_train.drop(columns = ['Survived','PassengerId']) , dfm_train['Survived']

X_train, X_test, y_train, y_test= train_test_split(X, y,test_size=0.2, random_state=123)

xg_cl = xgb.XGBClassifier(objective='binary:logistic',

n_estimators=20, seed=123)



xg_cl.fit(X_train, y_train)

preds = xg_cl.predict(X_test)

accuracy = float(np.sum(preds==y_test))/y_test.shape[0]

print("accuracy: %f" % (accuracy))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test , preds)
y_pred =  xg_cl.predict(dfm_test.drop(columns='PassengerId')).astype('int')



results = pd.DataFrame(data={'PassengerId':dfm_test['PassengerId'].astype('int'), 'Survived':y_pred})

results.to_csv('Titanic Prediction_XGB.csv', index=False)
import xgboost as xgb

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.model_selection import StratifiedKFold

X, y = dfm_train.drop(columns = ['Survived','PassengerId']) , dfm_train['Survived']

X_train, X_test, y_train, y_test= train_test_split(X, y,test_size=0.2, random_state=123)



# A parameter grid for XGBoost

params = {

        'min_child_weight': [1, 3,5,7 , 10],

        'gamma': [0.5, 1, 1.5, 2,3,4, 5],

        'subsample': [0.6,0.7, 0.8,0.9, 1.0],

        'colsample_bytree': [0.6,0.7, 0.8,0.9, 1.0],

        'max_depth': [3, 4, 5 , 6]

        }



xgb = xgb.XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',

                    silent=True, nthread=1)



folds = 3

param_comb = 5



skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)



random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,y), verbose=3, random_state=1001 )



random_search.fit(X, y)

random_search.best_params_
y_test = random_search.predict(dfm_test.drop(columns='PassengerId')).astype('int')



results = pd.DataFrame(data={'PassengerId':dfm_test['PassengerId'].astype('int'), 'Survived':y_test})

results.to_csv('Titanic Prediction_XGB_hp.csv', index=False)



#Got final score of 0.79904
#Spiltting Test and Train Datas:

dfm_train = df_knn[df_mice['Survived'] != 999]

dfm_test  = df_knn[df_mice['Survived'] == 999]



print('Train Shape :',dfm_train.shape)

print('Test Shape :',dfm_test.shape)



dfm_test.drop(columns = 'Survived' , inplace = True)
import xgboost as xgb

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.model_selection import StratifiedKFold

X, y = dfm_train.drop(columns = ['Survived','PassengerId']) , dfm_train['Survived']

X_train, X_test, y_train, y_test= train_test_split(X, y,test_size=0.2, random_state=123)



# A parameter grid for XGBoost

params = {

        'min_child_weight': [1, 3,5,7 , 10],

        'gamma': [0.5, 1, 1.5, 2,3,4, 5],

        'subsample': [0.6,0.7, 0.8,0.9, 1.0],

        'colsample_bytree': [0.6,0.7, 0.8,0.9, 1.0],

        'max_depth': [3, 4, 5 , 6]

        }



xgb = xgb.XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',

                    silent=True, nthread=1)



folds = 3

param_comb = 5



skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)



random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,y), verbose=3, random_state=1001 )



random_search.fit(X, y)

y_test = random_search.predict(dfm_test.drop(columns='PassengerId')).astype('int')



results = pd.DataFrame(data={'PassengerId':dfm_test['PassengerId'].astype('int'), 'Survived':y_test})

results.to_csv('Titanic Predictionkn.csv', index=False)



#Got a score of 0.7666
#Spiltting Test and Train Datas:

dfm_train = df_mice[df_mice['Survived'] != 999]

dfm_test  = df_mice[df_mice['Survived'] == 999]



print('Train Shape :',dfm_train.shape)

print('Test Shape :',dfm_test.shape)



dfm_test.drop(columns = 'Survived' , inplace = True)
#KNN randomized search



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import RandomizedSearchCV , GridSearchCV

from scipy.stats import randint as sp_randint



knn = KNeighborsClassifier()



params = {

    'n_neighbors' : sp_randint(1 , 20) ,

    'p' : sp_randint(1 , 5) ,

}



rsearch_knn = RandomizedSearchCV(knn , param_distributions = params , cv = 3 , random_state= 3  , n_jobs = -1 , return_train_score=True)



rsearch_knn.fit(X , y)
rsearch_knn.best_params_
#Random Forest randomized search

from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(random_state=3)

params = { 'n_estimators' : sp_randint(50 , 200) , 

           'max_features' : sp_randint(1 , 12) ,

           'max_depth' : sp_randint(2,10) , 

           'min_samples_split' : sp_randint(2,20) ,

           'min_samples_leaf' : sp_randint(1,20) ,

           'criterion' : ['gini' , 'entropy']

    

}



rsearch_rfc = RandomizedSearchCV(rfc , param_distributions= params , n_iter= 200 , cv = 3 , scoring='roc_auc' , random_state= 3 , return_train_score=True , n_jobs=-1)



rsearch_rfc.fit(X,y)
rsearch_rfc.best_params_
from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver = 'liblinear')

knn = KNeighborsClassifier(**rsearch_knn.best_params_)

rfc = RandomForestClassifier(**rsearch_rfc.best_params_)



clf = VotingClassifier(estimators=[('lr' ,lr) , ('knn' , knn) , ('rfc' , rfc)] , voting = 'soft')



clf.fit(X , y)
y_test = clf.predict(dfm_test.drop(columns='PassengerId')).astype('int')



results = pd.DataFrame(data={'PassengerId':dfm_test['PassengerId'].astype('int'), 'Survived':y_test})

results.to_csv('Titanic Prediction_Stack.csv', index=False)

X, y = dfm_train.drop(columns = ['Survived','PassengerId']) , dfm_train['Survived']

X_train, X_test, y_train, y_test= train_test_split(X, y,test_size=0.2, random_state=123)
RFM = RandomForestClassifier(criterion='gini',

                                           n_estimators=1750,

                                           max_depth=7,

                                           min_samples_split=6,

                                           min_samples_leaf=6,

                                           max_features='auto',

                                           oob_score=True,

                                           random_state=123,

                                           n_jobs=-1,

                                           verbose=1) 



RFM.fit(X,y)
y_pred = RFM.predict(X_test)
y_test = RFM.predict(dfm_test.drop(columns='PassengerId')).astype('int')



results = pd.DataFrame(data={'PassengerId':dfm_test['PassengerId'].astype('int'), 'Survived':y_test})

results.to_csv('Titanic PredictionRFM.csv', index=False)

X, y = dfm_train.drop(columns = ['Survived','PassengerId']) , dfm_train['Survived']

X_train, X_test, y_train, y_test= train_test_split(X, y,test_size=0.2, random_state=123)
from sklearn.ensemble import ExtraTreesClassifier

import sklearn.model_selection as model_selection



clf_ET = ExtraTreesClassifier(random_state=0, bootstrap=True, oob_score=True)



sss = model_selection.StratifiedShuffleSplit(n_splits=10, test_size=0.33, random_state= 0)

sss.get_n_splits(X, y)



parameters = {'n_estimators' : np.r_[10:210:10],

              'max_depth': np.r_[1:6]

             }



grid = model_selection.GridSearchCV(clf_ET, param_grid=parameters, scoring = 'accuracy', cv = sss, return_train_score=True, n_jobs=4, verbose=2)

grid.fit(X,y)
y_test = RFM.predict(dfm_test.drop(columns='PassengerId')).astype('int')



results = pd.DataFrame(data={'PassengerId':dfm_test['PassengerId'].astype('int'), 'Survived':y_test})

results.to_csv('Titanic PredictionETC.csv', index=False)