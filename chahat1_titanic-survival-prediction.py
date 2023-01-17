import numpy as np 

import pandas as pd 

import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score



import os

print(os.listdir("../input"))
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

combine_data = [train_data,test_data]
train_data.head()
test_data.head()
train_data.shape, test_data.shape
train_data.describe()
train_data.info()
#Extracting the 'Title' from the 'Name' Column using regular expresion to see its relation with the 'Survived' column

for data in combine_data:

    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    print(data['Title'].value_counts())
for data in combine_data:

    print(pd.crosstab(data['Title'],data['Sex']))

    print('-'*25)
 #Replacing all titles with Mr, Mrs, Miss and Master only



def replace_titles(x):

    title=x['Title']

    

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col','Sir']:

        return 'Mr'

    elif title in ['Countess', 'Mme','Lady','Dona']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title == 'Dr':

        if x['Sex'] == 'male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title

    



for data in combine_data:    

    data['Title'] = data.apply(replace_titles, axis=1)
for data in combine_data:    

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1 
for data in combine_data:    

    data['isAlone'] = 1

    data.loc[data['FamilySize'] > 1,'isAlone'] = 0
for data in combine_data:    

    data['Deck'] = data.Cabin.str.extract('([A-Z])',expand = False)
train_data.shape, test_data.shape
#Columns PassengerId and Ticket are not important ,hence we remove them.

#Title has been extracted from Name column and Deck from Cabin column,thus deleting these extra columns

#SibSp and Parch are transformed to form FamilySize, so we can delete them.



columns_removed = ['PassengerId','Name','SibSp','Parch','Ticket','Cabin']

for data in combine_data:    

    data.drop(columns_removed, axis=1,inplace = True)
train_data.head()
print(train_data.isnull().sum())

print("-"*20)

print(test_data.isnull().sum())
#Fill 'X' for NaN values



for data in combine_data:    

    data['Deck'].fillna('X', inplace=True)
#Fill NaN values with the most common occurrence(mode) of the Embarked column



train_data['Embarked'].fillna(train_data['Embarked'].mode()[0],inplace=True)
#Fill NaN values with the mean of the Fare column



test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression



for data in combine_data:

    temp_data = data.copy()

    if 'Survived' in data:

        temp_data.drop('Survived',axis=1,inplace = True)

        

    #LabelEncoding the Categorical features

    le = LabelEncoder()

    cols = ['Sex','Embarked','Deck','Title']

    for col in cols:

        temp_data[col] = le.fit_transform(temp_data[col])  



    # Separating data with null values from dataset

    null_rows = temp_data.loc[temp_data.isnull().any(axis=1),]

    new_data = temp_data.drop(null_rows.index)



    null_cols = ['Age']   #Columns with null values

    n = len(null_cols)



    for col in null_cols:



        #forming data for training

        y_train = new_data[col]                                 

        x_train = new_data.drop(col, axis=1)



        #Preparing data for testing

        y_test = null_rows[col]   # contains all null values , whose prediction is to be done using KNN

        x_test =  null_rows.drop(col, axis=1)



        #Applying Linear Regression algorithm to predict the missing values in the Age column

        indices = x_test.index

        lr = LinearRegression()

        lr.fit(x_train, y_train)

        predicted = lr.predict(x_test)

        predicted_rounded = [int(round(p)) for p in predicted ] #rounding to the nearest integer



        #Replacing the NaN values with the predicted values

        for i in range(0,len(indices)):

            data.loc[indices[i],col] = predicted_rounded[i]
print(train_data.isnull().sum())

print("-"*20)

print(test_data.isnull().sum())
for data in combine_data:

    data['Sex'] = data['Sex'].map( {'female': 0, 'male': 1})

    data['Title'] = data['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3})

    data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2})

    data['Deck'] = data['Deck'].map( {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F':5, 'G': 6, 'T': 7, 'X':8})
print(train_data.head())

print("-"*20)

print(test_data.head())
age_band= pd.cut(train_data['Age'], 5)  

age_band.value_counts()
for data in combine_data:

    data.loc[data['Age'] <= 16.336, 'Age'] = 0

    data.loc[(data['Age'] > 16.336) & (data['Age'] <= 32.252), 'Age'] = 1

    data.loc[(data['Age'] > 32.252) & (data['Age'] <= 48.168), 'Age'] = 2

    data.loc[(data['Age'] > 48.168) & (data['Age'] <= 64.084), 'Age'] = 3

    data.loc[ data['Age'] > 64.084, 'Age'] = 4

    data['Age'] = data['Age'].astype(int)
fare_band= pd.qcut(train_data['Fare'],5)

fare_band.value_counts()
for data in combine_data:

    data.loc[data['Fare'] <= 7.85, 'Fare'] = 0

    data.loc[(data['Fare'] > 7.85) & (data['Fare'] <= 10.5), 'Fare'] = 1

    data.loc[(data['Fare'] > 10.5) & (data['Fare'] <= 21.679), 'Fare']   = 2

    data.loc[(data['Fare'] >21.679) & (data['Fare'] <= 39.688), 'Fare'] = 3

    data.loc[ data['Fare'] >= 39.688, 'Fare'] = 4

    data['Fare'] = data['Fare'].astype(int)
print(train_data.head())

print("-"*20)

print(test_data.head())
train_labels = train_data['Survived']

train_data.drop('Survived', axis = 1, inplace = True)

train_data.shape, train_labels.shape
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(train_data,train_labels, test_size = 0.20, random_state =99)

x_train.shape, y_train.shape, x_test.shape, y_test.shape
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier



from sklearn.model_selection import cross_val_score
model = GaussianNB()

model.fit(x_train,y_train.values.ravel())

predicted= model.predict(x_test)

accuracy_score(y_test, predicted)
knn_model = KNeighborsClassifier()

knn_model.fit(x_train,y_train.values.ravel())

predicted= knn_model.predict(x_test)

accuracy_score(y_test, predicted)
rfc_model = RandomForestClassifier(n_estimators = 100,random_state = 99)

rfc_model.fit(x_train,y_train)

predicted = rfc_model.predict(x_test)

accuracy_score(y_test,predicted)
lr_model = LogisticRegression(solver = 'lbfgs')

lr_model.fit(x_train,y_train.values.ravel())

lr_predicted = lr_model.predict(x_test)

accuracy_score(y_test, lr_predicted)
svc_model = SVC(C= 0.7, gamma= 0.2, kernel= 'rbf')

svc_model.fit(x_train,y_train.values.ravel())

svc_predicted = svc_model.predict(x_test)

accuracy_score(y_test, svc_predicted)
dtree_model = DecisionTreeClassifier()

dtree_model.fit(x_train,y_train.values.ravel())

dtree_predicted = dtree_model.predict(x_test)

accuracy_score(y_test, dtree_predicted)
from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(x_train,y_train.values.ravel())

xgb_predicted = xgb.predict(x_test)

accuracy_score(y_test, xgb_predicted)
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,

                     hidden_layer_sizes=(9, 2), random_state=1)



mlp.fit(x_train,y_train.values.ravel())                         

mlp_predicted = mlp.predict(x_test)

accuracy_score(y_test, mlp_predicted)
abc = AdaBoostClassifier(n_estimators = 500,

                         learning_rate = 0.1,random_state = 100)

abc_model = abc.fit(x_train, y_train)

y_pred = abc_model.predict(x_test)

accuracy_score(y_test, y_pred)
gbc = GradientBoostingClassifier(learning_rate = 0.1,random_state = 100, n_estimators=200)

gbc.fit(x_train, y_train)

y_pred = gbc.predict(x_test)

accuracy_score(y_test, y_pred)
etc = ExtraTreesClassifier(n_estimators = 200, max_depth = 10, random_state = 100)

etc.fit(x_train, y_train)

y_pred = etc.predict(x_test)

accuracy_score(y_test, y_pred)
from sklearn.model_selection import GridSearchCV
knn = KNeighborsClassifier()

param_grid = dict(leaf_size =list(range(3,10)), n_neighbors = list(range(1,10)), weights = ['uniform'])



knn_grid = GridSearchCV(knn, param_grid, cv=10, verbose=1, scoring='accuracy')

knn_grid.fit(train_data, train_labels.values.ravel())

print(knn_grid.best_score_)

print(knn_grid.best_params_)
rfc = RandomForestClassifier()

param_grid = {'max_depth': [3,5,6,7,8], 'max_features': [6,7,8,9],  

              'min_samples_split': [5,6,7,8],'n_estimators':[10,50]}



rf_grid = GridSearchCV(rfc, param_grid, cv=10, refit=True, verbose=1)

rf_grid.fit(train_data, train_labels.values.ravel())

print(rf_grid.best_score_)

print(rf_grid.best_params_)
svc = SVC()

param_grid = {'C': [0.1,10, 100, 1000,5000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}

svc_grid = GridSearchCV(SVC(), param_grid, cv=10, refit=True, verbose=1)

svc_grid.fit(train_data, train_labels.values.ravel())

print(svc_grid.best_score_)

print(svc_grid.best_params_)
dtree = DecisionTreeClassifier()



param_grid = {'min_samples_split': [4,7,10,12],'max_depth': [2,4,6,8,10,15,20],'criterion': ['gini', 'entropy']}

dtree_grid = GridSearchCV(dtree, param_grid, cv=10, refit=True, verbose=1)

dtree_grid.fit(train_data, train_labels.values.ravel())

print(dtree_grid.best_score_)

print(dtree_grid.best_params_)
xgb = XGBClassifier()

param_grid = {'max_depth': [5,6,7,8], 'gamma': [0.5,1, 2, 4], 'learning_rate': [0.1, 0.2, 0.3, 0.5]}



xgb_grid = GridSearchCV(xgb, param_grid, cv=10, refit=True, verbose=1)

xgb_grid.fit(train_data, train_labels.values.ravel())

print(xgb_grid.best_score_)

print(xgb_grid.best_params_)
ada = AdaBoostClassifier()

param_grid = {'n_estimators': [30, 50, 100], 'learning_rate': [0.08, 0.1, 0.2]}



ada_grid = GridSearchCV(ada, param_grid, cv=10, refit=True, verbose=1)

ada_grid.fit(train_data, train_labels.values.ravel())

print(ada_grid.best_score_)

print(ada_grid.best_params_)
gbc = GradientBoostingClassifier()

param_grid = {'n_estimators': [50, 100], 'min_samples_split': [3, 4, 5, 6, 7],

              'max_depth': [3, 4, 5, 6]}



gbc_grid = GridSearchCV(gbc, param_grid, cv=10, refit=True, verbose=1)

gbc_grid.fit(train_data, train_labels.values.ravel())

print(gbc_grid.best_score_)

print(gbc_grid.best_params_)
ext = ExtraTreesClassifier(random_state = 100)

param_grid = {'max_depth': [7,8,9,10],'max_features': [7,8,9],

              'n_estimators': [50,100, 200,300]}



ext_grid = GridSearchCV(ext, param_grid, cv=10, refit=True, verbose=1)

ext_grid.fit(train_data, train_labels.values.ravel())

print(ext_grid.best_score_)

print(ext_grid.best_params_)
from sklearn.ensemble import VotingClassifier



etc = ExtraTreesClassifier(random_state=9)

gbc = GradientBoostingClassifier(random_state = 9)

rf = RandomForestClassifier(random_state=9)

knn = KNeighborsClassifier()

svc = SVC()

xgb = XGBClassifier()

dtree = DecisionTreeClassifier()

ada =  AdaBoostClassifier(random_state = 9)





vot  = VotingClassifier(estimators=[('gbc',gbc),('rf', rf),('svc',svc),('xgb',xgb),('etc',etc),('knn',knn),

                                    ('dt',dtree),('abc',ada)], voting='hard')

                                    

  

cross_val_score(vot, train_data,train_labels, cv  = 5).mean()

from sklearn.ensemble import VotingClassifier





gbc_best = gbc_grid.best_estimator_

rf_best = rf_grid.best_estimator_

svc_best = svc_grid.best_estimator_

xgb_best = xgb_grid.best_estimator_

ada_best = ada_grid.best_estimator_

#etc_best = ext_grid.best_estimator_

#knn_best = knn_grid.best_estimator_

#dtree_best = dtree_grid.best_estimator_





vot  = VotingClassifier(estimators=[('gbc',gbc_best),('rf', rf_best),('svc',svc_best),('xgb',xgb_best),

                                    ('abc',ada_best)], voting='hard')

                                    #('etc',etc_best),('knn',knn_best),('dt',dtree_best),

  

print(cross_val_score(vot, train_data,train_labels, cv  = 5).mean())

from sklearn.ensemble import VotingClassifier



#mlp = MLPClassifier()

#knn = KNeighborsClassifier()

#dtree = DecisionTreeClassifier()

#lr = LogisticRegression()

#naive = GaussianNB()

#etc = ExtraTreesClassifier()



svc = SVC()

rfc = RandomForestClassifier(random_state=99)

xgb = XGBClassifier()

gbc = GradientBoostingClassifier(random_state = 99)

abc =  AdaBoostClassifier(random_state = 99)





vot  = VotingClassifier(estimators=[('gbc',gbc),('svc',svc),('rf', rfc)],voting = 'soft')

    #('etc',etc),('gbc',gbc),('rf', rfc),('svc',svc),('xgb',xgb),('xgb',xgb),('dt',dtree)

  



params = {'gbc__n_estimators': [50], 'gbc__min_samples_split': [3],

          'svc__C': [10, 100] , 'svc__gamma': [0.1,0.01] , 'svc__kernel': ['rbf'] , 'svc__probability': [True],  

          'rf__max_features': [2,3,6], 'rf__max_depth': [7], 'rf__min_samples_split': [3] } 



votingclf_grid = GridSearchCV(estimator = vot, param_grid = params, cv = 10)

votingclf_grid.fit(train_data,train_labels)

print(votingclf_grid.best_score_)

print(votingclf_grid.best_params_)
votingclf_grid.fit(train_data,train_labels)

predicted = votingclf_grid.predict(test_data)



testData = pd.read_csv('../input/test.csv')

passengerId = testData['PassengerId']



predicted_df = pd.DataFrame({'PassengerId':passengerId,'Survived':predicted})



predicted_df.to_csv('VC_GS_Submission.csv',index = False)