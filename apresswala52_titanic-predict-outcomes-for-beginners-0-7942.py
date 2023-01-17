# importing libraries

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from mlxtend.classifier import StackingClassifier

from sklearn.metrics import confusion_matrix

from xgboost import XGBClassifier

from sklearn.preprocessing import normalize



from keras.models import Sequential 

from keras.layers import Dense, Activation, Dropout

from keras.layers.normalization import BatchNormalization
# loading the train dataset

train_df = pd.read_csv("../input/train.csv")

print("Number of data points:",train_df.shape[0])
# loading the test dataset

test_df = pd.read_csv("../input/test.csv")

print("Number of data points:",test_df.shape[0])
# combining both the datasets for EDA

titanic = pd.concat([train_df, test_df], sort=False)
titanic.head()
titanic.info()
titanic.describe()
titanic.isnull().sum()
# replacing the missing values of the Cabin column with 'unknown'

titanic.Cabin = titanic.Cabin.fillna("unknown")
# replacing the missing value of Embarked with the mode of the column

titanic.Embarked = titanic.Embarked.fillna(titanic['Embarked'].mode()[0])
# replacing the missing value of Fare with the mean of the column

titanic.Fare = titanic.Fare.fillna(titanic['Fare'].mean())
#using the title column to fill the age column

titanic['title']=titanic.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())
newtitles={

    "Capt":       "Officer",

    "Col":        "Officer",

    "Major":      "Officer",

    "Jonkheer":   "Royalty",

    "Don":        "Royalty",

    "Sir" :       "Royalty",

    "Dr":         "Officer",

    "Rev":        "Officer",

    "the Countess":"Royalty",

    "Dona":       "Royalty",

    "Mme":        "Mrs",

    "Mlle":       "Miss",

    "Ms":         "Mrs",

    "Mr" :        "Mr",

    "Mrs" :       "Mrs",

    "Miss" :      "Miss",

    "Master" :    "Master",

    "Lady" :      "Royalty"}
titanic['title']=titanic.title.map(newtitles)
titanic.groupby(['title','Sex']).Age.mean()
def newage (cols):

    title=cols[0]

    Sex=cols[1]

    Age=cols[2]

    if pd.isnull(Age):

        if title=='Master' and Sex=="male":

            return 4.57

        elif title=='Miss' and Sex=='female':

            return 21.8

        elif title=='Mr' and Sex=='male': 

            return 32.37

        elif title=='Mrs' and Sex=='female':

            return 35.72

        elif title=='Officer' and Sex=='female':

            return 49

        elif title=='Officer' and Sex=='male':

            return 46.56

        elif title=='Royalty' and Sex=='female':

            return 40.50

        else:

            return 42.33

    else:

        return Age
titanic.Age=titanic[['title','Sex','Age']].apply(newage, axis=1)
titanic.groupby('Survived')['PassengerId'].count().plot.bar()
# from the above plot

print('Passengers that survived {} %'.format(round(titanic['Survived'].mean()*100,2)))

print('Passengers that did not survive {} %'.format(100 - round(titanic['Survived'].mean()*100,2)))
corr = train_df.corr()

sns.heatmap(corr, cbar=True, annot=True, square=True, 

            fmt='.2f', annot_kws={'size': 10}, 

            yticklabels=corr.columns.values, xticklabels=corr.columns.values)
n = titanic.shape[0]

sns.pairplot(titanic[['Pclass', 'Fare', 'Age', 'SibSp', 'Parch', 'Survived']][0:n], hue='Survived', 

             vars=['Pclass', 'Fare', 'Age', 'SibSp', 'Parch'])

plt.show()
plt.figure(figsize=[12,10])

plt.subplot(2,2,1)

sns.barplot('Sex','Survived',data=train_df)

plt.subplot(2,2,2)

sns.barplot('Embarked','Survived',data=train_df)
# taking the count of characcters in Name

titanic['Name1'] = titanic.Name.apply(lambda x:len(x))
titanic['HasCabin'] = titanic['Cabin'].apply(lambda x:0 if x=='unknown' else 1)
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch']
titanic['IsAlone'] = titanic['FamilySize'].apply(lambda x:1 if x==0 else 0)
titanic = titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],axis=1)
# performing one hot encoding of the categorical features

titanic = pd.get_dummies(titanic)
# now the dataset is ready, ML model can be trained on it

titanic.head()
corr = titanic.corr()

fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(corr, cbar=True, annot=True, square=True, 

            fmt='.2f', annot_kws={'size': 8},

            yticklabels=corr.columns.values, xticklabels=corr.columns.values)
train_len = train_df.shape[0]

train=titanic[:train_len]

test=titanic[train_len:]
# changing the type of the class label from float to int

train.Survived=train.Survived.astype('int')

train.Survived.dtype
X_train = train.drop("Survived",axis=1)

y_train = train['Survived']

X_test = test.drop("Survived", axis=1)
# normalizing the train & test dataset

X_train = normalize(X_train)

X_test = normalize(X_test)
print(X_train.shape)

print(X_test.shape)
# initializing Logistic Regression model with L2 regularisation

lr = LogisticRegression(penalty='l2')



# C values we need to try on classifier

C = [1000,500,100,50,10,5,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]

param_grid = {'C':C}



# using GridSearchCV to find the optimal value of C

# using roc_auc as the scoring parameter & applying 10 fold CV

gscv = GridSearchCV(lr,param_grid,scoring='accuracy',cv=10,return_train_score=True)



gscv.fit(X_train,y_train)



print("Best C Value: ",gscv.best_params_)

print("Best Accuracy: %.5f"%(gscv.best_score_))
# determining optimal C

optimal_C = gscv.best_params_['C']



#training the model using the optimal C

lrf = LogisticRegression(penalty='l2', C=optimal_C)

lrf.fit(X_train,y_train)



#predicting the class label using test data 

y_pred = lrf.predict(X_test)
# confusion matrix on train data

y_predict = lrf.predict(X_train)

cm = confusion_matrix(y_train, y_predict)

sns.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='g')
output=pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':y_pred})

output.to_csv('submission.csv', index=False)
# initializing Linear SVM model with L1 regularisation

svm = SGDClassifier(loss='hinge', penalty='l1')



# C values we need to try on classifier

alpha_values = [1000,500,100,50,10,5,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.00001]

param_grid = {'alpha':alpha_values}



# using GridSearchCV to find the optimal value of alpha

# using roc_auc as the scoring parameter & applying 10 fold CV

gscv = GridSearchCV(svm,param_grid,scoring='accuracy',cv=10,return_train_score=True)



gscv.fit(X_train,y_train)



print("Best alpha Value: ",gscv.best_params_)

print("Best Accuracy: %.5f"%(gscv.best_score_))
# determining optimal alpha

optimal_alpha = gscv.best_params_['alpha']



#training the model using the optimal alpha

svm = SGDClassifier(loss='hinge', penalty='l1', alpha=optimal_alpha)

svm.fit(X_train,y_train)



#predicting the class label using test data 

y_pred = svm.predict(X_test)
# confusion matrix on train data

y_predict = svm.predict(X_train)

cm = confusion_matrix(y_train, y_predict)

sns.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='g')
output=pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':y_pred})

output.to_csv('submission.csv', index=False)
# initializing KNN model 

knn = KNeighborsClassifier(weights='uniform')



# C values we need to try on classifier

neighbors = [5, 7, 9, 11, 15, 21, 25, 31, 35, 41, 47, 52]

param_grid = {'n_neighbors':neighbors}



# using GridSearchCV to find the optimal value of k

# using roc_auc as the scoring parameter & applying 10 fold CV

gscv = GridSearchCV(knn,param_grid,scoring='accuracy',cv=10,return_train_score=True)



gscv.fit(X_train,y_train)



print("Best k Value: ",gscv.best_params_)

print("Best Accuracy: %.5f"%(gscv.best_score_))
# determining optimal neighbors

optimal_k = gscv.best_params_['n_neighbors']



#training the model using the optimal k

knn = KNeighborsClassifier(n_neighbors=optimal_k, weights='uniform')

knn.fit(X_train,y_train)



#predicting the class label using test data 

y_pred = knn.predict(X_test)
# confusion matrix on train data

y_predict = knn.predict(X_train)

cm = confusion_matrix(y_train, y_predict)

sns.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='g')
output=pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':y_pred})

output.to_csv('submission.csv', index=False)
# initializing DT model 

dt = DecisionTreeClassifier(class_weight='balanced')



# max_depth values we need to try on classifier

depth = [5, 7, 9, 11, 15, 21, 25, 31, 35, 41, 47, 52]

param_grid = {'max_depth':depth}



# using GridSearchCV to find the optimal value of k

# using roc_auc as the scoring parameter & applying 10 fold CV

gscv = GridSearchCV(dt,param_grid,scoring='accuracy',cv=10,return_train_score=True)



gscv.fit(X_train,y_train)



print("Best depth Value: ",gscv.best_params_)

print("Best Accuracy: %.5f"%(gscv.best_score_))
# determining optimal max_depth

optimal_depth = gscv.best_params_['max_depth']



#training the model using the optimal k

dt = DecisionTreeClassifier(max_depth=optimal_depth, class_weight='balanced')

dt.fit(X_train,y_train)



#predicting the class label using test data 

y_pred = dt.predict(X_test)
# confusion matrix on train data

y_predict = dt.predict(X_train)

cm = confusion_matrix(y_train, y_predict)

sns.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='g')
output=pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':y_pred})

output.to_csv('submission.csv', index=False)
rf=RandomForestClassifier(random_state=1)



params={'n_estimators': list(range(10,100,10)),

      'max_depth':[3,4,5,6,7,8,9,10],

      'criterion':['gini','entropy']}



gscv=GridSearchCV(estimator=rf, param_grid=params, scoring='accuracy', cv=10, return_train_score=True)

gscv.fit(X_train,y_train)

print("Best C Value: ",gscv.best_params_)

print("Best Accuracy: %.5f"%(gscv.best_score_))
#training the model using the optimal params

rf = RandomForestClassifier(random_state=1, criterion='gini', max_depth= 4, n_estimators=20)

rf.fit(X_train,y_train)



#predicting the class label using test data 

y_pred = rf.predict(X_test)
# confusion matrix on train data

y_predict = rf.predict(X_train)

cm = confusion_matrix(y_train, y_predict)

sns.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='g')
output=pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':y_pred})

output.to_csv('submission.csv', index=False)
param_grid={

    'max_depth':list(range(2,10)),

    'n_estimators':list(range(50,500,50)),

}



xgb = XGBClassifier(objective='binary:logistic')

rscv = GridSearchCV(xgb, param_grid, scoring='accuracy', n_jobs=-1, return_train_score=True)

rscv.fit(X_train, y_train)

print("Best Max_Depth:",rscv.best_params_['max_depth'])

print("Best N estimators:",rscv.best_params_['n_estimators'])

print("Best Accuracy: %.5f"%(rscv.best_score_))
#training the model using the optimal params

xgb = XGBClassifier(learning_rate=0.2, max_depth=4, n_estimators=150)

xgb.fit(X_train,y_train)



#predicting the class label using test data 

y_pred = xgb.predict(X_test)
# confusion matrix on train data

y_predict = xgb.predict(X_train)

cm = confusion_matrix(y_train, y_predict)

sns.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='g')
features = X_train.columns

importances = xgb.feature_importances_

indices = (np.argsort(importances))

plt.figure(figsize=(8,8))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='r', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
output=pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':y_pred})

output.to_csv('submission.csv', index=False)
clf1 = DecisionTreeClassifier(max_depth=5, class_weight='balanced')

clf2 = LogisticRegression(penalty='l2',C=10)

clf3 = XGBClassifier(learning_rate=0.2, max_depth=4, n_estimators=150)

rf = RandomForestClassifier(criterion='entropy', max_depth= 9, n_estimators=90)



sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], 

                          meta_classifier=rf)



sclf.fit(X_train, y_train)
#predicting the class label using test data 

y_pred = sclf.predict(X_test)
# confusion matrix on train data

y_predict = sclf.predict(X_train)

cm = confusion_matrix(y_train, y_predict)

sns.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='g')
output=pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':y_pred})

output.to_csv('submission.csv', index=False)
from keras.utils import to_categorical

y_train_new = to_categorical(y_train)
# using He Normalization for weights initialization

model = Sequential()



# Layer 1 - 64 neurons

model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))

model.add(BatchNormalization())



# Layer 2 - 32 neurons

model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))

model.add(BatchNormalization())



model.add(Dense(2, activation='softmax'))



model.summary()



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



history = model.fit(X_train, y_train_new, batch_size=128, epochs=10, verbose=1)
y_pred = model.predict(X_test)

y_classes = y_pred.argmax(axis=-1)
output=pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':y_classes})

output.to_csv('submission.csv', index=False)
from prettytable import PrettyTable 

x = PrettyTable()

x.field_names = ["Model", "Train Accuracy(%)", "Test Accuracy(%)"]

x.add_row(['Logistic Regression', '82.49', '77.51'])

x.add_row(['Support Vector Machines', '77.44', '72.24'])

x.add_row(['K Nearest Neighbor', '72.84', '70.34'])

x.add_row(['Decision Tree', '80.80', '76.55'])

x.add_row(['Random Forest', '83.95', '79.42'])

x.add_row(['XGBoost', '83.61', '75.11'])

x.add_row(['Stacking Classifier', '-', '73.68'])

x.add_row(['Neural Networks', '81.59', ''])

print(x)