# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# for handling data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# for visualisation
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# for machine learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.cross_validation import KFold
from sklearn import preprocessing

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.

# import data
df_train=pd.read_csv('../input/train.csv',sep=',')
df_test=pd.read_csv('../input/test.csv',sep=',')
df_data = df_train.append(df_test) # The entire data: train + test.

# exporting the submission
PassengerId = df_test['PassengerId']
Submission=pd.DataFrame()
Submission['PassengerId'] = df_test['PassengerId']
print('Components imported')
#check for any other unusable values
print(pd.isnull(df_data).sum())
# Get a statistical overview of the training data
df_train.describe()
# Get a statistical overview of the training data
df_test.describe()
# Get title
df_data["Title"] = df_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#Unify common titles. 
df_data["Title"] = df_data["Title"].replace('Mlle', 'Miss')
df_data["Title"] = df_data["Title"].replace('Master', 'Master')
df_data["Title"] = df_data["Title"].replace(['Mme', 'Dona', 'Ms'], 'Mrs')
df_data["Title"] = df_data["Title"].replace(['Jonkheer','Don'],'Mr')
df_data["Title"] = df_data["Title"].replace(['Capt','Major', 'Col','Rev','Dr'], 'Millitary')
df_data["Title"] = df_data["Title"].replace(['Lady', 'Countess','Sir'], 'Honor')

# Age in df_train and df_test:
df_train["Title"] = df_data['Title'][:891]
df_test["Title"] = df_data['Title'][891:]

# convert Title categories to Columns
titledummies=pd.get_dummies(df_train[['Title']], prefix_sep='_') #Title
df_train = pd.concat([df_train, titledummies], axis=1) 
ttitledummies=pd.get_dummies(df_test[['Title']], prefix_sep='_') #Title
df_test = pd.concat([df_test, ttitledummies], axis=1) 
print('Title Feature created')
#Fill the na values in Fare
df_data["Embarked"]=df_data["Embarked"].fillna('S')
df_train["Embarked"] = df_data['Embarked'][:891]
df_test["Embarked"] = df_data['Embarked'][891:]
print('Missing Embarkations Added')
# convert Embarked categories to Columns
dummies=pd.get_dummies(df_train[["Embarked"]], prefix_sep='_') #Embarked
df_train = pd.concat([df_train, dummies], axis=1) 
dummies=pd.get_dummies(df_test[["Embarked"]], prefix_sep='_') #Embarked
df_test = pd.concat([df_test, dummies], axis=1)
print("Embarked Feature created")
# Fill the na values in Fare based on average fare
df_data["Fare"]=df_data["Fare"].fillna(np.median(df_data["Fare"]))
df_train["Fare"] = df_data["Fare"][:891]
df_test["Fare"] = df_data["Fare"][891:]
print('Estimate missing Fare')
Pclass = [1,2,3]
for aclass in Pclass:
    fare_to_impute = df_data.groupby('Pclass')['Fare'].median()[aclass]
    df_data.loc[(df_data['Fare'].isnull()) & (df_data['Pclass'] == aclass), 'Fare'] = fare_to_impute
        
df_train["Fare"] = df_data["Fare"][:891]
df_test["Fare"] = df_data["Fare"][891:]        

#map Fare values into groups of numerical values
df_train["FareBand"] = pd.qcut(df_train['Fare'], 4, labels = [1, 2, 3, 4]).astype('category')
df_test["FareBand"] = pd.qcut(df_test['Fare'], 4, labels = [1, 2, 3, 4]).astype('category')

# convert FareBand categories to Columns
dummies=pd.get_dummies(df_train[["FareBand"]], prefix_sep='_') #Embarked
df_train = pd.concat([df_train, dummies], axis=1) 
dummies=pd.get_dummies(df_test[["FareBand"]], prefix_sep='_') #Embarked
df_test = pd.concat([df_test, dummies], axis=1)
print("Fareband categories created")
titles = ['Master', 'Miss', 'Mr', 'Mrs', 'Millitary','Honor']
for title in titles:
    age_to_impute = df_data.groupby('Title')['Age'].median()[title]
    df_data.loc[(df_data['Age'].isnull()) & (df_data['Title'] == title), 'Age'] = age_to_impute
# Age in df_train and df_test:
df_train["Age"] = df_data['Age'][:891]
df_test["Age"] = df_data['Age'][891:]
print('Missing Ages Estimated')
df_train["Pclass"]=df_train["Pclass"].astype('category')
df_test["Pclass"]=df_test["Pclass"].astype('category')
# convert Pclass categories to Columns
dummies=pd.get_dummies(df_train[["Pclass"]], prefix_sep='_') #Embarked
df_train = pd.concat([df_train, dummies], axis=1) 
dummies=pd.get_dummies(df_test[["Pclass"]], prefix_sep='_') #Embarked
df_test = pd.concat([df_test, dummies], axis=1)
print("Pclass Feature created")
# sort Age into band categories
bins = [0,12,24,45,60,np.inf]
labels = ['Child', 'Young Adult', 'Adult','Older Adult','Senior']
df_train["AgeGroup"] = pd.cut(df_train["Age"], bins, labels = labels)
df_test["AgeGroup"] = pd.cut(df_test["Age"], bins, labels = labels)
print('Age Feature created')

# convert AgeGroup categories to Columns
dummies=pd.get_dummies(df_train[["AgeGroup"]], prefix_sep='_') #Embarked
df_train = pd.concat([df_train, dummies], axis=1) 
dummies=pd.get_dummies(df_test[["AgeGroup"]], prefix_sep='_') #Embarked
df_test = pd.concat([df_test, dummies], axis=1)
print("AgeGroup categories created")
# convert categories to Columns
dummies=pd.get_dummies(df_train[['Sex']], prefix_sep='_') #Gender
df_train = pd.concat([df_train, dummies], axis=1) 
testdummies=pd.get_dummies(df_test[['Sex']], prefix_sep='_') #Gender
df_test = pd.concat([df_test, testdummies], axis=1)
print('Gender Categories created')
# People with parents or siblings
df_data["Alone"] = np.where(df_data['SibSp'] + df_data['Parch'] + 1 == 1, 1,0) # People travelling alone
# Age in df_train and df_test:
df_train["Alone"] = df_data['Alone'][:891]
df_test["Alone"] = df_data['Alone'][891:]
print('Lone Traveller feature created')
# get last name
df_data["Last_Name"] = df_data['Name'].apply(lambda x: str.split(x, ",")[0])
# Set survival value
DEFAULT_SURVIVAL_VALUE = 0.5
df_data["Family_Survival"] = DEFAULT_SURVIVAL_VALUE

# Find Family groups by Fare
for grp, grp_df in df_data[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                df_data.loc[df_data['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                df_data.loc[df_data['PassengerId'] == passID, 'Family_Survival'] = 0

print("Number of passengers with family survival information:", 
      df_data.loc[df_data['Family_Survival']!=0.5].shape[0])

# Find Family groups by Ticket
for _, grp_df in df_data.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    df_data.loc[df_data['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    df_data.loc[df_data['PassengerId'] == passID, 'Family_Survival'] = 0
                        
print("Number of passenger with family/group survival information: " 
      +str(df_data[df_data['Family_Survival']!=0.5].shape[0]))

# Family_Survival in df_train and df_test:
df_train["Family_Survival"] = df_data['Family_Survival'][:891]
df_test["Family_Survival"] = df_data['Family_Survival'][891:]
# check if cabin inf exists
df_data["HadCabin"] = (df_data["Cabin"].notnull().astype('int'))
# split Embanked into df_train and df_test:
df_train["HadCabin"] = df_data["HadCabin"][:891]
df_test["HadCabin"] = df_data["HadCabin"][891:]
print('HasCabin feature created')
#Map and Create Deck feature for training
df_data["Deck"] = df_data.Cabin.str.extract('([A-Za-z])', expand=False)
deck_mapping = {"0":0,"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
df_data['Deck'] = df_data['Deck'].map(deck_mapping)
df_data["Deck"] = df_data["Deck"].fillna("0")
df_data["Deck"]=df_data["Deck"].astype('int')

df_train["Deck"] = df_data['Deck'][:891]
df_test["Deck"] = df_data['Deck'][891:]
print('Deck feature created')

# convert categories to Columns
dummies=pd.get_dummies(df_train[['Deck']], prefix_sep='_') #Gender
df_train = pd.concat([df_train, dummies], axis=1) 
dummies=pd.get_dummies(df_test[['Deck']], prefix_sep='_') #Gender
df_test = pd.concat([df_test,dummies], axis=1)
print('Deck Categories created')
# convert SibSp categories to Columns
(df_train['SibSp'])=(df_train['SibSp']).astype('category')
dummies=pd.get_dummies(df_train[['SibSp']], prefix_sep='_') #Gender
df_train = pd.concat([df_train, dummies], axis=1) 
(df_test['SibSp'])=(df_test['SibSp']).astype('category')
dummies=pd.get_dummies(df_test[['SibSp']], prefix_sep='_') #Gender
df_test = pd.concat([df_test,dummies], axis=1)
print('Sibsp Categories created')
# convert SibSp categories to Columns
(df_train['Parch'])=(df_train['Parch']).astype('category')
dummies=pd.get_dummies(df_train[['Parch']], prefix_sep='_') #Gender
df_train = pd.concat([df_train, dummies], axis=1) 
(df_test['Parch'])=(df_test['Parch']).astype('category')
dummies=pd.get_dummies(df_test[['Parch']], prefix_sep='_') #Gender
df_test = pd.concat([df_test,dummies], axis=1)
print('Parch Categories created')
df_data=df_data.drop(['Cabin','Embarked','Title','Age','Sex','Name','Ticket','Deck','Fare'], axis=1)
df_train=df_train.drop(['Cabin','Embarked','Title','Age','Sex','Name','Ticket','AgeGroup','Deck','Pclass','Fare','FareBand','SibSp','Parch','Parch_7','Parch_8','Parch_9'], axis=1)
df_test=df_test.drop(['Cabin','Embarked','Title','Age','Sex','Name','Ticket','AgeGroup','Deck','Pclass','Fare','FareBand','SibSp','Parch','Parch_7','Parch_8','Parch_9'], axis=1)
print('None Numeric Columns droped')
#check for any other unusable values
print(pd.isnull(df_train).sum())
#check for any other unusable values
print(pd.isnull(df_test).sum())
df_train.head()
df_train.describe()
df_test.columns
# define columns to be used
NUMERIC_COLUMNS=['Alone','Family Size','Sex','Pclass','Fare','FareBand','Age','TitleCat','Embarked'] #72
ORIGINAL_NUMERIC_COLUMNS=['Pclass','Age','SibSp','Parch','Sex_female','Sex_male','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','Embarked'] #83
REVISED_NUMERIC_COLUMNS=['Title_Master', 'Title_Millitary',
       'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Embarked_C', 'Embarked_Q',
       'Embarked_S', 'FareBand_1', 'FareBand_2', 'FareBand_3', 'FareBand_4',
       'Pclass_1', 'Pclass_2', 'Pclass_3', 'AgeGroup_Child',
       'AgeGroup_Young Adult', 'AgeGroup_Adult', 'AgeGroup_Older Adult',
       'AgeGroup_Senior', 'Sex_female', 'Sex_male', 'Alone', 'Family_Survival',
       'HadCabin','SibSp_0', 'SibSp_1', 'SibSp_2',
       'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1',
       'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', ]

data_to_train = df_train[REVISED_NUMERIC_COLUMNS].fillna(-1000)

# create test and training data
y=df_train['Survived']
X=data_to_train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=21, stratify=y)
print('Model Split')
print(df_test.shape)
print(X.shape)
#df_training=df_test.drop(['AgeGroup_Baby'], axis=1)
#df_test=df_test.drop(['AgeGroup_Baby'], axis=1)
clf = SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc_clf = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_clf)
test = df_test[REVISED_NUMERIC_COLUMNS].fillna(-1000)
Submission['Survived']=clf.predict(test)
Submission.set_index('PassengerId', inplace=True)
# write data frame to csv file
Submission.to_csv('baselinemodel01.csv',sep=',')
print('Submission Created')
from sklearn.model_selection import train_test_split
REVISED_NUMERIC_COLUMNS=['Title_Master', 'Title_Millitary',
       'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Embarked_C', 'Embarked_Q',
       'Embarked_S', 'FareBand_1', 'FareBand_2', 'FareBand_3', 'FareBand_4',
       'Pclass_1', 'Pclass_2', 'Pclass_3', 'AgeGroup_Child',
       'AgeGroup_Young Adult', 'AgeGroup_Adult', 'AgeGroup_Older Adult',
       'AgeGroup_Senior', 'Sex_female', 'Sex_male', 'Alone', 'Family_Survival',
       'HadCabin','SibSp_0', 'SibSp_1', 'SibSp_2',
       'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1',
       'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', ]# create test and training data
predictors = df_train.drop(['Survived', 'PassengerId'], axis=1)
data_to_train = df_train[REVISED_NUMERIC_COLUMNS].fillna(-1000)
X=data_to_train
y = df_train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(data_to_train, y, test_size = 0.3,random_state=21, stratify=y)
print('Data Split')
# DecisionTree with RandomizedSearch

# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": np.arange(1, 6),
              "max_features": np.arange(1, 10),
              "min_samples_leaf": np.arange(1, 6),
              "criterion": ["gini","entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=30)

# Fit it to the data
tree_cv.fit(X,y)
y_pred = tree_cv.predict(x_val)
# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))
acc_tree_cv = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_tree_cv)
# Select columns
test = df_test[REVISED_NUMERIC_COLUMNS].fillna(-1000)
# select classifier
tree = DecisionTreeClassifier(max_depth=5,max_features=7,min_samples_leaf=1,criterion="entropy")

# train model
tree.fit(X,y)
# make predictions
Submission['Survived']=tree.predict(test)
print(Submission.head(5))
#Submission.set_index('PassengerId', inplace=True)
Submission.to_csv('Tunedtree1submission.csv',sep=',')
print("Submission Submitted")
from sklearn.model_selection import train_test_split
REVISED_NUMERIC_COLUMNS=['Title_Master', 'Title_Millitary',
       'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Embarked_C', 'Embarked_Q',
       'Embarked_S', 'FareBand_1', 'FareBand_2', 'FareBand_3', 'FareBand_4',
       'Pclass_1', 'Pclass_2', 'Pclass_3', 'AgeGroup_Child',
       'AgeGroup_Young Adult', 'AgeGroup_Adult', 'AgeGroup_Older Adult',
       'AgeGroup_Senior', 'Sex_female', 'Sex_male', 'Alone', 'Family_Survival',
       'HadCabin','SibSp_0', 'SibSp_1', 'SibSp_2',
       'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1',
       'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', ]
# create test and training data
predictors = df_train.drop(['Survived', 'PassengerId'], axis=1)
data_to_train = df_train[REVISED_NUMERIC_COLUMNS].fillna(-1000)
data_to_predict=df_test[REVISED_NUMERIC_COLUMNS].fillna(-1000)
y=df_train['Survived']
X=data_to_train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=21, stratify=y)
print('Data split')
# Import necessary modules

from __future__ import print_function
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical
print('Modules imported')
print('Training Data shape')
print(df_train.shape)
print(df_train.shape)
print('Test Data shape')
print(df_test.shape)

print(df_train.head())
print(df_test.head())
# create model
model = Sequential()
model.add(Dense(units=56, input_dim=X.shape[1], activation='selu'))
model.add(Dropout(0.5))
model.add(Dense(units=27, activation='selu')) 
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='tanh'))

# choose loss function and optimizing method
model.compile(loss='mse', optimizer='sgd')

print('Keras Model Created')
model.fit(X.values, y.values, epochs=500, verbose=0)
print('Keras model fitted')
print(df_test.columns)
print(X.columns)

df_test=df_test.set_index('PassengerId')
p_survived = model.predict_classes(df_test)
print('Prediction Completed')
submission = pd.DataFrame()

submission['PassengerId'] = df_test.index
submission['Survived'] = p_survived
print('predictions added to submission')
print(submission.shape)
print(submission.head(10))
submission.to_csv('DeepLearning03.csv', index=False)
print('csv created')
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
seed=70

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, y):
    # create model
    model = Sequential()
    model.add(Dense(54, input_dim=X.shape[1], activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(54, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(54, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='tanh'))
    # Compile model
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    # Fit the model
    model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)
    # evaluate the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
#df_test=df_test.set_index('PassengerId')
p_survived = model.predict_classes(df_test)
print('Prediction Completed')
submission = pd.DataFrame()

submission['PassengerId'] = df_test.index
submission['Survived'] = p_survived
print('predictions added to submission')
print(submission.shape)
print(submission.head(10))
submission.to_csv('OptimisedDeepLearning04.csv', index=False)
print('csv created')
