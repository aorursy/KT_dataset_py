import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
train_df = pd.read_csv("../input/train.csv")
train_df.head()
test_df = pd.read_csv("../input/test.csv")
test_df.head()
print(train_df.shape)
print(test_df.shape)
comb_fc = train_df[["Fare","Pclass","Embarked"]].groupby(["Pclass",'Embarked']).describe()
comb_fc
%matplotlib inline
import seaborn
seaborn.set() 

survived_class = train_df[train_df['Survived']==1]['Pclass'].value_counts()
dead_class = train_df[train_df['Survived']==0]['Pclass'].value_counts()
df_class = pd.DataFrame([survived_class,dead_class])
df_class.index = ['Survived','Died']
df_class.plot(kind='bar',stacked=True, figsize=(10,5), title="Survived/Died by Class")

from IPython.display import display
display(df_class)
Survived = train_df[train_df.Survived == 1]['Sex'].value_counts()
Died = train_df[train_df.Survived == 0]['Sex'].value_counts()
df_sex = pd.DataFrame([Survived , Died])
df_sex.index = ['Survived','Died']
df_sex.plot(kind='bar',stacked=True, figsize=(10,5), title="Survived/Died by Sex")

from IPython.display import display
display(df_sex) 
survived_embark = train_df[train_df['Survived']==1]['Embarked'].value_counts()
dead_embark = train_df[train_df['Survived']==0]['Embarked'].value_counts()
df_embark = pd.DataFrame([survived_embark,dead_embark])
df_embark.index = ['Survived','Died']
df_embark.plot(kind='bar',stacked=True, figsize=(10,6))

from IPython.display import display
display(df_embark)
train_df.isnull().sum()
test_df.isnull().sum()
train_df.corr()
ticket = train_df.pop("Ticket")
ticket_test = test_df.pop("Ticket")
name = train_df.pop("Name")
name_test =test_df.pop("Name")
q = test_df['Fare'].isnull()
test_df["Fare"][q]
test_df[152:153]
test_df["Fare"][152] = float(14.644083)
test_df["Fare"][152]
test_df.isnull().sum()
cabin_test = test_df.pop("Cabin")
test_df.isnull().sum()
cabin = train_df.pop('Cabin')
train_df.isnull().sum()
p = train_df['Embarked'][train_df['Embarked'].isnull()]
p
train_df[61:62]
train_df["Embarked"][61] = str('S')
train_df[829:830]
train_df["Embarked"][829] = str('S')
train_df["Embarked"][829]
train_df.isnull().sum()
train_labels = train_df.pop("Survived")
train_df.head()
test_df.head()
def modify_age (df):
    df.Age = df.Age.fillna(-0.5)
    bins =(-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_label = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    container = pd.cut(df.Age,bins, labels = group_label )
    df.Age = container
    return df
def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df
def transform_features(df):
    df = modify_age(df)
    df = simplify_fares(df)
    return df
train_df = transform_features(train_df)
test_df = transform_features(test_df)
from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = ['Sex', 'Embarked',"Age","Fare"]
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
train_data, test_data = encode_features(train_df,test_df)
train_data.head()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(train_data, train_labels,random_state=246, test_size=0.2)

from sklearn import svm
def fit_classifier(C_value=1.0):
    clf = svm.LinearSVC(C=C_value, loss='hinge')
    clf.fit(x_train,y_train)
    ## Get predictions on training data
    train_preds = clf.predict(x_train)
    train_error = float(np.sum((train_preds > 0.0) != (y_train > 0.0)))/len(y_train)
    ## Get predictions on test data
    test_preds = clf.predict(x_test)
    test_error = float(np.sum((test_preds > 0.0) != (y_test > 0.0)))/len(y_test)
    ##
    return train_error,test_error
fit_classifier(C_value=1)
def fit_classifier(C_value=1.0,gammaS=.1):
    clf = svm.SVC(C=C_value,kernel ="rbf",gamma=gammaS)
    clf.fit(x_train,y_train)
    ## Get predictions on training data
    train_preds = clf.predict(x_train)
    train_error = float(np.sum((train_preds > 0.0) != (y_train > 0.0)))/len(y_train)
    ## Get predictions on test data
    test_preds = clf.predict(x_test)
    test_error = float(np.sum((test_preds > 0.0) != (y_test > 0.0)))/len(y_test)
    ##
    return train_error,test_error
fit_classifier(C_value=2,gammaS = "auto")
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score

clf2=RandomForestClassifier(n_estimators = 10, criterion ="gini",max_features = "log2",max_depth =10 ,min_samples_split =5,min_samples_leaf=5)
def run_kfold(clf2):
    kf = KFold(891, n_folds=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        X_train, X_test = train_data.values[train_index], train_data.values[test_index]
        y_train, y_test = train_labels.values[train_index], train_labels.values[test_index]
        clf2.fit(X_train, y_train)
        predictions = clf2.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))     
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome)) 

run_kfold(clf2)
run_kfold(svm.LinearSVC(C=.1, loss='hinge'))
run_kfold(svm.SVC(C=2,kernel ="rbf",gamma='auto'))
predictions = clf2.predict(test_data)
ids = test_data['PassengerId']
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('titanic-predictions.csv', index = False)
output.head()
test_data.head()
